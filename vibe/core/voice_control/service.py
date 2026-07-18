from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum, auto
import time
from typing import TYPE_CHECKING

from vibe.core.logger import logger
from vibe.core.voice_control.wake_word import (
    WakeState,
    WakeWordMachine,
    Woke,
    is_cancel_phrase,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from vibe.core.voice_control.ambient_listener import AmbientListener
    from vibe.core.voice_control.speaker import SpeakerPort

_MAX_STATUS_TEXT = 160


class VoicePhase(StrEnum):
    OFF = auto()
    LISTENING = auto()  # awake for a wake word
    AWAKE = auto()  # heard the wake word, capturing a command
    THINKING = auto()  # command handed to the orchestrator, awaiting its reply
    SPEAKING = auto()


@dataclass(frozen=True, slots=True)
class VoiceControlConfig:
    greeting: str = "How can I help you?"
    silence_timeout: float = 1.3
    poll_interval: float = 0.2
    answer_timeout: float = 30.0
    # After a greeting or a spoken reply the mic stays armed for a follow-up so
    # the conversation continues without repeating the wake word. It only sleeps
    # (requiring the wake word again) after this many seconds of real silence.
    follow_up_timeout: float = 15.0


class VoiceControlService:
    """Drives the hands-free loop: wake word -> silence-ended capture ->
    command sink -> spoken reply, plus spoken question round-trips.

    Timing is injectable (``now``) and the delta/silence handling is exposed as
    plain coroutines so the whole flow can be unit-tested without real audio.
    """

    def __init__(
        self,
        *,
        command_sink: Callable[[str], Awaitable[None]],
        speaker: SpeakerPort,
        machine: WakeWordMachine | None = None,
        config: VoiceControlConfig | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._command_sink = command_sink
        self._speaker = speaker
        self._machine = machine or WakeWordMachine()
        self._config = config or VoiceControlConfig()
        self._now = now
        self._deltas: asyncio.Queue[str] = asyncio.Queue()
        self._last_delta_at: float | None = None
        self._awake_deadline: float | None = None
        self._pending_answer: asyncio.Future[str] | None = None
        self._speak_lock = asyncio.Lock()
        self._phase = VoicePhase.OFF
        self._phase_listeners: list[Callable[[VoicePhase], None]] = []
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self.last_heard: str = ""
        self.last_spoken: str = ""

    @property
    def phase(self) -> VoicePhase:
        return self._phase

    @property
    def is_running(self) -> bool:
        return self._running

    def add_phase_listener(self, listener: Callable[[VoicePhase], None]) -> None:
        self._phase_listeners.append(listener)

    def offer_delta(self, text: str) -> None:
        self._deltas.put_nowait(text)

    async def run(self, listener: AmbientListener) -> None:
        self._running = True
        self._set_phase(VoicePhase.LISTENING)
        listener.start()
        try:
            while self._running:
                try:
                    text = await asyncio.wait_for(
                        self._deltas.get(), timeout=self._config.poll_interval
                    )
                except TimeoutError:
                    await self.poll_silence()
                    continue
                await self.handle_delta(text)
                await self.poll_silence()
        finally:
            await listener.stop()
            self._set_phase(VoicePhase.OFF)

    async def stop(self) -> None:
        self._running = False

    async def handle_delta(self, text: str) -> None:
        if self._speaker.is_speaking:
            return
        self._last_delta_at = self._now()
        events = self._machine.feed(text)
        if self._machine.state == WakeState.CAPTURING:
            self._awake_deadline = self._now() + self._config.follow_up_timeout
        for event in events:
            if isinstance(event, Woke):
                self._set_phase(VoicePhase.AWAKE)
                if not self._machine.captured:
                    await self._speak(self._config.greeting, VoicePhase.AWAKE)

    async def poll_silence(self) -> None:
        if self._speaker.is_speaking:
            return
        if self._machine.state != WakeState.CAPTURING:
            return
        if self._machine.captured:
            if (
                self._last_delta_at is not None
                and self._now() - self._last_delta_at >= self._config.silence_timeout
            ):
                utterance = self._machine.flush()
                if utterance is not None:
                    await self._dispatch(utterance.text)
            return
        # Armed but nothing captured yet: sleep after the follow-up window so a
        # quiet room falls back to requiring the wake word.
        if self._pending_answer is not None or self._awake_deadline is None:
            return
        if self._now() >= self._awake_deadline:
            self._sleep()

    async def speak(
        self, text: str, *, next_phase: VoicePhase = VoicePhase.AWAKE
    ) -> None:
        await self._speak(text, next_phase)

    async def capture_answer(
        self, prompt: str, *, timeout: float | None = None
    ) -> str | None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_answer = future
        await self._speak(prompt, VoicePhase.AWAKE)
        self._last_delta_at = self._now()
        try:
            answer = await asyncio.wait_for(
                future, timeout or self._config.answer_timeout
            )
        except TimeoutError:
            answer = None
        self._pending_answer = None
        if answer is None:
            self._sleep()
        else:
            self._set_phase(VoicePhase.THINKING)
        return answer

    async def _dispatch(self, text: str) -> None:
        answer = self._pending_answer
        if answer is not None and not answer.done():
            answer.set_result(text)
            return
        if is_cancel_phrase(text):
            self._sleep()
            return
        self.last_heard = text[:_MAX_STATUS_TEXT]
        self._set_phase(VoicePhase.THINKING)
        try:
            await self._command_sink(text)
        except Exception:
            logger.error("Voice command sink failed", exc_info=True)
            self._sleep()

    async def _speak(self, text: str, next_phase: VoicePhase) -> None:
        clean = text.strip()
        if not clean:
            self._settle_after_speech(next_phase)
            return
        self.last_spoken = clean[:_MAX_STATUS_TEXT]
        async with self._speak_lock:
            self._set_phase(VoicePhase.SPEAKING)
            try:
                await self._speaker.speak(clean)
            finally:
                self._drain_deltas()
                self._settle_after_speech(next_phase)

    def _settle_after_speech(self, next_phase: VoicePhase) -> None:
        keep_awake = self._pending_answer is not None or next_phase == VoicePhase.AWAKE
        self._last_delta_at = self._now()
        if keep_awake:
            self._machine.arm_capture()
            self._awake_deadline = self._now() + self._config.follow_up_timeout
            self._set_phase(VoicePhase.AWAKE)
            return
        self._machine.reset()
        self._awake_deadline = None
        self._set_phase(next_phase)

    def _sleep(self) -> None:
        self._machine.reset()
        self._awake_deadline = None
        self._set_phase(VoicePhase.LISTENING)

    def _drain_deltas(self) -> None:
        while not self._deltas.empty():
            try:
                self._deltas.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _set_phase(self, phase: VoicePhase) -> None:
        if phase == self._phase:
            return
        self._phase = phase
        for listener in self._phase_listeners:
            try:
                listener(phase)
            except Exception:
                logger.error("Voice phase listener failed", exc_info=True)
