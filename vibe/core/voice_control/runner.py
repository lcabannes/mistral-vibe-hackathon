from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from vibe.core.logger import logger
from vibe.core.voice_control.ambient_listener import AmbientCallbacks, AmbientListener
from vibe.core.voice_control.service import (
    VoiceControlConfig,
    VoiceControlService,
    VoicePhase,
)
from vibe.core.voice_control.speaker import Speaker
from vibe.core.voice_control.wake_word import WakeWordMachine

if TYPE_CHECKING:
    from collections.abc import Callable

    from vibe.core.config import AnyVibeConfig

_START_TIMEOUT = 30.0
_STOP_TIMEOUT = 15.0


class VoiceControlRunner:
    """Runs the voice-control service on its own event loop in a background
    thread, and exposes thread-safe controls for the (threaded) HTTP server.

    ``command_handler`` is called (off the audio loop) with each recognized
    command; it is expected to hand the text to the orchestrator.
    """

    def __init__(
        self,
        *,
        config_getter: Callable[[], AnyVibeConfig],
        command_handler: Callable[[str], None],
        service_config: VoiceControlConfig | None = None,
        phase_listener: Callable[[VoicePhase], None] | None = None,
    ) -> None:
        self._config_getter = config_getter
        self._command_handler = command_handler
        self._service_config = service_config or VoiceControlConfig()
        self._phase_listener = phase_listener
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._service: VoiceControlService | None = None
        self._speaker: Speaker | None = None
        self._run_task: asyncio.Task[None] | None = None
        self._ready = threading.Event()
        self._start_error: str | None = None

    @property
    def is_running(self) -> bool:
        service = self._service
        return service is not None and service.is_running

    @property
    def phase(self) -> VoicePhase:
        service = self._service
        return service.phase if service is not None else VoicePhase.OFF

    @property
    def last_heard(self) -> str:
        service = self._service
        return service.last_heard if service is not None else ""

    @property
    def last_spoken(self) -> str:
        service = self._service
        return service.last_spoken if service is not None else ""

    @property
    def start_error(self) -> str | None:
        return self._start_error

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._ready.clear()
            self._start_error = None
            thread = threading.Thread(
                target=self._thread_main, name="voice-control", daemon=True
            )
            self._thread = thread
            thread.start()
        if not self._ready.wait(timeout=_START_TIMEOUT) and not self._start_error:
            self._start_error = "Voice control did not start in time"
        if self._start_error:
            self._join_and_reset()
            raise RuntimeError(self._start_error)

    def stop(self) -> None:
        with self._lock:
            loop = self._loop
        if loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._shutdown(), loop).result(
                    timeout=_STOP_TIMEOUT
                )
            except Exception:
                logger.error("Voice control shutdown failed", exc_info=True)
        self._join_and_reset()

    def speak(self, text: str) -> None:
        loop = self._loop
        service = self._service
        if loop is None or service is None:
            return
        asyncio.run_coroutine_threadsafe(service.speak(text), loop)

    def ask(self, prompt: str, *, timeout: float | None = None) -> str | None:
        loop = self._loop
        service = self._service
        if loop is None or service is None:
            return None
        wait = (timeout or self._service_config.answer_timeout) + 10
        try:
            return asyncio.run_coroutine_threadsafe(
                service.capture_answer(prompt, timeout=timeout), loop
            ).result(timeout=wait)
        except Exception:
            logger.error("Voice question capture failed", exc_info=True)
            return None

    def _join_and_reset(self) -> None:
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_STOP_TIMEOUT)
        with self._lock:
            self._thread = None
            self._loop = None
            self._service = None
            self._speaker = None
            self._run_task = None

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._amain())
        except Exception as exc:
            self._start_error = str(exc)
            logger.error("Voice control loop crashed", exc_info=True)
            self._ready.set()
        finally:
            loop.close()

    async def _amain(self) -> None:
        try:
            listener, service = self._build()
        except Exception as exc:
            self._start_error = f"Audio unavailable: {exc}"
            logger.error("Voice control setup failed", exc_info=True)
            self._ready.set()
            return
        self._service = service
        if self._phase_listener is not None:
            service.add_phase_listener(self._phase_listener)
        self._run_task = asyncio.create_task(service.run(listener))
        self._ready.set()
        try:
            await self._run_task
        except asyncio.CancelledError:
            pass

    def _build(self) -> tuple[AmbientListener, VoiceControlService]:
        from vibe.core.audio_player.audio_player import AudioPlayer
        from vibe.core.audio_recorder.audio_recorder import AudioRecorder
        from vibe.core.transcribe.factory import make_transcribe_client
        from vibe.core.tts.factory import make_tts_client

        config = self._config_getter()
        transcribe_model = config.get_active_transcribe_model()
        transcribe_provider = config.get_transcribe_provider_for_model(transcribe_model)
        tts_model = config.get_active_tts_model()
        tts_provider = config.get_tts_provider_for_model(tts_model)

        speaker = Speaker(make_tts_client(tts_provider, tts_model), AudioPlayer())
        self._speaker = speaker

        async def sink(text: str) -> None:
            await asyncio.to_thread(self._command_handler, text)

        service = VoiceControlService(
            command_sink=sink,
            speaker=speaker,
            machine=WakeWordMachine(),
            config=self._service_config,
        )
        listener = AmbientListener(
            AudioRecorder(),
            lambda: make_transcribe_client(transcribe_provider, transcribe_model),
            sample_rate=transcribe_model.sample_rate,
            callbacks=AmbientCallbacks(on_delta=service.offer_delta),
        )
        return listener, service

    async def _shutdown(self) -> None:
        service = self._service
        if service is not None:
            await service.stop()
        task = self._run_task
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=5)
            except (TimeoutError, asyncio.CancelledError):
                task.cancel()
        speaker = self._speaker
        if speaker is not None:
            await speaker.close()
