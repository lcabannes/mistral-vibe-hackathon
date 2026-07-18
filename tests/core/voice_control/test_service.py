from __future__ import annotations

import asyncio

import pytest

from vibe.core.voice_control.service import (
    VoiceControlConfig,
    VoiceControlService,
    VoicePhase,
)


class FakeSpeaker:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self._speaking = False

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    def stop(self) -> None:
        self._speaking = False

    async def close(self) -> None:
        pass


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _build() -> tuple[
    VoiceControlService, FakeSpeaker, Clock, list[str], list[VoicePhase]
]:
    speaker = FakeSpeaker()
    clock = Clock()
    commands: list[str] = []
    phases: list[VoicePhase] = []

    async def sink(text: str) -> None:
        commands.append(text)

    service = VoiceControlService(
        command_sink=sink,
        speaker=speaker,
        config=VoiceControlConfig(silence_timeout=1.0),
        now=clock,
    )
    service.add_phase_listener(phases.append)
    return service, speaker, clock, commands, phases


@pytest.mark.asyncio
async def test_wake_then_command_dispatches_after_silence() -> None:
    service, speaker, clock, commands, _ = _build()

    await service.handle_delta("hello orchestrator")
    assert speaker.spoken == ["How can I help you?"]

    await service.handle_delta("start a build agent")
    assert not commands  # still capturing

    clock.advance(2.0)
    await service.poll_silence()
    assert commands == ["start a build agent"]
    assert service.phase == VoicePhase.THINKING


@pytest.mark.asyncio
async def test_command_in_same_breath_skips_greeting() -> None:
    service, speaker, clock, commands, _ = _build()

    await service.handle_delta("hi orchestrator spin up a research agent")
    assert speaker.spoken == []  # no greeting when a command already followed

    clock.advance(2.0)
    await service.poll_silence()
    assert commands == ["spin up a research agent"]


@pytest.mark.asyncio
async def test_cancel_phrase_does_not_forward() -> None:
    service, _, clock, commands, phases = _build()

    await service.handle_delta("hello agent")
    await service.handle_delta("never mind")
    clock.advance(2.0)
    await service.poll_silence()

    assert commands == []
    assert service.phase == VoicePhase.LISTENING


@pytest.mark.asyncio
async def test_no_wake_word_means_no_dispatch() -> None:
    service, _, clock, commands, _ = _build()

    await service.handle_delta("just talking about the weather")
    clock.advance(2.0)
    await service.poll_silence()
    assert commands == []


@pytest.mark.asyncio
async def test_follow_up_needs_no_wake_word() -> None:
    service, _, clock, commands, _ = _build()

    await service.handle_delta("hey orchestrator do the thing")
    clock.advance(2.0)
    await service.poll_silence()
    assert commands == ["do the thing"]

    # The orchestrator replies; the conversation stays engaged.
    await service.speak("Started a build agent.")
    assert service.phase == VoicePhase.AWAKE

    # A follow-up within the window needs no wake word.
    await service.handle_delta("now add tests")
    clock.advance(2.0)
    await service.poll_silence()
    assert commands == ["do the thing", "now add tests"]


@pytest.mark.asyncio
async def test_conversation_sleeps_after_follow_up_timeout() -> None:
    service, _, clock, commands, _ = _build()

    await service.handle_delta("hey orchestrator do the thing")
    clock.advance(2.0)
    await service.poll_silence()
    await service.speak("Done.")

    # A long silence past the follow-up window puts it back to sleep.
    clock.advance(20.0)
    await service.poll_silence()
    assert service.phase == VoicePhase.LISTENING

    # Now a bare command is ignored until the wake word is spoken again.
    await service.handle_delta("another command")
    clock.advance(2.0)
    await service.poll_silence()
    assert commands == ["do the thing"]


@pytest.mark.asyncio
async def test_pending_answer_is_not_clobbered_by_a_reply() -> None:
    service, _, clock, _, _ = _build()

    task = asyncio.create_task(service.capture_answer("Which repo?", timeout=5))
    await asyncio.sleep(0)

    # A concurrent reply tries to settle back to sleep, but a pending answer
    # must keep the mic armed so the answer is captured without a wake word.
    await service.speak("Meanwhile, thinking...", next_phase=VoicePhase.LISTENING)
    await service.handle_delta("the backend one")
    clock.advance(2.0)
    await service.poll_silence()

    assert await task == "the backend one"


@pytest.mark.asyncio
async def test_mic_is_ducked_while_speaking() -> None:
    service, speaker, _, _, _ = _build()
    speaker._speaking = True

    await service.handle_delta("hello orchestrator")
    assert speaker.spoken == []
    assert service.phase == VoicePhase.OFF


@pytest.mark.asyncio
async def test_capture_answer_round_trip() -> None:
    service, speaker, clock, _, _ = _build()

    task = asyncio.create_task(service.capture_answer("Which repository?", timeout=5))
    await asyncio.sleep(0)
    assert "Which repository?" in speaker.spoken

    await service.handle_delta("the backend one")
    clock.advance(2.0)
    await service.poll_silence()

    answer = await task
    assert answer == "the backend one"


@pytest.mark.asyncio
async def test_capture_answer_times_out() -> None:
    service, _, _, _, _ = _build()
    answer = await service.capture_answer("Anything?", timeout=0.01)
    assert answer is None


@pytest.mark.asyncio
async def test_phase_progression() -> None:
    service, _, clock, _, phases = _build()

    await service.handle_delta("hello orchestrator build me a thing")
    clock.advance(2.0)
    await service.poll_silence()

    assert VoicePhase.AWAKE in phases
    assert VoicePhase.THINKING in phases
