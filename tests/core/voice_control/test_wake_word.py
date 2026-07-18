from __future__ import annotations

from vibe.core.voice_control.wake_word import (
    Utterance,
    WakeState,
    WakeWordMachine,
    Woke,
    is_cancel_phrase,
    normalize,
)


def test_scanning_ignores_non_wake_speech() -> None:
    machine = WakeWordMachine()
    assert machine.feed("the weather is nice today") == []
    assert machine.state == WakeState.SCANNING


def test_wake_phrase_transitions_to_capturing() -> None:
    machine = WakeWordMachine()
    events = machine.feed("hello orchestrator")
    assert events == [Woke()]
    assert machine.state == WakeState.CAPTURING


def test_wake_phrase_split_across_deltas() -> None:
    machine = WakeWordMachine()
    assert machine.feed("hey") == []
    assert machine.feed("agent") == [Woke()]
    assert machine.state == WakeState.CAPTURING


def test_command_in_same_utterance_is_captured_after_wake() -> None:
    machine = WakeWordMachine()
    assert machine.feed("hi orchestrator start a build agent") == [Woke()]
    assert machine.captured == "start a build agent"
    utterance = machine.flush()
    assert utterance == Utterance(text="start a build agent")
    assert machine.state == WakeState.SCANNING


def test_command_accumulates_across_deltas_then_flushes() -> None:
    machine = WakeWordMachine()
    machine.feed("hello agent")
    machine.feed("spin up a research")
    machine.feed("worker for auth")
    assert machine.captured == "spin up a research worker for auth"
    assert machine.flush() == Utterance(text="spin up a research worker for auth")


def test_flush_while_scanning_returns_none() -> None:
    machine = WakeWordMachine()
    machine.feed("nothing to see")
    assert machine.flush() is None


def test_flush_with_empty_capture_returns_none_and_rescans() -> None:
    machine = WakeWordMachine()
    machine.feed("okay orchestrator")
    assert machine.captured == ""
    assert machine.flush() is None
    assert machine.state == WakeState.SCANNING


def test_arm_capture_skips_wake_phrase() -> None:
    machine = WakeWordMachine()
    machine.arm_capture()
    assert machine.state == WakeState.CAPTURING
    machine.feed("use the second option")
    assert machine.flush() == Utterance(text="use the second option")


def test_punctuation_and_case_are_normalized() -> None:
    machine = WakeWordMachine()
    assert machine.feed("Hello, Orchestrator!") == [Woke()]


def test_reset_returns_to_scanning() -> None:
    machine = WakeWordMachine()
    machine.feed("hey vibe do something")
    machine.reset()
    assert machine.state == WakeState.SCANNING
    assert machine.captured == ""


def test_second_wake_after_flush() -> None:
    machine = WakeWordMachine()
    machine.feed("hi agent first task")
    assert machine.flush() == Utterance(text="first task")
    assert machine.feed("hello orchestrator second task") == [Woke()]
    assert machine.flush() == Utterance(text="second task")


def test_normalize_collapses_whitespace_and_strips_apostrophes() -> None:
    assert normalize("  It's   a   TEST!  ") == "its a test"


def test_is_cancel_phrase_matches_whole_utterance_only() -> None:
    assert is_cancel_phrase("never mind")
    assert is_cancel_phrase("Stop listening.")
    assert not is_cancel_phrase("cancel that build agent")
