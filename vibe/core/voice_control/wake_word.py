from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
import re

DEFAULT_WAKE_PHRASES = (
    "hey orchestrator",
    "hi orchestrator",
    "hello orchestrator",
    "okay orchestrator",
    "hey agent",
    "hi agent",
    "hello agent",
    "okay agent",
    "hey vibe",
    "hi vibe",
    "hello vibe",
)

# Spoken all by itself while awake, these cancel the pending command instead of
# forwarding it. Kept deliberately small so they do not truncate real commands
# (e.g. "cancel that build agent"): the service only treats them as a cancel
# when they are the *entire* captured utterance.
DEFAULT_CANCEL_PHRASES = ("never mind", "nevermind", "stop listening", "forget it")

_SCANNING_BUFFER_LIMIT = 200


class WakeState(StrEnum):
    SCANNING = auto()
    CAPTURING = auto()


@dataclass(frozen=True, slots=True)
class Woke:
    pass


@dataclass(frozen=True, slots=True)
class Utterance:
    text: str


WakeEvent = Woke | Utterance


def normalize(text: str) -> str:
    lowered = text.lower().replace("'", "")
    stripped = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", stripped).strip()


def is_cancel_phrase(
    text: str, cancel_phrases: tuple[str, ...] = DEFAULT_CANCEL_PHRASES
) -> bool:
    normalized = normalize(text)
    return normalized in {normalize(p) for p in cancel_phrases}


def _find_phrase(haystack: str, phrase: str) -> tuple[int, int] | None:
    if not phrase:
        return None
    padded = f" {haystack} "
    needle = f" {phrase} "
    index = padded.find(needle)
    if index == -1:
        return None
    return index, index + len(phrase)


def _find_first(haystack: str, phrases: tuple[str, ...]) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    for phrase in phrases:
        match = _find_phrase(haystack, phrase)
        if match is not None and (best is None or match[0] < best[0]):
            best = match
    return best


class WakeWordMachine:
    """Turns a stream of transcript deltas into wake / utterance events.

    Activation is a pure string match on the running transcript (there is no
    wake-word model). Once awake, the machine simply accumulates the command;
    end-of-utterance is decided by the caller via ``flush()`` (driven by a
    silence gap), rather than by an explicit submit phrase.
    """

    def __init__(self, wake_phrases: tuple[str, ...] = DEFAULT_WAKE_PHRASES) -> None:
        self._wake = tuple(normalize(p) for p in wake_phrases)
        self._state = WakeState.SCANNING
        self._buffer = ""

    @property
    def state(self) -> WakeState:
        return self._state

    @property
    def captured(self) -> str:
        return self._buffer if self._state == WakeState.CAPTURING else ""

    def reset(self) -> None:
        self._state = WakeState.SCANNING
        self._buffer = ""

    def arm_capture(self) -> None:
        # Force capture mode without a wake phrase (used when an answer is expected).
        self._state = WakeState.CAPTURING
        self._buffer = ""

    def feed(self, delta: str) -> list[WakeEvent]:
        normalized = normalize(delta)
        if normalized:
            self._buffer = f"{self._buffer} {normalized}".strip()
        if self._state == WakeState.SCANNING:
            return self._advance_scanning()
        return []

    def flush(self) -> Utterance | None:
        if self._state != WakeState.CAPTURING:
            return None
        message = self._buffer.strip()
        self._buffer = ""
        self._state = WakeState.SCANNING
        if not message:
            return None
        return Utterance(text=message)

    def _advance_scanning(self) -> list[WakeEvent]:
        match = _find_first(self._buffer, self._wake)
        if match is None:
            if len(self._buffer) > _SCANNING_BUFFER_LIMIT:
                self._buffer = self._buffer[-_SCANNING_BUFFER_LIMIT:]
            return []
        self._buffer = self._buffer[match[1] :].strip()
        self._state = WakeState.CAPTURING
        return [Woke()]
