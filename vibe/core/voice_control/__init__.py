from __future__ import annotations

from vibe.core.voice_control.ambient_listener import AmbientCallbacks, AmbientListener
from vibe.core.voice_control.answers import answer_from_speech, spoken_question_prompt
from vibe.core.voice_control.runner import VoiceControlRunner
from vibe.core.voice_control.service import (
    VoiceControlConfig,
    VoiceControlService,
    VoicePhase,
)
from vibe.core.voice_control.speaker import Speaker, SpeakerPort
from vibe.core.voice_control.wake_word import (
    DEFAULT_CANCEL_PHRASES,
    DEFAULT_WAKE_PHRASES,
    Utterance,
    WakeEvent,
    WakeState,
    WakeWordMachine,
    Woke,
    is_cancel_phrase,
    normalize,
)

__all__ = [
    "DEFAULT_CANCEL_PHRASES",
    "DEFAULT_WAKE_PHRASES",
    "AmbientCallbacks",
    "AmbientListener",
    "Speaker",
    "SpeakerPort",
    "Utterance",
    "VoiceControlConfig",
    "VoiceControlRunner",
    "VoiceControlService",
    "VoicePhase",
    "WakeEvent",
    "WakeState",
    "WakeWordMachine",
    "Woke",
    "answer_from_speech",
    "is_cancel_phrase",
    "normalize",
    "spoken_question_prompt",
]
