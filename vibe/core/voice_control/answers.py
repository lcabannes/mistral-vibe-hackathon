from __future__ import annotations

import difflib
from typing import Any

from vibe.core.voice_control.wake_word import normalize

_MATCH_THRESHOLD = 0.6


def spoken_question_prompt(question: dict[str, Any]) -> str:
    """Render a question (from AskUserQuestionArgs) as a short spoken prompt."""
    text = str(question.get("question") or "").strip()
    labels = _labels(question)
    if not labels or question.get("hide_other"):
        return text
    return f"{text} Options are: {', '.join(labels)}."


def answer_from_speech(
    question: dict[str, Any], spoken: str, *, threshold: float = _MATCH_THRESHOLD
) -> dict[str, Any]:
    """Turn a spoken reply into an `Answer` dict for a single question.

    Matches the transcript against the option labels; falls back to a free-text
    ("Other") answer when nothing matches confidently.
    """
    q_text = str(question.get("question") or "")
    text = spoken.strip()
    labels = _labels(question)
    best_label, best_score = _best_label(text, labels)

    if best_label is not None and best_score >= threshold:
        return {"question": q_text, "answer": best_label, "is_other": False}
    if question.get("hide_other") and best_label is not None:
        return {"question": q_text, "answer": best_label, "is_other": False}
    return {"question": q_text, "answer": text, "is_other": True}


def _labels(question: dict[str, Any]) -> list[str]:
    options = question.get("options") or []
    return [
        str(option.get("label") or "")
        for option in options
        if isinstance(option, dict) and option.get("label")
    ]


def _best_label(spoken: str, labels: list[str]) -> tuple[str | None, float]:
    norm_spoken = normalize(spoken)
    best_label: str | None = None
    best_score = 0.0
    for label in labels:
        norm_label = normalize(label)
        if not norm_label:
            continue
        if norm_label in norm_spoken or norm_spoken in norm_label:
            score = 1.0
        else:
            score = difflib.SequenceMatcher(None, norm_label, norm_spoken).ratio()
        if score > best_score:
            best_score = score
            best_label = label
    return best_label, best_score
