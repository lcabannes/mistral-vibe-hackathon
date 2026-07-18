from __future__ import annotations

from vibe.core.voice_control.answers import answer_from_speech, spoken_question_prompt

_QUESTION = {
    "question": "Which category should this agent be?",
    "options": [
        {"label": "Build", "description": "Implements code"},
        {"label": "Research", "description": "Investigates"},
        {"label": "Review", "description": "Reviews changes"},
    ],
}


def test_prompt_includes_options() -> None:
    prompt = spoken_question_prompt(_QUESTION)
    assert "Which category" in prompt
    assert "Build, Research, Review" in prompt


def test_prompt_omits_options_when_hidden() -> None:
    question = {"question": "What is the task?", "options": [], "hide_other": True}
    assert spoken_question_prompt(question) == "What is the task?"


def test_exact_label_match() -> None:
    answer = answer_from_speech(_QUESTION, "research")
    assert answer["answer"] == "Research"
    assert answer["is_other"] is False


def test_label_match_within_a_sentence() -> None:
    answer = answer_from_speech(_QUESTION, "let's make it a build agent")
    assert answer["answer"] == "Build"
    assert answer["is_other"] is False


def test_fuzzy_label_match() -> None:
    answer = answer_from_speech(_QUESTION, "reviu")
    assert answer["answer"] == "Review"
    assert answer["is_other"] is False


def test_no_match_falls_back_to_other() -> None:
    answer = answer_from_speech(_QUESTION, "something completely unrelated")
    assert answer["is_other"] is True
    assert answer["answer"] == "something completely unrelated"


def test_question_text_is_preserved() -> None:
    answer = answer_from_speech(_QUESTION, "build")
    assert answer["question"] == _QUESTION["question"]
