from __future__ import annotations

from datetime import UTC, datetime

from pydantic import ValidationError
import pytest

from vibe.core.team_workspace import (
    ActivityState,
    ActivitySummary,
    ConversationRole,
    HistoryScope,
    PrivacyMode,
    TeamActivityEvent,
    TeamConversationEntry,
)

NOW = datetime(2026, 7, 18, 10, 0, tzinfo=UTC)


def _event(**overrides: object) -> TeamActivityEvent:
    values: dict[str, object] = {
        "workspace_id": "ws_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "event_id": "event_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "member_id": "member_cccccccccccccccccccccccccccccccc",
        "member_display_name": "Ada Lovelace",
        "client_id": "client_dddddddddddddddddddddddddddddddd",
        "sequence": 1,
        "run_id": "run_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        "agent_name": "default",
        "agent_display_name": "Default",
        "state": ActivityState.WORKING,
        "privacy_mode": PrivacyMode.STATUS,
        "occurred_at": NOW,
    }
    values.update(overrides)
    return TeamActivityEvent.model_validate(values)


def test_activity_event_surface_cannot_carry_sensitive_content() -> None:
    forbidden = {
        "prompt",
        "chat",
        "reasoning",
        "args",
        "result",
        "command",
        "output",
        "path",
        "config",
        "environment",
        "approval",
        "response",
        "task",
        "current_activity",
    }

    assert forbidden.isdisjoint(TeamActivityEvent.model_fields)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _event(prompt="private prompt")


def test_status_privacy_rejects_even_enum_summary() -> None:
    with pytest.raises(ValidationError, match="cannot include activity summaries"):
        _event(summary=ActivitySummary.USING_TOOL)


def test_summaries_privacy_accepts_only_fixed_summary_enum() -> None:
    event = _event(
        privacy_mode=PrivacyMode.SUMMARIES, summary=ActivitySummary.USING_TOOL
    )
    assert event.summary is ActivitySummary.USING_TOOL

    with pytest.raises(ValidationError):
        _event(privacy_mode=PrivacyMode.SUMMARIES, summary="cat ~/.ssh/id_rsa")


@pytest.mark.parametrize("field", ["member_display_name", "agent_display_name"])
def test_shared_labels_reject_paths(field: str) -> None:
    with pytest.raises(ValidationError, match="path separators"):
        _event(**{field: "/Users/alice/private"})


def test_shared_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError, match="UTC offset"):
        _event(occurred_at=datetime(2026, 7, 18, 10, 0))


def test_activity_terminal_states_are_explicit() -> None:
    assert ActivityState.COMPLETED.is_terminal
    assert ActivityState.FAILED.is_terminal
    assert ActivityState.CANCELLED.is_terminal
    assert not ActivityState.WORKING.is_terminal


def test_message_history_redacts_secrets_paths_commands_and_code() -> None:
    entry = TeamConversationEntry(
        workspace_id="ws_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        entry_id="entry_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        member_id="member_cccccccccccccccccccccccccccccccc",
        client_id="client_dddddddddddddddddddddddddddddddd",
        sequence=1,
        run_id="run_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        role=ConversationRole.USER,
        history_scope=HistoryScope.MESSAGES,
        text=(
            "Inspect /Users/alice/private/token.txt\n"
            "API_KEY=super-secret\n"
            "$ cat ~/.ssh/id_rsa\n"
            "```sh\nrm -rf /\n```"
        ),
        occurred_at=NOW,
    )

    assert entry.text is not None
    assert "alice" not in entry.text
    assert "super-secret" not in entry.text
    assert "cat ~/.ssh" not in entry.text
    assert "rm -rf" not in entry.text
    assert "[Filtered]" in entry.text
    assert "[Command omitted]" in entry.text
    assert "[Code omitted]" in entry.text


def test_conversation_history_cannot_represent_system_or_tool_roles() -> None:
    values = {
        "workspace_id": "ws_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "entry_id": "entry_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "member_id": "member_cccccccccccccccccccccccccccccccc",
        "client_id": "client_dddddddddddddddddddddddddddddddd",
        "sequence": 1,
        "run_id": "run_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        "role": "tool",
        "history_scope": HistoryScope.MESSAGES,
        "text": "tool output",
        "occurred_at": NOW,
    }

    with pytest.raises(ValidationError):
        TeamConversationEntry.model_validate(values)


def test_status_and_marker_history_cannot_publish_text() -> None:
    common = {
        "workspace_id": "ws_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "entry_id": "entry_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "member_id": "member_cccccccccccccccccccccccccccccccc",
        "client_id": "client_dddddddddddddddddddddddddddddddd",
        "sequence": 1,
        "run_id": "run_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        "role": ConversationRole.ASSISTANT,
        "occurred_at": NOW,
    }
    with pytest.raises(ValidationError, match="status history"):
        TeamConversationEntry(**common, history_scope=HistoryScope.STATUS)
    with pytest.raises(ValidationError, match="cannot include text"):
        TeamConversationEntry(
            **common, history_scope=HistoryScope.MARKERS, text="private"
        )
