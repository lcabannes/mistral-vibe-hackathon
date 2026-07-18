from __future__ import annotations

from datetime import UTC, datetime, timedelta

from vibe.cli.textual_ui.workspace.models import AgentRunState
from vibe.cli.textual_ui.workspace.team_presenter import (
    coworkers_view,
    team_activity_snapshot,
    team_sync_summary,
)
from vibe.core.team_workspace import (
    ActivityState,
    ActivitySummary,
    ConnectionState,
    ConversationRole,
    HistoryScope,
    PresenceState,
    PrivacyMode,
    TeamConversationEntry,
    TeamMemberSnapshot,
    TeamRunSnapshot,
    TeamWorkspaceIdentity,
    TeamWorkspaceSnapshot,
)

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def _snapshot() -> TeamWorkspaceSnapshot:
    history = TeamConversationEntry(
        workspace_id="ws_12345678",
        entry_id="entry_12345678",
        member_id="member_12345678",
        client_id="client_12345678",
        sequence=2,
        run_id="run_12345678",
        role=ConversationRole.ASSISTANT,
        history_scope=HistoryScope.MESSAGES,
        text="Shared status update",
        occurred_at=NOW - timedelta(seconds=40),
    )
    run = TeamRunSnapshot(
        run_id="run_12345678",
        member_id="member_12345678",
        member_display_name="Alice",
        client_id="client_12345678",
        agent_name="explore",
        agent_display_name="Explore",
        state=ActivityState.ATTENTION,
        summary=ActivitySummary.WAITING_FOR_INPUT,
        started_at=NOW - timedelta(minutes=5),
        updated_at=NOW - timedelta(seconds=40),
        sequence=2,
        history=(history,),
    )
    return TeamWorkspaceSnapshot(
        identity=TeamWorkspaceIdentity(
            workspace_id="ws_12345678",
            project_fingerprint="a" * 64,
            display_name="mistral-vibe",
        ),
        privacy_mode=PrivacyMode.SUMMARIES,
        connection_state=ConnectionState.CONNECTED,
        generated_at=NOW,
        members=(
            TeamMemberSnapshot(
                member_id="member_12345678",
                display_name="Alice",
                presence=PresenceState.ONLINE,
                branch="codex/team-workspace",
                last_seen_at=NOW - timedelta(seconds=5),
                client_count=1,
                active_run_count=1,
            ),
        ),
        runs=(run,),
    )


def test_team_snapshot_maps_owner_branch_history_and_deterministic_ages() -> None:
    snapshot = _snapshot()

    coworkers = coworkers_view(snapshot)
    activity = team_activity_snapshot(snapshot).activities[0]

    assert coworkers.workspace_name == "mistral-vibe"
    assert coworkers.members[0].updated_label == "5s ago"
    assert coworkers.members[0].agents[0].state is AgentRunState.ATTENTION
    assert coworkers.members[0].agents[0].updated_label == "40s ago"
    assert coworkers.members[0].agents[0].history[0].text == "Shared status update"
    assert activity.owner_display_name == "Alice"
    assert activity.branch == "codex/team-workspace"
    assert activity.state is AgentRunState.ATTENTION
    assert team_sync_summary(snapshot) == "✓ Live team workspace"


def test_disabled_snapshot_exposes_only_single_join_hint() -> None:
    snapshot = _snapshot().model_copy(
        update={
            "connection_state": ConnectionState.DISABLED,
            "members": (),
            "runs": (),
        }
    )

    view = coworkers_view(snapshot)

    assert view.join_hint == "vibe team join <team-repo-url>"
    assert view.members == ()

