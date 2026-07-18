from __future__ import annotations

from datetime import timedelta

from vibe.cli.textual_ui.workspace.coworkers import (
    CoworkerAgentViewModel,
    CoworkerConversationEntryViewModel,
    CoworkersViewModel,
    CoworkerViewModel,
)
from vibe.cli.textual_ui.workspace.models import (
    AgentActivity,
    AgentActivitySnapshot,
    AgentRunState,
)
from vibe.core.team_workspace import (
    ActivityState,
    ActivitySummary,
    ConnectionState,
    PresenceState,
    PrivacyMode,
    TeamMemberSnapshot,
    TeamRunSnapshot,
    TeamWorkspaceSnapshot,
)

_ACTIVE_STATES = frozenset({
    ActivityState.REQUESTED,
    ActivityState.RUNNING,
    ActivityState.WORKING,
    ActivityState.ATTENTION,
})
_NOW_SECONDS = 5
_SECONDS_PER_MINUTE = 60
_MINUTES_PER_HOUR = 60
_HOURS_PER_DAY = 24


def _agent_state(state: ActivityState) -> AgentRunState:
    return AgentRunState(state.value)


def _summary_label(summary: ActivitySummary | None) -> str:
    match summary:
        case ActivitySummary.STARTING:
            label = "Starting"
        case ActivitySummary.THINKING:
            label = "Thinking"
        case ActivitySummary.USING_TOOL:
            label = "Using tool"
        case ActivitySummary.WAITING_FOR_APPROVAL:
            label = "Waiting for approval"
        case ActivitySummary.WAITING_FOR_INPUT:
            label = "Waiting for input"
        case ActivitySummary.FINISHED:
            label = "Finished"
        case ActivitySummary.FAILED:
            label = "Failed"
        case ActivitySummary.CANCELLED:
            label = "Cancelled"
        case None:
            label = ""
    return label


def _age_label(age: timedelta) -> str:
    seconds = max(0, int(age.total_seconds()))
    if seconds < _NOW_SECONDS:
        return "now"
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds}s ago"
    minutes = seconds // _SECONDS_PER_MINUTE
    if minutes < _MINUTES_PER_HOUR:
        return f"{minutes}m ago"
    hours = minutes // _MINUTES_PER_HOUR
    if hours < _HOURS_PER_DAY:
        return f"{hours}h ago"
    return f"{hours // _HOURS_PER_DAY}d ago"


def _member_presence(member: TeamMemberSnapshot, connection: ConnectionState) -> str:
    if member.presence is PresenceState.OFFLINE:
        return "offline"
    if connection is ConnectionState.DEGRADED:
        return "stale"
    return "online"


def _run_sort_key(run: TeamRunSnapshot) -> tuple[int, float, str]:
    if run.state is ActivityState.ATTENTION:
        priority = 0
    elif run.state in _ACTIVE_STATES:
        priority = 1
    elif run.state is ActivityState.FAILED:
        priority = 2
    else:
        priority = 3
    return priority, -run.updated_at.timestamp(), run.run_id


def _member_sort_key(
    member: TeamMemberSnapshot, runs: tuple[TeamRunSnapshot, ...]
) -> tuple[int, str, str]:
    member_runs = tuple(run for run in runs if run.member_id == member.member_id)
    if any(run.state is ActivityState.ATTENTION for run in member_runs):
        priority = 0
    elif member.active_run_count:
        priority = 1
    elif member.presence is PresenceState.ONLINE:
        priority = 2
    else:
        priority = 3
    return priority, member.display_name.casefold(), member.member_id


def coworkers_view(snapshot: TeamWorkspaceSnapshot) -> CoworkersViewModel:
    runs_by_member: dict[str, list[TeamRunSnapshot]] = {}
    for run in snapshot.runs:
        runs_by_member.setdefault(run.member_id, []).append(run)

    members: list[CoworkerViewModel] = []
    for member in sorted(
        snapshot.members, key=lambda item: _member_sort_key(item, snapshot.runs)
    ):
        member_runs = tuple(
            sorted(runs_by_member.get(member.member_id, ()), key=_run_sort_key)
        )
        agents = tuple(
            CoworkerAgentViewModel(
                run_id=run.run_id,
                display_name=run.agent_display_name,
                state=_agent_state(run.state),
                summary=_summary_label(run.summary),
                updated_label=_age_label(snapshot.generated_at - run.updated_at),
                history=tuple(
                    CoworkerConversationEntryViewModel(
                        entry_id=entry.entry_id,
                        role=entry.role.value,
                        text=entry.text,
                        updated_label=_age_label(
                            snapshot.generated_at - entry.occurred_at
                        ),
                    )
                    for entry in run.history
                ),
            )
            for run in member_runs
        )
        recent_summary = next((agent.summary for agent in agents if agent.summary), "")
        members.append(
            CoworkerViewModel(
                member_id=member.member_id,
                display_name=member.display_name,
                presence=_member_presence(member, snapshot.connection_state),
                branch=member.branch,
                summary=recent_summary,
                updated_label=_age_label(snapshot.generated_at - member.last_seen_at),
                active_run_count=member.active_run_count,
                agents=agents,
            )
        )

    return CoworkersViewModel(
        workspace_name=snapshot.identity.display_name,
        connection_state=snapshot.connection_state.value,
        privacy_label=(
            "summaries shared"
            if snapshot.privacy_mode is PrivacyMode.SUMMARIES
            else "status only"
        ),
        members=tuple(members),
        error=snapshot.error.value.replace("_", " ") if snapshot.error else None,
        join_hint=(
            "vibe team join <team-repo-url>"
            if snapshot.connection_state is ConnectionState.DISABLED
            else None
        ),
    )


def team_activity_snapshot(snapshot: TeamWorkspaceSnapshot) -> AgentActivitySnapshot:
    branches = {member.member_id: member.branch for member in snapshot.members}
    activities = tuple(
        AgentActivity(
            tool_call_id=run.run_id,
            parent_session_id=snapshot.identity.workspace_id,
            agent_name=run.agent_name,
            agent_display_name=run.agent_display_name,
            task="Shared agent run",
            state=_agent_state(run.state),
            started_at=run.started_at.timestamp(),
            updated_at=run.updated_at.timestamp(),
            current_activity=_summary_label(run.summary) or None,
            owner_display_name=run.member_display_name,
            branch=branches.get(run.member_id),
        )
        for run in snapshot.runs
    )
    return AgentActivitySnapshot(
        session_id=snapshot.identity.workspace_id, activities=activities
    )


def team_sync_summary(snapshot: TeamWorkspaceSnapshot) -> str:
    match snapshot.connection_state:
        case ConnectionState.CONNECTED:
            return "✓ Live team workspace"
        case ConnectionState.DEGRADED:
            suffix = f" · {snapshot.error.value}" if snapshot.error else ""
            return f"! Stale team workspace{suffix}"
        case ConnectionState.DISCONNECTED:
            return "○ Team workspace offline"
        case ConnectionState.DISABLED:
            return "○ Team workspace disabled"


__all__ = ["coworkers_view", "team_activity_snapshot", "team_sync_summary"]
