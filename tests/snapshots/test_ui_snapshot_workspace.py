from __future__ import annotations

from tests.snapshots.base_snapshot_test_app import BaseSnapshotTestApp, default_config
from tests.snapshots.snap_compare import SnapCompare
from vibe.cli.textual_ui.app import StartupOptions
from vibe.cli.textual_ui.workspace.models import (
    AgentActivity,
    AgentActivitySnapshot,
    AgentRunState,
    WorkspaceView,
)
from vibe.cli.textual_ui.workspace.pages import (
    HomePage,
    OfficeViewModel,
    UsagePage,
    UsageViewModel,
)
from vibe.core.agent_room.models import AgentRoomConversationMessage


def _activity(index: int, state: AgentRunState) -> AgentActivity:
    return AgentActivity(
        tool_call_id=f"snapshot-{index}",
        parent_session_id="parent",
        agent_name="explore",
        agent_display_name=f"Explore {index}",
        task=f"Review a long responsive task description {index}",
        state=state,
        started_at=float(index),
        updated_at=float(index + 1),
        current_activity=f"Waiting for approval on responsive item {index}",
        managed_agent_id=f"snapshot-agent-{index}",
        conversation=(
            AgentRoomConversationMessage(
                id=f"snapshot-user-{index}",
                role="user",
                content=f"Please investigate task {index}.",
            ),
            AgentRoomConversationMessage(
                id=f"snapshot-assistant-{index}",
                role="assistant",
                content=f"I found the relevant implementation for task {index}.",
            ),
        ),
    )


def _populated_home_view() -> OfficeViewModel:
    states = (
        AgentRunState.FAILED,
        AgentRunState.COMPLETED,
        AgentRunState.IDLE,
        AgentRunState.STOPPED,
        AgentRunState.CANCELLED,
        AgentRunState.FAILED,
        AgentRunState.COMPLETED,
        AgentRunState.IDLE,
        AgentRunState.STOPPED,
        AgentRunState.CANCELLED,
    )
    snapshot = AgentActivitySnapshot(
        session_id="parent",
        activities=tuple(_activity(index, state) for index, state in enumerate(states)),
    )
    return OfficeViewModel(snapshot, "Agent Room · main")


def _populated_usage_view() -> UsageViewModel:
    return UsageViewModel(
        steps=12,
        prompt_tokens=12_345,
        completion_tokens=6_789,
        context_tokens=98_765,
        tool_calls_succeeded=42,
        tool_calls_failed=2,
        tool_calls_rejected=1,
        session_cost=1.2345,
        last_turn_duration=12.3,
        tokens_per_second=456.7,
    )


class WorkspaceHomeSnapshotApp(BaseSnapshotTestApp):
    snapshot_theme = "textual-dark"

    def __init__(self) -> None:
        config = default_config()
        config.theme = self.snapshot_theme
        super().__init__(config=config, startup=StartupOptions())

    async def on_ready(self) -> None:
        await super().on_ready()
        self.query_one(HomePage).update_view(_populated_home_view())
        self._focus_workspace_view()


class WorkspaceHomeLightSnapshotApp(WorkspaceHomeSnapshotApp):
    snapshot_theme = "textual-light"


class WorkspaceUsageLightSnapshotApp(WorkspaceHomeLightSnapshotApp):
    async def on_ready(self) -> None:
        await super().on_ready()
        self.action_show_workspace(WorkspaceView.USAGE.value)
        self.query_one(UsagePage).update_view(_populated_usage_view())


def test_snapshot_workspace_home_wide(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_workspace.py:WorkspaceHomeSnapshotApp",
        terminal_size=(140, 40),
    )


def test_snapshot_workspace_home_medium_light(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_workspace.py:WorkspaceHomeLightSnapshotApp",
        terminal_size=(100, 32),
    )


def test_snapshot_workspace_home_narrow(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_workspace.py:WorkspaceHomeSnapshotApp", terminal_size=(70, 24)
    )


def test_snapshot_workspace_usage_narrow_light(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_workspace.py:WorkspaceUsageLightSnapshotApp",
        terminal_size=(70, 24),
    )
