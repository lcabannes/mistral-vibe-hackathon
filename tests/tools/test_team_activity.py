from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.agents.manager import AgentManager
from vibe.core.config import TeamWorkspaceConfig
from vibe.core.config.orchestrator_legacy import LegacyConfigOrchestrator
from vibe.core.team_workspace import (
    ActivityState,
    ActivitySummary,
    ConnectionState,
    PrivacyMode,
    TeamActivityEvent,
    TeamMemberPresence,
    discover_workspace_identity,
)
from vibe.core.team_workspace.file_store import SharedTeamWorkspaceStore
from vibe.core.tools.base import InvokeContext
from vibe.core.tools.builtins.team_activity import (
    MAX_TOOL_MEMBERS,
    TeamActivity,
    TeamActivityArgs,
    TeamActivityConfig,
    TeamActivityResult,
)

NOW = datetime(2026, 7, 18, 10, 0, tzinfo=UTC)
MEMBER = "member_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
CLIENT = "client_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def _manager(
    shared_root: Path, privacy: Literal["status", "summaries"] = "summaries"
) -> AgentManager:
    config = build_test_vibe_config(
        team_workspace=TeamWorkspaceConfig(
            enabled=True, shared_root=str(shared_root), privacy_mode=privacy
        )
    )
    return AgentManager(LegacyConfigOrchestrator(config))


def _populate(root: Path, project_root: Path, count: int = 1) -> None:
    identity = discover_workspace_identity(project_root)
    store = SharedTeamWorkspaceStore(
        shared_root=root,
        identity=identity,
        privacy_mode=PrivacyMode.SUMMARIES,
        member_id=MEMBER,
        client_id=CLIENT,
    )
    store.initialize(NOW)
    store.write_presence(
        TeamMemberPresence(
            workspace_id=identity.workspace_id,
            member_id=MEMBER,
            member_display_name="Ada",
            client_id=CLIENT,
            branch="feature/team",
            revision=1,
            last_seen_at=NOW,
        )
    )
    for sequence in range(1, count + 1):
        store.write_event(
            TeamActivityEvent(
                workspace_id=identity.workspace_id,
                event_id=f"event_{sequence:032x}",
                member_id=MEMBER,
                member_display_name="Ada",
                client_id=CLIENT,
                sequence=sequence,
                run_id=f"run_{sequence:032x}",
                agent_name="default",
                agent_display_name="Default",
                state=ActivityState.WORKING,
                privacy_mode=PrivacyMode.SUMMARIES,
                summary=ActivitySummary.USING_TOOL,
                occurred_at=NOW + timedelta(seconds=sequence),
            )
        )


@pytest.mark.asyncio
async def test_team_activity_reads_bounded_materialized_snapshot_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    shared = tmp_path / "shared"
    monkeypatch.chdir(project)
    _populate(shared, project, count=15)
    before = {
        path: path.stat().st_mtime_ns for path in shared.rglob("*") if path.is_file()
    }
    manager = _manager(shared)
    tool = cast(TeamActivity, TeamActivity.from_config(TeamActivityConfig))

    results = [
        item
        async for item in tool.run(
            TeamActivityArgs(limit=4),
            InvokeContext(tool_call_id="call", agent_manager=manager),
        )
    ]

    assert len(results) == 1
    result = cast(TeamActivityResult, results[0])
    assert result.connection_state is ConnectionState.CONNECTED
    assert result.project == "project"
    assert len(result.members) <= MAX_TOOL_MEMBERS
    assert len(result.runs) == 4
    assert all(run.summary is ActivitySummary.USING_TOOL for run in result.runs)
    after = {
        path: path.stat().st_mtime_ns for path in shared.rglob("*") if path.is_file()
    }
    assert after == before


@pytest.mark.asyncio
async def test_team_activity_returns_degraded_state_for_missing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.chdir(project)
    manager = _manager(shared)
    tool = cast(TeamActivity, TeamActivity.from_config(TeamActivityConfig))

    results = [
        item
        async for item in tool.run(
            TeamActivityArgs(),
            InvokeContext(tool_call_id="call", agent_manager=manager),
        )
    ]
    result = cast(TeamActivityResult, results[0])

    assert result.connection_state is ConnectionState.DEGRADED
    assert not result.members
    assert not result.runs


def test_team_activity_is_available_only_for_configured_shared_workspace(
    tmp_path: Path,
) -> None:
    disabled = build_test_vibe_config()
    no_root = build_test_vibe_config(team_workspace=TeamWorkspaceConfig(enabled=True))
    enabled = build_test_vibe_config(
        team_workspace=TeamWorkspaceConfig(enabled=True, shared_root=str(tmp_path))
    )

    assert not TeamActivity.is_available()
    assert not TeamActivity.is_available(disabled)
    assert not TeamActivity.is_available(no_root)
    assert TeamActivity.is_available(enabled)


def test_team_activity_result_has_no_remote_control_or_sensitive_fields() -> None:
    parameters = TeamActivity.get_parameters()
    description = TeamActivity.get_full_description()

    assert set(parameters["properties"]) == {"limit"}
    assert "cannot contact, control, message, approve, or cancel" in description
    for sensitive in ("prompt", "reasoning", "command", "output", "approval"):
        assert sensitive not in parameters["properties"]
