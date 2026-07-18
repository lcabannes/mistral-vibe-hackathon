from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.team_workspace import (
    ConnectionState,
    HistoryScope,
    PrivacyMode,
    TeamMemberSnapshot,
    TeamRunSnapshot,
    discover_workspace_identity,
)
from vibe.core.team_workspace.file_store import SharedTeamWorkspaceStore
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData

if TYPE_CHECKING:
    from vibe.core.config import AnyVibeConfig

MAX_TOOL_MEMBERS = 20
MAX_TOOL_RUNS = 20


class TeamActivityArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(
        default=10,
        ge=1,
        le=MAX_TOOL_RUNS,
        description="Maximum number of recent sanitized agent runs to return",
    )


class TeamActivityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project: str
    connection_state: ConnectionState
    privacy_mode: PrivacyMode
    history_scope: HistoryScope
    members: tuple[TeamMemberSnapshot, ...] = ()
    runs: tuple[TeamRunSnapshot, ...] = ()


class TeamActivityConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class TeamActivity(
    BaseTool[TeamActivityArgs, TeamActivityResult, TeamActivityConfig, BaseToolState],
    ToolUIData[TeamActivityArgs, TeamActivityResult],
):
    @classmethod
    def get_status_text(cls) -> str:
        return "Reading team activity"

    @classmethod
    def is_available(cls, config: AnyVibeConfig | None = None) -> bool:
        return bool(
            config is not None
            and config.team_workspace.enabled
            and (
                config.team_workspace.shared_root
                or config.team_workspace.team_repository_url
            )
        )

    @classmethod
    def format_call_display(cls, args: TeamActivityArgs) -> ToolCallDisplay:
        return ToolCallDisplay(summary="Reading shared team activity")

    @classmethod
    def format_result_display(cls, result: TeamActivityResult) -> ToolResultDisplay:
        return ToolResultDisplay(
            success=result.connection_state is ConnectionState.CONNECTED,
            message=f"Found {len(result.members)} coworkers and {len(result.runs)} runs",
        )

    async def run(
        self, args: TeamActivityArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[TeamActivityResult, None]:
        if ctx is None or ctx.agent_manager is None:
            raise ToolError("team_activity requires an agent manager context")
        config = ctx.agent_manager.config.team_workspace
        if not config.enabled or not (config.shared_root or config.team_repository_url):
            raise ToolError("Team workspace is not enabled")

        identity = await asyncio.to_thread(discover_workspace_identity, Path.cwd())
        shared_root = await asyncio.to_thread(
            _materialized_root,
            identity.workspace_id,
            config.shared_root,
            bool(config.team_repository_url),
        )
        if shared_root is None:
            yield TeamActivityResult(
                project=identity.display_name,
                connection_state=ConnectionState.DISCONNECTED,
                privacy_mode=PrivacyMode(config.privacy_mode),
                history_scope=HistoryScope(config.history_scope),
            )
            return
        store = SharedTeamWorkspaceStore(
            shared_root=shared_root,
            identity=identity,
            privacy_mode=PrivacyMode(config.privacy_mode),
            history_scope=HistoryScope(config.history_scope),
            history_limit=config.history_limit,
            presence_ttl_seconds=config.presence_ttl_seconds,
        )
        snapshot = await asyncio.to_thread(store.read_snapshot, _utc_now())
        yield TeamActivityResult(
            project=snapshot.identity.display_name,
            connection_state=snapshot.connection_state,
            privacy_mode=snapshot.privacy_mode,
            history_scope=snapshot.history_scope,
            members=snapshot.members[:MAX_TOOL_MEMBERS],
            runs=snapshot.runs[: args.limit],
        )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _materialized_root(
    workspace_id: str, configured_root: str, uses_git: bool
) -> Path | None:
    if not uses_git:
        return Path(configured_root) if configured_root else None
    if configured_root:
        base = Path(configured_root)
    else:
        from vibe.core.paths import VIBE_HOME

        base = VIBE_HOME.path / "team-workspaces"
    clients_root = base / workspace_id
    try:
        candidates = [
            candidate / "repo" / "state"
            for candidate in clients_root.iterdir()
            if candidate.is_dir()
            and not candidate.is_symlink()
            and (candidate / "repo" / "state").is_dir()
        ]
        return max(candidates, key=lambda path: path.stat().st_mtime)
    except (OSError, ValueError):
        return None
