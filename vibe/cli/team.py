from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rich.console import Console

from vibe.core.config.team_metadata import (
    TeamProjectMetadata,
    resolve_project_root,
    write_team_project_metadata,
)
from vibe.core.paths import VIBE_HOME
from vibe.core.team_workspace import (
    ConnectionState,
    HistoryScope,
    PrivacyMode,
    SyncError,
    build_team_workspace_service,
    discover_workspace_identity,
)


async def _join(args: argparse.Namespace, cwd: Path) -> tuple[bool, str]:
    project_root = resolve_project_root(cwd)
    identity = discover_workspace_identity(project_root)
    metadata = TeamProjectMetadata(
        team_repo_url=args.team_repo_url,
        branch=args.branch,
        history_scope=args.history_scope,
        history_limit=args.history_limit,
        workspace_id=identity.workspace_id,
    )
    service = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project_root,
        team_repository_url=metadata.team_repo_url,
        team_branch=metadata.branch,
        cache_root=VIBE_HOME.path / "team-workspaces",
        privacy_mode=PrivacyMode.STATUS,
        history_scope=HistoryScope(metadata.history_scope),
        history_limit=metadata.history_limit,
    )
    try:
        snapshot = await service.start()
    finally:
        await service.stop()

    connected = snapshot.connection_state in {
        ConnectionState.CONNECTED,
        ConnectionState.DEGRADED,
    }
    if not connected or snapshot.error is SyncError.TRANSPORT_FAILED:
        reason = snapshot.error.value if snapshot.error else "connection failed"
        return False, reason

    changed = write_team_project_metadata(metadata, project_root)
    action = "Joined" if changed else "Already joined"
    return True, f"{action} {identity.display_name} ({identity.workspace_id})"


def run_team_command(args: argparse.Namespace, cwd: Path) -> int:
    console = Console(stderr=True)
    if args.team_action != "join":
        console.print(f"[red]Unknown team command: {args.team_action}[/]")
        return 2
    try:
        success, message = asyncio.run(_join(args, cwd))
    except (OSError, ValueError) as error:
        console.print(f"[red]Could not join team workspace: {error}[/]")
        return 1
    if not success:
        console.print(f"[red]Could not join team workspace: {message}[/]")
        return 1
    console.print(f"[green]{message}[/]")
    console.print("[dim]Committed settings: .vibe/team.toml[/]")
    return 0
