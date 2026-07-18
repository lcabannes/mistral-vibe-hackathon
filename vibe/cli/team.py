from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console

from vibe.core.config.team_metadata import (
    TeamProjectMetadata,
    clear_team_workspace_leave,
    leave_team_workspace,
    load_team_project_metadata,
    resolve_project_root,
    write_team_project_metadata,
)
from vibe.core.paths import VIBE_HOME
from vibe.core.team_workspace import (
    ConnectionState,
    HistoryScope,
    PrivacyMode,
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
        respect_local_leave=False,
    )
    try:
        snapshot = await service.start()
    finally:
        await service.stop()

    if snapshot.connection_state is not ConnectionState.CONNECTED or snapshot.error:
        reason = snapshot.error.value if snapshot.error else "connection failed"
        return False, reason

    changed = write_team_project_metadata(metadata, project_root)
    clear_team_workspace_leave(project_root)
    action = "Joined" if changed else "Already joined"
    settings = "Wrote .vibe/team.toml" if changed else ".vibe/team.toml already matches"
    return True, (
        f"{action} {identity.display_name} ({identity.workspace_id}). {settings}; "
        "commit and push it so trusted clones autojoin"
    )


def _leave(cwd: Path) -> tuple[bool, str]:
    project_root = resolve_project_root(cwd)
    metadata = load_team_project_metadata(project_root)
    if metadata is None:
        return False, "this project has no committed .vibe/team.toml"
    identity = discover_workspace_identity(project_root)
    changed = leave_team_workspace(project_root)
    action = "Left" if changed else "Already left"
    return True, (
        f"{action} {identity.display_name} ({identity.workspace_id}) locally. "
        "Committed team metadata was not changed; run `vibe team join "
        f"{metadata.team_repo_url}` to rejoin"
    )


def _validation_error_message(error: ValidationError) -> str:
    messages = [
        str(item["msg"]).removeprefix("Value error, ") for item in error.errors()
    ]
    return "; ".join(messages) or "invalid team workspace settings"


def _run_leave_command(console: Console, cwd: Path) -> int:
    try:
        success, message = _leave(cwd)
    except (OSError, ValueError) as error:
        console.print(f"[red]Could not leave team workspace: {error}[/]")
        return 1
    if not success:
        console.print(f"[red]Could not leave team workspace: {message}[/]")
        return 1
    console.print(f"[green]{message}[/]")
    return 0


def run_team_command(args: argparse.Namespace, cwd: Path) -> int:
    console = Console(stderr=True)
    if args.team_action == "leave":
        return _run_leave_command(console, cwd)
    if args.team_action != "join":
        console.print(f"[red]Unknown team command: {args.team_action}[/]")
        return 2
    try:
        success, message = asyncio.run(_join(args, cwd))
    except ValidationError as error:
        console.print(
            f"[red]Could not join team workspace: {_validation_error_message(error)}[/]"
        )
        return 1
    except (OSError, ValueError) as error:
        console.print(f"[red]Could not join team workspace: {error}[/]")
        return 1
    if not success:
        console.print(f"[red]Could not join team workspace: {message}[/]")
        return 1
    console.print(f"[green]{message}[/]")
    return 0
