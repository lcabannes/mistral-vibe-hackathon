from __future__ import annotations

import getpass
import hashlib
from pathlib import Path
import platform
import re
from urllib.parse import urlparse
from uuid import uuid4

from git import InvalidGitRepositoryError, Repo

from vibe.core.team_workspace.models import TeamWorkspaceIdentity

_SCP_REMOTE = re.compile(r"^(?:[^@]+@)?(?P<host>[^:]+):(?P<path>.+)$")


def _opaque_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:32]
    return f"{prefix}_{digest}"


def normalize_project_remote(remote: str) -> str:
    value = remote.strip().rstrip("/")
    if match := _SCP_REMOTE.match(value):
        normalized = f"{match.group('host')}/{match.group('path')}"
    else:
        parsed = urlparse(value)
        host = parsed.hostname or ""
        normalized = f"{host}/{parsed.path.lstrip('/')}" if host else value
    normalized = normalized.rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.casefold()


def discover_workspace_identity(project_root: Path) -> TeamWorkspaceIdentity:
    resolved = project_root.expanduser().resolve()
    try:
        repo = Repo(resolved, search_parent_directories=True)
    except InvalidGitRepositoryError:
        return _identity_from_source(str(resolved), resolved.name or "Local project")

    working_tree = Path(repo.working_tree_dir or resolved).resolve()
    remote = _preferred_remote(repo)
    if remote:
        source = normalize_project_remote(remote)
    else:
        common_dir = Path(repo.git.rev_parse("--git-common-dir"))
        source = str(
            common_dir.resolve()
            if common_dir.is_absolute()
            else (working_tree / common_dir).resolve()
        )
    return _identity_from_source(source, working_tree.name or "Git project")


def discover_current_branch(project_root: Path) -> str | None:
    try:
        repo = Repo(project_root.expanduser().resolve(), search_parent_directories=True)
        if repo.head.is_detached:
            return None
        return repo.active_branch.name
    except (InvalidGitRepositoryError, OSError, TypeError, ValueError):
        return None


def resolve_team_repository_url(project_root: Path, configured: str) -> str | None:
    value = configured.strip()
    if value != "origin":
        return value or None
    try:
        repo = Repo(project_root.expanduser().resolve(), search_parent_directories=True)
        urls = list(repo.remote("origin").urls)
    except (InvalidGitRepositoryError, OSError, ValueError):
        return None
    return str(urls[0]) if urls else None


def _preferred_remote(repo: Repo) -> str | None:
    remotes = list(repo.remotes)
    remotes.sort(key=lambda remote: remote.name != "origin")
    for remote in remotes:
        urls = list(remote.urls)
        if urls:
            return str(urls[0])
    return None


def _identity_from_source(source: str, display_name: str) -> TeamWorkspaceIdentity:
    fingerprint = hashlib.sha256(source.encode()).hexdigest()
    return TeamWorkspaceIdentity(
        workspace_id=f"ws_{fingerprint[:32]}",
        project_fingerprint=fingerprint,
        display_name=display_name,
    )


def resolve_member_identity(
    workspace_id: str, member_name: str = "", identity_hint: str = ""
) -> tuple[str, str]:
    display_name = member_name.strip() or _git_user_name() or getpass.getuser()
    seed = identity_hint.strip() or _git_user_email()
    if not seed:
        seed = f"{display_name}@{platform.node()}"
    return _opaque_id("member", workspace_id, seed.casefold()), display_name


def new_client_id() -> str:
    return f"client_{uuid4().hex}"


def derive_run_id(workspace_id: str, member_id: str, local_run_id: str) -> str:
    return _opaque_id("run", workspace_id, member_id, local_run_id)


def derive_event_id(client_id: str, sequence: int, run_id: str) -> str:
    return _opaque_id("event", client_id, str(sequence), run_id)


def derive_entry_id(client_id: str, sequence: int, run_id: str) -> str:
    return _opaque_id("entry", client_id, str(sequence), run_id)


def _git_user_name() -> str:
    return _git_config_value("name")


def _git_user_email() -> str:
    return _git_config_value("email")


def _git_config_value(key: str) -> str:
    try:
        value = (
            Repo(Path.cwd(), search_parent_directories=True)
            .config_reader()
            .get_value("user", key, "")
        )
    except (InvalidGitRepositoryError, OSError, ValueError):
        return ""
    return str(value).strip()
