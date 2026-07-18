from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import re
import tempfile
import tomllib
from typing import BinaryIO, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator
import tomli_w

from vibe.core.config.harness_files import get_harness_files_manager
from vibe.core.paths import VIBE_HOME
from vibe.core.utils.io import read_safe

TEAM_METADATA_PATH = Path(".vibe") / "team.toml"
TEAM_LEAVE_MARKERS_DIR = "team-workspace-leaves"
TEAM_WORKSPACE_LOCKS_DIR = "team-workspace-locks"


class TeamMetadataError(ValueError):
    pass


class TeamProjectMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    team_repo_url: str = Field(min_length=1)
    branch: str = Field(default="vibe-team-demo", min_length=1)
    history_limit: int = Field(default=50, ge=1, le=200)
    history_scope: Literal["status", "markers", "messages"] = "markers"
    workspace_id: str | None = Field(default=None, pattern=r"^ws_[a-f0-9]{32}$")

    @field_validator("team_repo_url", "branch")
    @classmethod
    def trim_nonblank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @field_validator("team_repo_url")
    @classmethod
    def reject_credentials(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.password is not None or (
            parsed.scheme in {"http", "https"} and parsed.username is not None
        ):
            raise ValueError("team repository URL must not contain credentials")
        return value

    @field_validator("branch")
    @classmethod
    def validate_branch(cls, value: str) -> str:
        invalid_shape = (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value)
            or value.startswith(("/", ".", "-"))
            or value.endswith(("/", "."))
        )
        invalid_fragment = any(item in value for item in ("..", "//", "@{"))
        invalid_segment = any(
            part.startswith(".") or part.endswith(".lock") for part in value.split("/")
        )
        if invalid_shape or invalid_fragment or invalid_segment:
            raise ValueError("invalid team branch")
        return value


def resolve_project_root(path: Path | None = None) -> Path:
    from git import InvalidGitRepositoryError, Repo

    candidate = (path or Path.cwd()).expanduser().resolve()
    try:
        repo = Repo(candidate, search_parent_directories=True)
    except InvalidGitRepositoryError:
        return candidate
    return Path(repo.working_tree_dir or candidate).resolve()


def load_team_project_metadata(
    project_root: Path | None = None,
) -> TeamProjectMetadata | None:
    root = project_root or _trusted_project_root()
    if root is None:
        return None
    path = _find_team_metadata_path(root)
    if path is None:
        return None
    repository_root = resolve_project_root(root)
    if path != repository_root / TEAM_METADATA_PATH:
        return None
    try:
        metadata = TeamProjectMetadata.model_validate(
            tomllib.loads(read_safe(path).text)
        )
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        raise TeamMetadataError(f"Invalid team metadata at {path}: {exc}") from exc

    from vibe.core.team_workspace.identity import discover_workspace_identity

    identity = discover_workspace_identity(repository_root)
    if metadata.workspace_id and metadata.workspace_id != identity.workspace_id:
        raise TeamMetadataError(
            "Team metadata workspace_id does not match this repository remote"
        )
    return metadata


def write_team_project_metadata(
    metadata: TeamProjectMetadata, project_root: Path | None = None
) -> bool:
    from vibe.core.team_workspace.identity import discover_workspace_identity

    root = resolve_project_root(project_root)
    identity = discover_workspace_identity(root)
    if metadata.workspace_id and metadata.workspace_id != identity.workspace_id:
        raise TeamMetadataError(
            "Team metadata workspace_id does not match this repository remote"
        )

    path = root / TEAM_METADATA_PATH
    if path.is_file():
        try:
            existing = TeamProjectMetadata.model_validate(
                tomllib.loads(read_safe(path).text)
            )
        except (OSError, tomllib.TOMLDecodeError, ValueError):
            existing = None
        if existing == metadata:
            return False

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".team-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            tomli_w.dump(metadata.model_dump(exclude_none=True), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def team_leave_marker_path(project_root: Path) -> Path:
    from vibe.core.team_workspace.identity import discover_workspace_identity

    identity = discover_workspace_identity(resolve_project_root(project_root))
    return team_leave_marker_path_for_id(identity.workspace_id)


def team_leave_marker_path_for_id(workspace_id: str) -> Path:
    return VIBE_HOME.path / TEAM_LEAVE_MARKERS_DIR / workspace_id


def team_workspace_lock_path_for_id(workspace_id: str) -> Path:
    return VIBE_HOME.path / TEAM_WORKSPACE_LOCKS_DIR / f"{workspace_id}.lock"


@contextmanager
def team_workspace_lock_for_id(workspace_id: str) -> Iterator[None]:
    path = team_workspace_lock_path_for_id(workspace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        _acquire_file_lock(lock_file)
        try:
            yield
        finally:
            _release_file_lock(lock_file)


def is_team_workspace_left_id(workspace_id: str) -> bool:
    return team_leave_marker_path_for_id(workspace_id).is_file()


def is_team_workspace_left(project_root: Path) -> bool:
    return team_leave_marker_path(project_root).is_file()


def leave_team_workspace(project_root: Path) -> bool:
    marker = team_leave_marker_path(project_root)
    workspace_id = marker.name
    with team_workspace_lock_for_id(workspace_id):
        if marker.is_file():
            return False
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("locally disabled\n", encoding="utf-8")
        return True


def clear_team_workspace_leave(project_root: Path) -> bool:
    marker = team_leave_marker_path(project_root)
    workspace_id = marker.name
    with team_workspace_lock_for_id(workspace_id):
        if not marker.is_file():
            return False
        marker.unlink()
        return True


def team_workspace_config_data(
    metadata: TeamProjectMetadata | None = None, project_root: Path | None = None
) -> dict[str, object]:
    root = project_root
    if metadata is None:
        root = root or _trusted_project_root()
        metadata = load_team_project_metadata(root)
    if metadata is None:
        return {}
    enabled = root is None or not is_team_workspace_left(root)
    return {
        "team_workspace": {
            "enabled": enabled,
            "team_repository_url": metadata.team_repo_url,
            "team_branch": metadata.branch,
            "history_limit": metadata.history_limit,
            "history_scope": metadata.history_scope,
        }
    }


def _trusted_project_root() -> Path | None:
    roots = get_harness_files_manager().project_roots
    return roots[0] if roots else None


def _find_team_metadata_path(project_root: Path) -> Path | None:
    root = project_root.expanduser().resolve()
    for directory in (root, *root.parents):
        candidate = directory / TEAM_METADATA_PATH
        if candidate.is_file():
            return candidate
    return None


def _acquire_file_lock(lock_file: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        return

    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)


def _release_file_lock(lock_file: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
