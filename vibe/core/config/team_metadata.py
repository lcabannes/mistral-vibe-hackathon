from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
import tomllib
from typing import Literal
from urllib.parse import urlparse

from git import InvalidGitRepositoryError, Repo
from pydantic import BaseModel, ConfigDict, Field, field_validator
import tomli_w

from vibe.core.config.harness_files import get_harness_files_manager
from vibe.core.team_workspace.identity import discover_workspace_identity
from vibe.core.utils.io import read_safe

TEAM_METADATA_PATH = Path(".vibe") / "team.toml"


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
    def reject_http_credentials(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"} and parsed.username is not None:
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
    path = resolve_project_root(root) / TEAM_METADATA_PATH
    if not path.is_file():
        return None
    try:
        metadata = TeamProjectMetadata.model_validate(
            tomllib.loads(read_safe(path).text)
        )
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        raise TeamMetadataError(f"Invalid team metadata at {path}: {exc}") from exc

    identity = discover_workspace_identity(resolve_project_root(root))
    if metadata.workspace_id and metadata.workspace_id != identity.workspace_id:
        raise TeamMetadataError(
            "Team metadata workspace_id does not match this repository remote"
        )
    return metadata


def write_team_project_metadata(
    metadata: TeamProjectMetadata, project_root: Path | None = None
) -> bool:
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


def team_workspace_config_data(
    metadata: TeamProjectMetadata | None = None,
) -> dict[str, object]:
    metadata = metadata or load_team_project_metadata()
    if metadata is None:
        return {}
    return {
        "team_workspace": {
            "enabled": True,
            "team_repository_url": metadata.team_repo_url,
            "team_branch": metadata.branch,
            "history_limit": metadata.history_limit,
            "history_scope": metadata.history_scope,
        }
    }


def _trusted_project_root() -> Path | None:
    roots = get_harness_files_manager().project_roots
    return roots[0] if roots else None
