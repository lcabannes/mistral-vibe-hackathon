from __future__ import annotations

from pathlib import Path

from git import Repo
import pytest

from vibe.core.config import VibeConfig, build_default_orchestrator
from vibe.core.config.team_metadata import (
    TeamMetadataError,
    TeamProjectMetadata,
    load_team_project_metadata,
    write_team_project_metadata,
)
from vibe.core.team_workspace import discover_workspace_identity
from vibe.core.trusted_folders import trusted_folders_manager


def _repository(path: Path, remote: str = "git@example.com:team/project.git") -> Repo:
    repository = Repo.init(path)
    repository.create_remote("origin", remote)
    return repository


def test_metadata_is_idempotent_and_clone_stable(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _repository(first)
    _repository(second)
    first_identity = discover_workspace_identity(first)
    second_identity = discover_workspace_identity(second)
    metadata = TeamProjectMetadata(
        team_repo_url="origin", workspace_id=first_identity.workspace_id
    )

    assert first_identity.workspace_id == second_identity.workspace_id
    assert write_team_project_metadata(metadata, first) is True
    assert write_team_project_metadata(metadata, first) is False
    assert load_team_project_metadata(first) == metadata
    assert write_team_project_metadata(metadata, second) is True
    assert load_team_project_metadata(second) == metadata


def test_metadata_rejects_repository_identity_mismatch(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _repository(project)
    metadata = TeamProjectMetadata(
        team_repo_url="origin", workspace_id="ws_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )

    with pytest.raises(TeamMetadataError, match="does not match"):
        write_team_project_metadata(metadata, project)


def test_metadata_rejects_credentialed_http_remote() -> None:
    with pytest.raises(ValueError, match="must not contain credentials"):
        TeamProjectMetadata(team_repo_url="https://token@example.com/team.git")


@pytest.mark.parametrize(
    "branch", ["-unsafe", "bad branch", "topic..name", "topic.lock", "a/@{b"]
)
def test_metadata_rejects_unsafe_team_branches(branch: str) -> None:
    with pytest.raises(ValueError, match="invalid team branch"):
        TeamProjectMetadata(team_repo_url="origin", branch=branch)


def test_trusted_metadata_enables_workspace_and_runtime_override_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    _repository(project)
    write_team_project_metadata(
        TeamProjectMetadata(
            team_repo_url="origin", history_scope="messages", history_limit=75
        ),
        project,
    )
    monkeypatch.chdir(project)
    trusted_folders_manager.trust_for_session(project)

    config = VibeConfig.load()
    overridden = VibeConfig.load(team_workspace={"enabled": False})

    assert config.team_workspace.enabled is True
    assert config.team_workspace.team_repository_url == "origin"
    assert config.team_workspace.team_branch == "vibe-team-demo"
    assert config.team_workspace.history_scope == "messages"
    assert config.team_workspace.history_limit == 75
    assert overridden.team_workspace.enabled is False


def test_untrusted_metadata_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    _repository(project)
    write_team_project_metadata(TeamProjectMetadata(team_repo_url="origin"), project)
    monkeypatch.chdir(project)

    assert VibeConfig.load().team_workspace.enabled is False


@pytest.mark.asyncio
async def test_layered_config_merges_metadata_below_runtime_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    _repository(project)
    write_team_project_metadata(
        TeamProjectMetadata(team_repo_url="origin", history_scope="messages"), project
    )
    monkeypatch.chdir(project)
    trusted_folders_manager.trust_for_session(project)

    orchestrator = await build_default_orchestrator(
        data={"team_workspace": {"history_scope": "status"}}
    )

    assert orchestrator.config.team_workspace.enabled is True
    assert orchestrator.config.team_workspace.team_repository_url == "origin"
    assert orchestrator.config.team_workspace.history_scope == "status"
