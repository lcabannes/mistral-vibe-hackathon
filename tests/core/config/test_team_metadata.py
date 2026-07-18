from __future__ import annotations

from pathlib import Path

from git import Repo
import pytest
import tomli_w

from vibe.core.config import VibeConfig, build_default_orchestrator
from vibe.core.config.team_metadata import (
    TeamMetadataError,
    TeamProjectMetadata,
    clear_team_workspace_leave,
    leave_team_workspace,
    load_team_project_metadata,
    team_leave_marker_path,
    team_workspace_config_data,
    write_team_project_metadata,
)
from vibe.core.team_workspace import discover_workspace_identity
from vibe.core.trusted_folders import trusted_folders_manager


def _repository(path: Path, remote: str = "git@example.com:team/project.git") -> Repo:
    repository = Repo.init(path)
    repository.create_remote("origin", remote)
    return repository


def _write_production_default(config_dir: Path) -> None:
    with (config_dir / "config.toml").open("wb") as file:
        tomli_w.dump(VibeConfig.create_default(), file)


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


@pytest.mark.parametrize(
    "remote",
    [
        "https://token@example.com/team.git",
        "ssh://user:password@example.com/team.git",
        "ftp://user:password@example.com/team.git",
        "file://user:password@localhost/team.git",
    ],
)
def test_metadata_rejects_credentialed_remote(remote: str) -> None:
    with pytest.raises(ValueError, match="must not contain credentials"):
        TeamProjectMetadata(team_repo_url=remote)


@pytest.mark.parametrize(
    "remote", ["ssh://git@example.com/team.git", "git@example.com:team/project.git"]
)
def test_metadata_allows_non_secret_ssh_usernames(remote: str) -> None:
    assert TeamProjectMetadata(team_repo_url=remote).team_repo_url == remote


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


@pytest.mark.asyncio
async def test_production_default_config_does_not_override_committed_team_metadata(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    _repository(project)
    write_team_project_metadata(
        TeamProjectMetadata(team_repo_url="origin", history_scope="messages"), project
    )
    _write_production_default(config_dir)
    monkeypatch.chdir(project)
    trusted_folders_manager.trust_for_session(project)

    legacy = VibeConfig.load()
    orchestrator = await build_default_orchestrator()

    assert legacy.team_workspace.enabled is True
    assert legacy.team_workspace.team_repository_url == "origin"
    assert legacy.team_workspace.history_scope == "messages"
    assert orchestrator.config.team_workspace.enabled is True
    assert orchestrator.config.team_workspace.team_repository_url == "origin"
    assert orchestrator.config.team_workspace.history_scope == "messages"


@pytest.mark.asyncio
async def test_workspace_leave_marker_overrides_metadata_without_mutating_project(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    repository = _repository(project)
    write_team_project_metadata(
        TeamProjectMetadata(team_repo_url="origin", history_scope="messages"), project
    )
    (project / ".vibe" / "config.toml").write_text(
        tomli_w.dumps(VibeConfig.create_default()), encoding="utf-8"
    )
    repository.index.add([".vibe/team.toml", ".vibe/config.toml"])
    repository.index.commit("Configure project")
    _write_production_default(config_dir)
    status_before = repository.git.status("--short")

    assert leave_team_workspace(project) is True
    assert leave_team_workspace(project) is False
    assert team_leave_marker_path(project).is_relative_to(config_dir)
    assert repository.git.status("--short") == status_before

    monkeypatch.chdir(project)
    trusted_folders_manager.trust_for_session(project)
    legacy = VibeConfig.load()
    orchestrator = await build_default_orchestrator()

    assert legacy.team_workspace.enabled is False
    assert legacy.team_workspace.team_repository_url == "origin"
    assert legacy.team_workspace.history_scope == "messages"
    assert orchestrator.config.team_workspace.enabled is False
    assert orchestrator.config.team_workspace.team_repository_url == "origin"
    assert orchestrator.config.team_workspace.history_scope == "messages"

    assert clear_team_workspace_leave(project) is True
    assert clear_team_workspace_leave(project) is False
    assert VibeConfig.load().team_workspace.enabled is True


def test_leave_marker_is_scoped_to_one_workspace(
    tmp_path: Path, config_dir: Path
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _repository(first, "git@example.com:team/first.git")
    _repository(second, "git@example.com:team/second.git")
    write_team_project_metadata(TeamProjectMetadata(team_repo_url="origin"), first)
    write_team_project_metadata(TeamProjectMetadata(team_repo_url="origin"), second)

    leave_team_workspace(first)

    assert team_leave_marker_path(first).is_file()
    assert team_leave_marker_path(first).is_relative_to(config_dir)
    assert not team_leave_marker_path(second).exists()
    assert team_workspace_config_data(project_root=first)["team_workspace"] == {
        "enabled": False,
        "team_repository_url": "origin",
        "team_branch": "vibe-team-demo",
        "history_limit": 50,
        "history_scope": "markers",
    }
    assert team_workspace_config_data(project_root=second)["team_workspace"] == {
        "enabled": True,
        "team_repository_url": "origin",
        "team_branch": "vibe-team-demo",
        "history_limit": 50,
        "history_scope": "markers",
    }
