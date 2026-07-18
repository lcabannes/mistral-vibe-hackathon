from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from git import Repo
import pytest

import vibe.cli.team as team_module
from vibe.cli.team import _join, _leave, run_team_command
from vibe.core.config.team_metadata import (
    is_team_workspace_left,
    load_team_project_metadata,
)
from vibe.core.team_workspace import ConnectionState, SyncError


def _args(remote: str = "origin") -> argparse.Namespace:
    return argparse.Namespace(
        team_action="join",
        team_repo_url=remote,
        branch="vibe-team-demo",
        history_scope="markers",
        history_limit=50,
    )


@pytest.mark.asyncio
async def test_join_uses_origin_branch_and_is_idempotent(tmp_path: Path) -> None:
    remote = tmp_path / "source.git"
    Repo.init(remote, bare=True)
    project = tmp_path / "project"
    repository = Repo.init(project)
    repository.create_remote("origin", str(remote))

    first_success, first_message = await _join(_args(), project)
    second_success, second_message = await _join(_args(), project)

    assert first_success is True
    assert first_message.startswith("Joined ")
    assert second_success is True
    assert second_message.startswith("Already joined ")
    metadata = load_team_project_metadata(project)
    assert metadata is not None
    assert metadata.team_repo_url == "origin"
    assert metadata.history_scope == "markers"
    assert metadata.history_limit == 50
    assert metadata.workspace_id is not None
    assert {head.name for head in Repo(remote).heads} == {"vibe-team-demo"}


@pytest.mark.asyncio
async def test_leave_is_idempotent_and_join_clears_local_marker(tmp_path: Path) -> None:
    remote = tmp_path / "source.git"
    Repo.init(remote, bare=True)
    project = tmp_path / "project"
    repository = Repo.init(project)
    repository.create_remote("origin", str(remote))
    assert (await _join(_args(), project))[0] is True
    status_before = repository.git.status("--short")

    first_success, first_message = _leave(project)
    second_success, second_message = _leave(project)

    assert first_success is True
    assert first_message.startswith("Left ")
    assert second_success is True
    assert second_message.startswith("Already left ")
    assert is_team_workspace_left(project)
    assert repository.git.status("--short") == status_before

    assert (await _join(_args(), project))[0] is True
    assert not is_team_workspace_left(project)


@pytest.mark.asyncio
async def test_failed_join_does_not_write_metadata(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Repo.init(project)

    success, message = await _join(_args(str(tmp_path / "missing.git")), project)

    assert success is False
    assert message == "transport_failed"
    assert not (project / ".vibe" / "team.toml").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        SyncError.READ_FAILED,
        SyncError.WRITE_FAILED,
        SyncError.MANIFEST_MISMATCH,
        SyncError.INVALID_ROOT,
    ],
)
async def test_degraded_join_leaves_existing_metadata_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: SyncError
) -> None:
    project = tmp_path / "project"
    Repo.init(project)
    metadata_path = project / ".vibe" / "team.toml"
    metadata_path.parent.mkdir()
    metadata_path.write_text("existing settings\n", encoding="utf-8")

    class DegradedService:
        async def start(self):
            return SimpleNamespace(
                connection_state=ConnectionState.DEGRADED, error=error
            )

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(
        team_module, "build_team_workspace_service", lambda **_kwargs: DegradedService()
    )

    success, message = await _join(_args(str(tmp_path / "team.git")), project)

    assert success is False
    assert message == error.value
    assert metadata_path.read_text(encoding="utf-8") == "existing settings\n"


def test_join_command_reports_written_settings_and_push_guidance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    remote = tmp_path / "source.git"
    Repo.init(remote, bare=True)
    project = tmp_path / "project"
    repository = Repo.init(project)
    repository.create_remote("origin", str(remote))

    result = run_team_command(_args(), project)
    output = capsys.readouterr().err

    assert result == 0
    assert "Wrote .vibe/team.toml" in output
    assert "commit and push it so trusted clones autojoin" in output
    assert "Committed settings" not in output


def test_join_command_rejects_password_without_echoing_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    Repo.init(project)
    secret = "do-not-print-this"

    result = run_team_command(
        _args(f"ssh://git:{secret}@example.com/team.git"), project
    )
    output = capsys.readouterr().err

    assert result == 1
    assert "must not contain credentials" in output
    assert secret not in output
