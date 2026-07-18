from __future__ import annotations

import argparse
from pathlib import Path

from git import Repo
import pytest

from vibe.cli.team import _join
from vibe.core.config.team_metadata import load_team_project_metadata


def _args(remote: str = "origin") -> argparse.Namespace:
    return argparse.Namespace(
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
async def test_failed_join_does_not_write_metadata(tmp_path: Path) -> None:
    project = tmp_path / "project"
    Repo.init(project)

    success, message = await _join(_args(str(tmp_path / "missing.git")), project)

    assert success is False
    assert message == "transport_failed"
    assert not (project / ".vibe" / "team.toml").exists()
