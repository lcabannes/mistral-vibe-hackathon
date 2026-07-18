from __future__ import annotations

from pathlib import Path

from git import Repo
import pytest

from vibe.core.team_workspace import (
    ActivityState,
    HistoryScope,
    PrivacyMode,
    resolve_team_repository_url,
)
from vibe.core.team_workspace.git_transport import (
    GitTeamWorkspaceError,
    GitTeamWorkspaceTransport,
)
from vibe.core.team_workspace.service import build_team_workspace_service


def _bare_remote(path: Path) -> Path:
    Repo.init(path, bare=True)
    return path


def _transport(remote: Path, checkout: Path) -> GitTeamWorkspaceTransport:
    return GitTeamWorkspaceTransport(
        remote_url=str(remote),
        checkout_dir=checkout,
        branch="vibe-team-demo",
        timeout_seconds=5,
    )


def _write_client_state(transport: GitTeamWorkspaceTransport, name: str) -> None:
    transport.prepare()
    path = transport.materialization_root / "workspace" / "clients" / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "presence.json").write_text(f'{{"client":"{name}"}}', encoding="utf-8")


def test_two_clients_converge_through_real_bare_remote(tmp_path: Path) -> None:
    remote = _bare_remote(tmp_path / "team.git")
    first = _transport(remote, tmp_path / "first")
    second = _transport(remote, tmp_path / "second")
    _write_client_state(first, "first")
    _write_client_state(second, "second")

    first.sync()
    second.sync()
    first.sync()

    assert (
        first.materialization_root
        / "workspace"
        / "clients"
        / "second"
        / "presence.json"
    ).is_file()
    assert (
        second.materialization_root
        / "workspace"
        / "clients"
        / "first"
        / "presence.json"
    ).is_file()
    assert "vibe-team-demo" in Repo(remote).heads


def test_offline_commit_pushes_after_remote_returns(tmp_path: Path) -> None:
    remote = _bare_remote(tmp_path / "team.git")
    unavailable = tmp_path / "team-offline.git"
    first = _transport(remote, tmp_path / "first")
    second = _transport(remote, tmp_path / "second")
    _write_client_state(first, "first")
    first.sync()

    remote.rename(unavailable)
    _write_client_state(first, "offline-update")
    with pytest.raises(GitTeamWorkspaceError):
        first.sync()
    unavailable.rename(remote)

    first.sync()
    _write_client_state(second, "second")
    second.sync()

    assert (
        second.materialization_root
        / "workspace"
        / "clients"
        / "offline-update"
        / "presence.json"
    ).is_file()


def test_origin_sentinel_uses_separate_branch_without_touching_source_checkout(
    tmp_path: Path,
) -> None:
    remote = _bare_remote(tmp_path / "source.git")
    source = tmp_path / "source"
    repo = Repo.init(source, initial_branch="main")
    (source / "README.md").write_text("source\n", encoding="utf-8")
    repo.index.add(["README.md"])
    repo.index.commit("Initial source commit")
    repo.create_remote("origin", str(remote))
    repo.remote("origin").push("main:main")
    source_head = repo.head.commit.hexsha

    resolved = resolve_team_repository_url(source, "origin")
    assert resolved == str(remote)
    assert resolved is not None
    transport = GitTeamWorkspaceTransport(
        remote_url=resolved,
        checkout_dir=tmp_path / "team-cache",
        branch="vibe-team-demo",
    )
    _write_client_state(transport, "demo")
    transport.sync()

    assert repo.active_branch.name == "main"
    assert repo.head.commit.hexsha == source_head
    assert not repo.is_dirty(untracked_files=True)
    assert {head.name for head in Repo(remote).heads} == {"main", "vibe-team-demo"}


@pytest.mark.asyncio
async def test_two_services_converge_activity_through_bare_git_remote(
    tmp_path: Path,
) -> None:
    remote = _bare_remote(tmp_path / "team.git")
    project = tmp_path / "project"
    project.mkdir()
    first = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint="first@example.com",
        privacy_mode=PrivacyMode.SUMMARIES,
        history_scope=HistoryScope.MESSAGES,
    )
    second = build_team_workspace_service(
        enabled=True,
        shared_root=None,
        project_root=project,
        team_repository_url=str(remote),
        cache_root=tmp_path / "cache",
        identity_hint="second@example.com",
        privacy_mode=PrivacyMode.SUMMARIES,
        history_scope=HistoryScope.MESSAGES,
    )
    try:
        await first.start()
        await first.publish_activity(
            local_run_id="primary",
            agent_name="default",
            agent_display_name="Default",
            state=ActivityState.WORKING,
        )
        await second.start()

        first_snapshot = await first.refresh()
        second_snapshot = await second.refresh()

        assert len(first_snapshot.members) == 2
        assert len(second_snapshot.members) == 2
        assert first_snapshot.runs[0].state is ActivityState.WORKING
        assert second_snapshot.runs[0].state is ActivityState.WORKING
    finally:
        await first.stop()
        await second.stop()
