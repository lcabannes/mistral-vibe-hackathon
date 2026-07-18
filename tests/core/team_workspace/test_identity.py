from __future__ import annotations

from pathlib import Path

from git import Repo

from vibe.core.team_workspace import (
    derive_run_id,
    discover_workspace_identity,
    normalize_project_remote,
    resolve_member_identity,
)


def test_remote_normalization_removes_credentials_and_transport() -> None:
    assert (
        normalize_project_remote("https://token@example.com/Org/Repo.git")
        == "example.com/org/repo"
    )
    assert (
        normalize_project_remote("git@example.com:Org/Repo.git")
        == "example.com/org/repo"
    )


def test_nested_paths_in_same_repository_share_workspace_identity(
    tmp_path: Path,
) -> None:
    repo = Repo.init(tmp_path)
    repo.create_remote("origin", "git@example.com:Team/Project.git")
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)

    root_identity = discover_workspace_identity(tmp_path)
    nested_identity = discover_workspace_identity(nested)

    assert root_identity == nested_identity
    assert root_identity.display_name == tmp_path.name


def test_member_identity_is_not_keyed_by_display_or_os_username_alone() -> None:
    one, display = resolve_member_identity(
        "ws_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        member_name="Same Name",
        identity_hint="one@example.com",
    )
    two, _ = resolve_member_identity(
        "ws_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        member_name="Same Name",
        identity_hint="two@example.com",
    )

    assert display == "Same Name"
    assert one != two


def test_run_id_is_stable_and_project_scoped() -> None:
    one = derive_run_id("ws_aaaaaaaa", "member_bbbbbbbb", "local-run")
    two = derive_run_id("ws_aaaaaaaa", "member_bbbbbbbb", "local-run")
    other = derive_run_id("ws_cccccccc", "member_bbbbbbbb", "local-run")

    assert one == two
    assert one != other
