from __future__ import annotations

import pytest

from vibe.cli.entrypoint import parse_arguments


def _parse(monkeypatch: pytest.MonkeyPatch, argv: list[str]):
    monkeypatch.setattr("sys.argv", ["vibe", *argv])
    return parse_arguments()


def test_disabled_tools_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, [])
    assert args.disabled_tools is None


def test_disabled_tools_appends_multiple(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, ["--disabled-tools", "bash", "--disabled-tools", "web*"])
    assert args.disabled_tools == ["bash", "web*"]


def test_enabled_and_disabled_tools_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse(monkeypatch, ["--enabled-tools", "read", "--disabled-tools", "bash"])
    assert args.enabled_tools == ["read"]
    assert args.disabled_tools == ["bash"]


def test_team_join_defaults_to_marker_only_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse(monkeypatch, ["team", "join", "origin"])

    assert args.command == "team"
    assert args.team_action == "join"
    assert args.team_repo_url == "origin"
    assert args.branch == "vibe-team-demo"
    assert args.history_scope == "markers"
    assert args.history_limit == 50


def test_team_join_accepts_explicit_message_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse(
        monkeypatch,
        ["team", "join", "origin", "--history", "messages", "--history-limit", "9"],
    )

    assert args.history_scope == "messages"
    assert args.history_limit == 9
