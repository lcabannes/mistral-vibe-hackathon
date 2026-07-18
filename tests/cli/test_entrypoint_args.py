from __future__ import annotations

from pathlib import Path

import pytest

from vibe.cli.entrypoint import _run_agent_room_server, parse_arguments


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


def test_server_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    defaults = _parse(monkeypatch, ["--server"])
    configured = _parse(
        monkeypatch,
        ["--server", "--server-port", "4183", "--server-network-mode", "direct"],
    )

    assert defaults.server is True
    assert defaults.server_port == 4173
    assert defaults.server_network_mode == "auto"
    assert configured.server_port == 4183
    assert configured.server_network_mode == "direct"


def test_server_rejects_invalid_port(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--server", "--server-port", "70000"])


def test_server_executes_shared_backend_for_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executed: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        "vibe.cli.entrypoint.os.execv",
        lambda executable, command: executed.append((executable, command)),
    )

    _run_agent_room_server(tmp_path, port=4183, network_mode="direct")

    assert len(executed) == 1
    executable, command = executed[0]
    assert command[0] == executable
    assert command[1].endswith("web/agent-room/server.py")
    assert command[2:] == [
        "--port",
        "4183",
        "--workdir",
        str(tmp_path.resolve()),
        "--network-mode",
        "direct",
    ]


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
