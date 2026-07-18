from __future__ import annotations

from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from vibe.core.worktree import PreparedWorktree
from web.agent_room_test_support import load_agent_room_server

room = load_agent_room_server()


class FakeWorker:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.ready = Event()
        self.ready.set()
        self.sent: list[dict[str, Any]] = []
        self.stopped = False

    def send(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    def stop(self) -> None:
        self.stopped = True


def make_run(run_id: str = "worker-1") -> dict[str, Any]:
    return {
        "tool_call_id": run_id,
        "parent_session_id": "parent",
        "child_session_id": "child",
        "agent_name": "default",
        "agent_display_name": "Worker",
        "task": "Initial task",
        "state": "idle",
        "started_at": 1.0,
        "updated_at": 1.0,
        "current_activity": None,
        "turns_used": 0,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "context_tokens": 0,
        "context_limit": 100_000,
        "estimated_cost_usd": 0.0,
        "model": "model",
        "is_primary": False,
        "is_orchestrator": False,
        "group_id": "unassigned",
        "coat": "orange",
        "source": "live",
        "runtime_live": True,
        "events": [],
        "conversation": [],
        "approvals": [],
        "questions": [],
        "error": None,
        "resumable": True,
        "queued_messages": 0,
        "worktree_path": "/tmp/worker",
        "worktree_root": "/tmp/worker",
        "worktree_name": "room-worker",
        "branch": "room-worker",
        "base_commit": "abc",
        "worktree_dirty": False,
        "uncommitted_files": 0,
        "new_commit_count": 0,
        "merge_status": "not_ready",
    }


@pytest.fixture
def store(tmp_path: Path) -> Any:
    instance = object.__new__(room.AgentRoomStore)
    instance._workdir = tmp_path
    instance._lock = room.RLock()
    instance._workers = {}
    instance._session_root = tmp_path / "sessions"
    instance._registry_path = tmp_path / "runs.json"
    instance._worker_environment = {"PYTHONUNBUFFERED": "1"}
    instance._network = {}
    instance._profiles = [
        {
            "name": "default",
            "display_name": "Default",
            "description": "Default profile",
            "safety": "neutral",
        }
    ]
    instance._runs = {}
    return instance


def test_auto_network_bypasses_a_broken_inherited_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in room.PROXY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:9")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    provider = SimpleNamespace(api_key_env_var="MISTRAL_API_KEY")
    config = SimpleNamespace(
        get_active_model=lambda: object(),
        get_provider_for_model=lambda _model: provider,
    )
    monkeypatch.setattr(room.VibeConfig, "load", lambda: config)
    monkeypatch.setattr(room, "resolve_api_key", lambda _env_key: "keyring-key")

    def probe(*, trust_env: bool, credential: str | None = None) -> dict[str, Any]:
        if credential:
            return {"reachable": True, "status": 200, "error": None}
        if trust_env:
            return {"reachable": False, "status": None, "error": "proxy rejected"}
        return {"reachable": True, "status": 401, "error": None}

    monkeypatch.setattr(room, "probe_mistral_transport", probe)

    environment, status = room.resolve_worker_network("auto")

    assert status["selected_mode"] == "direct"
    assert status["proxy_reachable"] is False
    assert status["direct_reachable"] is True
    assert status["authenticated"] is True
    assert status["credential_source"] == "keyring"
    assert all(key not in environment for key in room.PROXY_ENV_KEYS)


def test_inherit_network_mode_preserves_a_configured_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in room.PROXY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    proxy = "http://localhost:8080"
    monkeypatch.setenv("HTTPS_PROXY", proxy)
    provider = SimpleNamespace(api_key_env_var="MISTRAL_API_KEY")
    config = SimpleNamespace(
        get_active_model=lambda: object(),
        get_provider_for_model=lambda _model: provider,
    )
    monkeypatch.setattr(room.VibeConfig, "load", lambda: config)
    monkeypatch.setattr(room, "resolve_api_key", lambda _env_key: "environment-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "environment-key")
    monkeypatch.setattr(
        room,
        "probe_mistral_transport",
        lambda **_kwargs: {"reachable": True, "status": 200, "error": None},
    )

    environment, status = room.resolve_worker_network("inherit")

    assert status["selected_mode"] == "inherit"
    assert status["authenticated"] is True
    assert status["credential_source"] == "environment"
    assert environment["HTTPS_PROXY"] == proxy


def test_message_acceptance_is_idempotent_and_fifo(store: Any) -> None:
    run = make_run()
    worker = FakeWorker(run["tool_call_id"])
    store._runs[run["tool_call_id"]] = run
    store._workers[run["tool_call_id"]] = worker

    first = store.send_message(
        run["tool_call_id"], {"content": "first", "client_message_id": "client-1"}
    )
    duplicate = store.send_message(
        run["tool_call_id"], {"content": "first", "client_message_id": "client-1"}
    )
    second = store.send_message(
        run["tool_call_id"], {"content": "second", "client_message_id": "client-2"}
    )

    assert first["message"]["id"] == duplicate["message"]["id"]
    assert [item["content"] for item in run["conversation"]] == ["first", "second"]
    assert [item["content"] for item in worker.sent] == ["first", "second"]
    assert first["message"]["id"] != second["message"]["id"]


def test_slash_command_is_typed_and_never_forwarded(store: Any) -> None:
    run = make_run()
    worker = FakeWorker(run["tool_call_id"])
    store._runs[run["tool_call_id"]] = run
    store._workers[run["tool_call_id"]] = worker

    result = store.send_message(
        run["tool_call_id"], {"content": "/status", "client_message_id": "status-1"}
    )

    assert result["command"] == "/status"
    assert worker.sent == []
    assert run["conversation"][-1]["role"] == "system"
    with pytest.raises(ValueError, match="Unknown command"):
        store.send_message(
            run["tool_call_id"], {"content": "/shell rm", "client_message_id": "bad-1"}
        )


def test_approval_first_response_wins(store: Any) -> None:
    run = make_run()
    run["approvals"].append({"id": "approval-1", "status": "pending"})
    worker = FakeWorker(run["tool_call_id"])
    store._runs[run["tool_call_id"]] = run
    store._workers[run["tool_call_id"]] = worker

    resolved = store.resolve_approval(
        run["tool_call_id"], "approval-1", {"decision": "approve_once"}
    )

    assert resolved["status"] == "approved"
    assert worker.sent[-1]["type"] == "approval_response"
    with pytest.raises(RuntimeError, match="already been resolved"):
        store.resolve_approval(run["tool_call_id"], "approval-1", {"decision": "deny"})


def test_worker_events_update_message_usage_and_memory(store: Any) -> None:
    run = make_run()
    message = store._message("user", "work", "client", "queued")
    run["conversation"].append(message)
    worker = FakeWorker(run["tool_call_id"])
    store._runs[run["tool_call_id"]] = run
    store._workers[run["tool_call_id"]] = worker

    store.observe_worker_event(
        worker,
        {
            "type": "state",
            "state": "running",
            "activity": "Thinking",
            "message_id": message["id"],
            "queued_messages": 0,
        },
    )
    store.observe_worker_event(
        worker,
        {
            "type": "usage",
            "turns_used": 3,
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "context_tokens": 1_500,
            "context_limit": 10_000,
            "estimated_cost_usd": 0.03,
            "model": "devstral",
        },
    )

    assert message["status"] == "running"
    assert run["usage"] == {"prompt_tokens": 120, "completion_tokens": 30}
    assert run["context_tokens"] == 1_500
    assert run["context_limit"] == 10_000
    assert run["estimated_cost_usd"] == 0.03


def test_stop_keeps_the_worktree_and_history(store: Any) -> None:
    run = make_run()
    run["conversation"].append(store._message("user", "work", "client", "running"))
    worker = FakeWorker(run["tool_call_id"])
    store._runs[run["tool_call_id"]] = run
    store._workers[run["tool_call_id"]] = worker

    stopped = store.stop(run["tool_call_id"])

    assert stopped["state"] == "cancelled"
    assert stopped["runtime_live"] is False
    assert stopped["worktree_path"] == "/tmp/worker"
    assert stopped["conversation"][0]["status"] == "cancelled"
    assert worker.stopped is True


def test_launch_records_a_distinct_worktree(
    monkeypatch: pytest.MonkeyPatch, store: Any, tmp_path: Path
) -> None:
    prepared = PreparedWorktree(
        name="room-worker-1234",
        branch="room-worker-1234",
        root=tmp_path / "worktree",
        path=tmp_path / "worktree",
        repo_root=tmp_path,
        base_commit="deadbeef",
        created=True,
        branch_created=True,
    )
    prepared.path.mkdir()

    class LaunchWorker(FakeWorker):
        def __init__(self, _store: Any, run_id: str, *_args: Any) -> None:
            super().__init__(run_id)
            self.process = type("Process", (), {"poll": lambda self: None})()

        def start(self) -> None:
            self.ready.set()

    monkeypatch.setattr(room, "prepare_worktree_session", lambda *_args: prepared)
    monkeypatch.setattr(room, "AgentWorker", LaunchWorker)

    run = store._launch_worker(
        profile="default",
        display_name="Worker",
        group_id="unassigned",
        task="work",
        is_orchestrator=False,
        client_message_id="client",
    )

    assert run["worktree_path"] == str(prepared.path)
    assert run["branch"] == prepared.branch
    assert run["base_commit"] == "deadbeef"
    assert store._workers[run["tool_call_id"]].sent[0]["type"] == "prompt"


def test_merge_requires_stopped_clean_committed_worker(store: Any) -> None:
    run = make_run()
    store._runs[run["tool_call_id"]] = run

    with pytest.raises(ValueError, match="Stop the agent"):
        store.merge(run["tool_call_id"])


def test_orchestrator_remote_start_returns_typed_snapshot(store: Any) -> None:
    orchestrator = make_run(room.ORCHESTRATOR_ID)
    orchestrator["is_orchestrator"] = True
    worker = FakeWorker(room.ORCHESTRATOR_ID)
    store._runs[room.ORCHESTRATOR_ID] = orchestrator
    store._workers[room.ORCHESTRATOR_ID] = worker
    created = make_run("created-worker")
    store.create = lambda payload: created

    store._handle_remote_request(
        worker,
        {
            "request_id": "request-1",
            "operation": "start",
            "payload": {"profile": "default", "task": "work", "name": "Builder"},
        },
    )

    response = worker.sent[-1]
    assert response["ok"] is True
    assert response["result"]["agent_id"] == "created-worker"
    assert response["result"]["state"] == "idle"


def test_orchestrator_commands_use_a_fixed_allowlist(store: Any) -> None:
    orchestrator = make_run(room.ORCHESTRATOR_ID)
    orchestrator["is_orchestrator"] = True
    store._runs[room.ORCHESTRATOR_ID] = orchestrator
    calls: list[str] = []
    store.cancel = lambda agent_id: calls.append(agent_id)

    store._record_control_request({"action": "command", "command": "/cancel worker-1"})

    assert calls == ["worker-1"]
    with pytest.raises(ValueError, match="Supported Agent Room commands"):
        store._record_control_request({
            "action": "command",
            "command": "/shell worker-1",
        })
