from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
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
        "enabled_tools": ["bash", "web_search"],
        "tool_policy": "selected",
        "auto_approve": True,
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
    instance._lifecycle_locks = {}
    instance._workers = {}
    instance._session_root = tmp_path / "sessions"
    instance._registry_path = tmp_path / "runs.json"
    instance._worker_environment = {"PYTHONUNBUFFERED": "1"}
    instance._instance_id = "test-room-instance"
    instance._revision = 0
    instance._network = {}
    instance._tools = [
        {"name": "bash", "display_name": "Bash"},
        {"name": "web_search", "display_name": "Web Search"},
    ]
    instance._profiles = [
        {
            "name": "default",
            "display_name": "Default",
            "description": "Default profile",
            "safety": "neutral",
        },
        {
            "name": "auto-approve",
            "display_name": "Auto Approve",
            "description": "Auto approve",
            "safety": "yolo",
        },
    ]
    instance._runs = {}
    instance._integration_branch = "codex/test-room"
    instance._audio = room.AudioControlController(instance)
    return instance


class FakeRunner:
    def __init__(self, **_kwargs: Any) -> None:
        self.started = False
        self.stopped = False
        self.spoken: list[str] = []
        self.asked: list[str] = []
        self.answer: str | None = "build"

    @property
    def is_running(self) -> bool:
        return self.started and not self.stopped

    @property
    def phase(self) -> Any:
        return room.VoicePhase.LISTENING if self.is_running else room.VoicePhase.OFF

    last_heard = ""
    last_spoken = ""

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def ask(self, prompt: str, *, timeout: float | None = None) -> str | None:
        self.asked.append(prompt)
        return self.answer


def test_snapshot_exposes_revisioned_shared_backend_identity(store: Any) -> None:
    snapshot = store.snapshot()

    assert snapshot["api_version"] == 1
    assert snapshot["instance_id"] == "test-room-instance"
    assert snapshot["revision"] == 0
    assert snapshot["workspace"] == {
        "workdir": str(store._workdir),
        "integration_branch": "codex/test-room",
    }


def test_snapshot_includes_audio_off_state(store: Any) -> None:
    assert store.snapshot()["audio"] == {
        "enabled": False,
        "phase": "off",
        "last_heard": "",
        "last_spoken": "",
        "error": None,
    }


def test_start_and_stop_audio_toggle_runner(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(room, "VoiceControlRunner", FakeRunner)
    orchestrator = FakeWorker(room.ORCHESTRATOR_ID)
    store._workers[room.ORCHESTRATOR_ID] = orchestrator

    started = store.start_audio()
    assert started["enabled"] is True
    assert started["phase"] == "listening"
    assert store.snapshot()["audio"]["enabled"] is True
    injected = [m for m in orchestrator.sent if m["type"] == "inject_context"]
    assert injected and "voice" in injected[0]["content"].lower()

    stopped = store.stop_audio()
    assert stopped["enabled"] is False
    assert store.snapshot()["audio"]["enabled"] is False


def test_voice_command_is_sent_to_the_orchestrator(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[tuple[str, dict[str, Any], bool]] = []

    def fake_send(
        run_id: str, payload: dict[str, Any], *, interpret_commands: bool = True
    ) -> dict[str, Any]:
        sent.append((run_id, payload, interpret_commands))
        return {}

    monkeypatch.setattr(store, "send_message", fake_send)
    store._audio._on_command("spin up a research agent")

    assert len(sent) == 1
    run_id, payload, interpret = sent[0]
    assert run_id == room.ORCHESTRATOR_ID
    assert payload["content"] == "spin up a research agent"
    assert interpret is False


def test_orchestrator_reply_is_spoken(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(room, "VoiceControlRunner", FakeRunner)
    monkeypatch.setattr(store, "_persist_locked", lambda: None)
    monkeypatch.setattr(store, "_refresh_worktree_status_locked", lambda _run: None)
    orchestrator = FakeWorker(room.ORCHESTRATOR_ID)
    store._workers[room.ORCHESTRATOR_ID] = orchestrator
    store._runs[room.ORCHESTRATOR_ID] = make_run(room.ORCHESTRATOR_ID)
    store.start_audio()

    store.observe_worker_event(
        orchestrator, {"type": "assistant_final", "content": "Started a build agent."}
    )

    assert store._audio._runner.spoken == ["Started a build agent."]


def test_question_event_forwards_to_the_controller(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    forwarded: list[tuple[str, list[dict[str, Any]]]] = []
    monkeypatch.setattr(
        store._audio,
        "notify_question",
        lambda qid, questions: forwarded.append((qid, questions)),
    )
    monkeypatch.setattr(store, "_persist_locked", lambda: None)
    monkeypatch.setattr(store, "_refresh_worktree_status_locked", lambda _run: None)
    orchestrator = FakeWorker(room.ORCHESTRATOR_ID)
    store._workers[room.ORCHESTRATOR_ID] = orchestrator
    store._runs[room.ORCHESTRATOR_ID] = make_run(room.ORCHESTRATOR_ID)

    store.observe_worker_event(
        orchestrator,
        {
            "type": "question_requested",
            "request_id": "q-42",
            "questions": [{"question": "Which?", "options": []}],
        },
    )

    assert forwarded == [("q-42", [{"question": "Which?", "options": []}])]


def test_answer_question_builds_a_voice_answer(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, Any]] = []

    def fake_answer(
        run_id: str, question_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        captured.append({"run_id": run_id, "id": question_id, **payload})
        return {}

    monkeypatch.setattr(store, "answer_question", fake_answer)
    runner = FakeRunner()
    runner.start()
    runner.answer = "research"
    store._audio._runner = runner

    store._audio._answer_question(
        "q-1", [{"question": "Which category?", "options": [{"label": "Research"}]}]
    )

    assert len(captured) == 1
    assert captured[0]["run_id"] == room.ORCHESTRATOR_ID
    assert captured[0]["id"] == "q-1"
    assert captured[0]["answers"][0]["answer"] == "Research"
    assert runner.asked


def test_only_one_backend_can_own_a_vibe_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIBE_HOME", str(tmp_path))
    first = room.AgentRoomOwnerLock()
    second = room.AgentRoomOwnerLock()
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            second.acquire()
    finally:
        first.release()


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


def test_message_validates_and_forwards_image_input(store: Any) -> None:
    run = make_run()
    worker = FakeWorker(run["tool_call_id"])
    store._runs[run["tool_call_id"]] = run
    store._workers[run["tool_call_id"]] = worker

    result = store.send_message(
        run["tool_call_id"],
        {
            "content": "Inspect this",
            "client_message_id": "image-1",
            "images": [
                {"alias": "pixel.png", "mime_type": "image/png", "data": "iVBORw0KGgo="}
            ],
        },
    )

    assert result["message"]["attachments"] == [
        {"alias": "pixel.png", "mime_type": "image/png"}
    ]
    assert worker.sent[-1]["images"][0]["source"]["kind"] == "inline"
    with pytest.raises(ValueError, match="valid base64"):
        store.send_message(
            run["tool_call_id"],
            {
                "content": "Bad image",
                "images": [
                    {
                        "alias": "bad.png",
                        "mime_type": "image/png",
                        "data": "not-base64!",
                    }
                ],
            },
        )


def test_message_restarts_a_stopped_agent_in_place(
    monkeypatch: pytest.MonkeyPatch, store: Any, tmp_path: Path
) -> None:
    prepared = PreparedWorktree(
        name="room-worker-existing",
        branch="room-worker-existing",
        root=tmp_path / "worktree",
        path=tmp_path / "worktree",
        repo_root=tmp_path,
        base_commit="deadbeef",
        created=False,
        branch_created=False,
    )
    prepared.path.mkdir()
    run = make_run()
    run.update({
        "runtime_live": False,
        "state": "stopped",
        "worktree_name": prepared.name,
        "worktree_path": str(prepared.path),
        "worktree_root": str(prepared.root),
        "branch": prepared.branch,
        "base_commit": prepared.base_commit,
    })
    run["conversation"].append(
        store._message("assistant", "Earlier answer", None, "succeeded")
    )
    store._runs[run["tool_call_id"]] = run
    launches: list[Any] = []

    class RelaunchedWorker(FakeWorker):
        def __init__(
            self, _store: Any, run_id: str, *_args: Any, **kwargs: Any
        ) -> None:
            super().__init__(run_id)
            self.ready.clear()
            self.process = type("Process", (), {"poll": lambda self: None})()
            launches.append(kwargs)

        def start(self) -> None:
            self.ready.set()

    monkeypatch.setattr(room, "prepare_worktree_session", lambda *_args: prepared)
    monkeypatch.setattr(room, "AgentWorker", RelaunchedWorker)

    result = store.send_message(
        run["tool_call_id"],
        {"content": "Continue", "client_message_id": "resume-message"},
    )

    assert result["run"]["tool_call_id"] == run["tool_call_id"]
    assert result["run"]["worktree_path"] == str(prepared.path)
    assert [item["content"] for item in run["conversation"]] == [
        "Earlier answer",
        "Continue",
    ]
    assert run["runtime_live"] is True
    assert launches[0]["resume_session_id"] == "child"
    assert launches[0]["enabled_tools"] == ("bash", "web_search")
    assert launches[0]["force_auto_approve"] is True
    assert store._workers[run["tool_call_id"]].sent[-1]["content"] == "Continue"


def test_create_defaults_to_yolo_and_accepts_zero_tools(store: Any) -> None:
    captured: dict[str, Any] = {}

    def launch(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return make_run()

    store._launch_worker = launch

    store.create({"task": "Work", "enabled_tools": []})

    assert captured["profile"] == "default"
    assert captured["enabled_tools"] == []
    assert captured["auto_approve"] is True


def test_all_selected_tools_means_unrestricted_remote_discovery(store: Any) -> None:
    assert store._enabled_tools({}) is None
    assert store._enabled_tools({"enabled_tools": ["bash", "web_search"]}) is None
    assert store._enabled_tools({"enabled_tools": []}) == []


def test_create_rejects_unknown_tools(store: Any) -> None:
    with pytest.raises(ValueError, match="Unknown tools: shell_everything"):
        store.create({"task": "Work", "enabled_tools": ["shell_everything"]})


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
    run["approvals"].append({"id": "old-approval", "status": "pending"})
    run["questions"].append({"id": "old-question", "status": "pending"})
    worker = FakeWorker(run["tool_call_id"])
    store._runs[run["tool_call_id"]] = run
    store._workers[run["tool_call_id"]] = worker

    stopped = store.stop(run["tool_call_id"])

    assert stopped["state"] == "stopped"
    assert stopped["runtime_live"] is False
    assert stopped["resumable"] is True
    assert stopped["worktree_path"] == "/tmp/worker"
    assert stopped["conversation"][0]["status"] == "cancelled"
    assert stopped["approvals"][0]["status"] == "expired"
    assert stopped["questions"][0]["status"] == "expired"
    assert worker.stopped is True


def test_stale_worker_exit_cannot_stop_a_replacement(store: Any) -> None:
    run = make_run()
    old_worker = FakeWorker(run["tool_call_id"])
    replacement = FakeWorker(run["tool_call_id"])
    store._runs[run["tool_call_id"]] = run
    store._workers[run["tool_call_id"]] = replacement

    store.observe_worker_exit(old_worker, 1)

    assert run["runtime_live"] is True
    assert store._workers[run["tool_call_id"]] is replacement

    store.observe_worker_event(
        old_worker, {"type": "state", "state": "failed", "error": "stale failure"}
    )

    assert run["state"] == "idle"
    assert run["error"] is None


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
        def __init__(
            self, _store: Any, run_id: str, *_args: Any, **_kwargs: Any
        ) -> None:
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


def test_merge_serializes_same_agent_resume(
    monkeypatch: pytest.MonkeyPatch, store: Any, tmp_path: Path
) -> None:
    run = make_run()
    worktree = tmp_path / "worker"
    worktree.mkdir()
    run.update({
        "runtime_live": False,
        "state": "stopped",
        "worktree_path": str(worktree),
    })
    store._runs[run["tool_call_id"]] = run
    validation_started = Event()
    release_validation = Event()
    ensure_called = Event()
    worker = FakeWorker(run["tool_call_id"])

    def validate(*_args: Any) -> dict[str, str]:
        validation_started.set()
        assert release_validation.wait(1)
        return {"merge_commit": "merged", "validation_summary": "ok"}

    def ensure(_run_id: str) -> FakeWorker:
        ensure_called.set()
        return worker

    monkeypatch.setattr(store, "_validate_and_merge", validate)
    monkeypatch.setattr(store, "_ensure_worker", ensure)
    merge_thread = Thread(target=store.merge, args=(run["tool_call_id"],))
    send_thread = Thread(
        target=store.send_message, args=(run["tool_call_id"], {"content": "Continue"})
    )

    merge_thread.start()
    assert validation_started.wait(1)
    send_thread.start()
    assert not ensure_called.wait(0.05)
    release_validation.set()
    merge_thread.join(1)
    send_thread.join(1)

    assert not merge_thread.is_alive()
    assert not send_thread.is_alive()
    assert ensure_called.is_set()


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


def test_stale_orchestrator_cannot_control_the_room(store: Any) -> None:
    orchestrator = make_run(room.ORCHESTRATOR_ID)
    orchestrator["is_orchestrator"] = True
    stale = FakeWorker(room.ORCHESTRATOR_ID)
    current = FakeWorker(room.ORCHESTRATOR_ID)
    store._runs[room.ORCHESTRATOR_ID] = orchestrator
    store._workers[room.ORCHESTRATOR_ID] = current
    calls: list[dict[str, Any]] = []
    store.create = lambda payload: calls.append(payload)

    store._handle_remote_request(
        stale,
        {
            "request_id": "stale-request",
            "operation": "start",
            "payload": {"profile": "default", "task": "work"},
        },
    )

    assert calls == []
    assert stale.sent == []


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
