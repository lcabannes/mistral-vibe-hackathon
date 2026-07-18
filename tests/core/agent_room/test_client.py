from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from vibe.core.agent_room.client import (
    AgentRoomClient,
    discover_agent_room,
    ensure_agent_room_backend,
    launch_agent_room_backend,
)
from vibe.core.agents.models import ManagedAgentState


def _run(run_id: str = "agent-1") -> dict[str, Any]:
    return {
        "tool_call_id": run_id,
        "parent_session_id": None,
        "child_session_id": f"session-{run_id}",
        "agent_name": "default",
        "agent_display_name": "Builder",
        "task": "Build the feature",
        "state": "idle",
        "started_at": 1.0,
        "updated_at": 2.0,
        "current_activity": "Ready",
        "turns_used": 3,
        "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        "context_tokens": 150,
        "context_limit": 1000,
        "estimated_cost_usd": 0.012,
        "model": "mistral-vibe-cli-latest",
        "group_id": "implementation",
        "runtime_live": True,
        "conversation": [
            {
                "id": "message-1",
                "role": "assistant",
                "content": "Initial work is complete.",
                "status": "succeeded",
            }
        ],
        "approvals": [{"id": "approval-1", "status": "pending"}],
        "questions": [{"id": "question-1", "status": "pending"}],
        "queued_messages": 0,
        "worktree_path": "/tmp/agent-1",
        "branch": "room-agent-1",
    }


class _RoomAPI:
    def __init__(self) -> None:
        self.revision = 1
        self.runs = [_run()]
        self.requests: list[tuple[str, str, dict[str, Any]]] = []

    def snapshot(self) -> dict[str, Any]:
        return {
            "api_version": 1,
            "instance_id": "room-instance",
            "revision": self.revision,
            "connected": True,
            "workspace": {"integration_branch": "codex/agent-unified"},
            "activities": self.runs,
            "profiles": [
                {"name": "default", "display_name": "Default"},
                {"name": "orchestrator", "display_name": "Orchestrator"},
            ],
            "tools": [],
            "coordination": {},
            "network": {},
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content or b"{}")
        self.requests.append((request.method, request.url.path, payload))
        if request.method == "GET":
            return httpx.Response(200, json=self.snapshot())
        parts = request.url.path.strip("/").split("/")
        if request.url.path == "/api/agent-runs":
            created = _run("agent-2")
            created.update({
                "task": payload["task"],
                "agent_display_name": payload["display_name"],
                "state": "requested",
            })
            self.runs.append(created)
            self.revision += 1
            return httpx.Response(202, json=created)
        run = next(item for item in self.runs if item["tool_call_id"] == parts[2])
        if parts[3] == "messages":
            run["conversation"].append({
                "id": "message-cli",
                "client_message_id": payload["client_message_id"],
                "role": "user",
                "content": payload["content"],
                "status": "queued",
            })
            run["state"] = "working"
            run["updated_at"] += 1
            self.revision += 1
            return httpx.Response(202, json={"run": run})
        if parts[3] == "cancel":
            return httpx.Response(200, json=run)
        if parts[3] == "stop":
            run["state"] = "stopped"
            run["runtime_live"] = False
            run["updated_at"] += 1
            self.revision += 1
            return httpx.Response(200, json=run)
        if parts[3] == "approvals":
            return httpx.Response(
                200, json={"id": parts[4], "status": payload["decision"]}
            )
        if parts[3] == "questions":
            return httpx.Response(200, json={"id": parts[4], "status": "answered"})
        return httpx.Response(404, json={"error": "not found"})


@pytest.mark.asyncio
async def test_two_clients_share_runs_conversations_and_commands() -> None:
    api = _RoomAPI()
    transport = httpx.MockTransport(api)
    cli = AgentRoomClient("http://127.0.0.1:4173", "cli-session", transport=transport)
    web_peer = AgentRoomClient(
        "http://127.0.0.1:4173", "peer-session", transport=transport
    )

    snapshot = await cli.refresh()
    run = snapshot.activities[0]
    assert run.child_session_id == "session-agent-1"
    assert run.last_response == "Initial work is complete."
    assert run.managed_snapshot().state is ManagedAgentState.IDLE
    assert cli.available_profiles() == ("default",)

    await cli.message("agent-1", "Continue from the CLI")
    peer_snapshot = await web_peer.refresh()
    peer_run = peer_snapshot.activities[0]
    assert peer_run.state == "working"
    assert peer_run.conversation[-1].content == "Continue from the CLI"

    await cli.cancel("agent-1")
    await cli.resolve_approval("agent-1", "approval-1", "approve_once")
    await cli.answer_question(
        "agent-1",
        "question-1",
        [{"question": "Proceed?", "answer": "Yes", "is_other": False}],
    )
    assert any(
        path.endswith("/approvals/approval-1")
        and payload == {"decision": "approve_once"}
        for _method, path, payload in api.requests
    )
    assert any(
        path.endswith("/questions/question-1")
        and payload["answers"][0]["answer"] == "Yes"
        for _method, path, payload in api.requests
    )

    await web_peer.stop("agent-1")
    await cli.refresh()
    assert cli.output("agent-1").state is ManagedAgentState.STOPPED

    created = await cli.start("default", "Review the integration", name="Reviewer")
    assert created.agent_id == "agent-2"
    assert len((await web_peer.refresh()).activities) == 2

    await cli.close()
    await web_peer.close()


@pytest.mark.asyncio
async def test_room_subscription_emits_staging_lifecycle_payload() -> None:
    api = _RoomAPI()
    client = AgentRoomClient(
        "http://127.0.0.1:4173", "cli-session", transport=httpx.MockTransport(api)
    )
    events = client.subscribe_events()

    event = await anext(events)

    assert event.task == "Build the feature"
    assert event.last_response == "Initial work is complete."
    assert event.error is None
    assert event.usage is not None
    assert event.usage.prompt_tokens == 120
    await events.aclose()
    await client.close()


def test_discovery_uses_the_backend_owner_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "agent-room" / "server.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"url": "http://127.0.0.1:4321", "pid": 123}), encoding="utf-8"
    )
    monkeypatch.setenv("VIBE_HOME", str(tmp_path))
    monkeypatch.delenv("VIBE_AGENT_ROOM_URL", raising=False)

    assert discover_agent_room() == "http://127.0.0.1:4321"


def test_backend_autostart_can_be_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VIBE_AGENT_ROOM_AUTOSTART", "0")

    assert launch_agent_room_backend(tmp_path) is False


def test_ensure_backend_reuses_healthy_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "vibe.core.agent_room.client.discover_agent_room",
        lambda: "http://127.0.0.1:4321",
    )
    monkeypatch.setattr(
        "vibe.core.agent_room.client._agent_room_reachable", lambda _url: True
    )

    assert ensure_agent_room_backend(tmp_path) == "http://127.0.0.1:4321"


def test_ensure_backend_starts_requested_port_and_waits_until_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reachable = iter((False, True))
    launched: list[tuple[Path, int, str, bool]] = []
    monkeypatch.setattr(
        "vibe.core.agent_room.client.discover_agent_room", lambda: None
    )
    monkeypatch.setattr(
        "vibe.core.agent_room.client._agent_room_reachable",
        lambda _url: next(reachable),
    )

    process = SimpleNamespace(poll=lambda: None, returncode=None)

    def launch(
        workdir: Path, *, port: int, network_mode: str, force: bool
    ) -> object:
        launched.append((workdir, port, network_mode, force))
        return process

    monkeypatch.setattr(
        "vibe.core.agent_room.client._spawn_agent_room_backend", launch
    )

    assert (
        ensure_agent_room_backend(tmp_path, port=4183, network_mode="direct")
        == "http://127.0.0.1:4183"
    )
    assert launched == [(tmp_path, 4183, "direct", True)]


def test_ensure_backend_retires_unresponsive_discovered_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovery_path = tmp_path / "agent-room" / "server.json"
    discovery_path.parent.mkdir(parents=True)
    discovery_path.write_text(
        json.dumps({"url": "http://127.0.0.1:4173", "pid": 4321}),
        encoding="utf-8",
    )
    monkeypatch.setenv("VIBE_HOME", str(tmp_path))
    monkeypatch.delenv("VIBE_AGENT_ROOM_URL", raising=False)
    reachable = iter((False, True))
    retired: list[int] = []
    process = SimpleNamespace(poll=lambda: None, returncode=None)
    monkeypatch.setattr(
        "vibe.core.agent_room.client._agent_room_reachable",
        lambda _url: next(reachable),
    )
    monkeypatch.setattr(
        "vibe.core.agent_room.client._stop_unresponsive_owner",
        lambda record: retired.append(record["pid"]) or True,
    )
    monkeypatch.setattr(
        "vibe.core.agent_room.client._spawn_agent_room_backend",
        lambda *_args, **_kwargs: process,
    )

    assert ensure_agent_room_backend(tmp_path) == "http://127.0.0.1:4173"
    assert retired == [4321]


def test_ensure_backend_reports_child_exit_without_waiting_for_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = SimpleNamespace(poll=lambda: 1, returncode=1)
    monkeypatch.setattr(
        "vibe.core.agent_room.client._agent_room_reachable", lambda _url: False
    )
    monkeypatch.setattr(
        "vibe.core.agent_room.client._spawn_agent_room_backend",
        lambda *_args, **_kwargs: process,
    )

    with pytest.raises(ValueError, match="exited during startup with status 1"):
        ensure_agent_room_backend(tmp_path)
