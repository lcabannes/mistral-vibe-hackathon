from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from http import HTTPStatus
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

import httpx

from vibe.core.agent_room.models import AgentRoomRun, AgentRoomSnapshot
from vibe.core.agents.events import ManagedAgentLifecycleEvent
from vibe.core.agents.models import ManagedAgentSnapshot
from vibe.core.types import LLMUsage

DEFAULT_AGENT_ROOM_URL = "http://127.0.0.1:4173"
AGENT_ROOM_DISCOVERY_FILE = "agent-room/server.json"
POLL_INTERVAL_SECONDS = 1.0
BACKEND_STARTUP_TIMEOUT_SECONDS = 45.0
BACKEND_SHUTDOWN_TIMEOUT_SECONDS = 5.0

type AgentRoomListener = Callable[[AgentRoomSnapshot], None]


class AgentRoomUnavailable(ValueError):
    pass


def _vibe_home() -> Path:
    return Path(os.environ.get("VIBE_HOME", "~/.vibe")).expanduser()


def _read_agent_room_discovery() -> dict[str, Any] | None:
    try:
        payload = json.loads(
            (_vibe_home() / AGENT_ROOM_DISCOVERY_FILE).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    url = payload.get("url") if isinstance(payload, dict) else None
    if not isinstance(url, str) or not url.startswith((
        "http://127.0.0.1:",
        "http://localhost:",
    )):
        return None
    return {**payload, "url": url.rstrip("/")}


def discover_agent_room() -> str | None:
    configured = os.environ.get("VIBE_AGENT_ROOM_URL")
    if configured:
        return configured.rstrip("/")
    discovery = _read_agent_room_discovery()
    return str(discovery["url"]) if discovery is not None else None


def _spawn_agent_room_backend(
    workdir: Path,
    *,
    port: int,
    network_mode: str,
    force: bool,
) -> subprocess.Popen[bytes] | None:
    if not force and os.environ.get("VIBE_AGENT_ROOM_AUTOSTART", "1") == "0":
        return None
    repository_root = Path(__file__).resolve().parents[3]
    server = repository_root / "web" / "agent-room" / "server.py"
    if not server.is_file() or not workdir.is_dir():
        return None
    try:
        git_repository = subprocess.run(
            ["git", "-C", str(workdir), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if git_repository.returncode != 0:
            return None
        return subprocess.Popen(
            [
                sys.executable,
                str(server),
                "--port",
                str(port),
                "--workdir",
                str(workdir),
                "--network-mode",
                network_mode,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def launch_agent_room_backend(
    workdir: Path,
    *,
    port: int = 4173,
    network_mode: str = "auto",
    force: bool = False,
) -> bool:
    return (
        _spawn_agent_room_backend(
            workdir,
            port=port,
            network_mode=network_mode,
            force=force,
        )
        is not None
    )


def _agent_room_reachable(url: str, *, timeout: float = 0.5) -> bool:
    try:
        with httpx.Client(base_url=url, timeout=timeout, trust_env=False) as client:
            response = client.get("/api/health")
            if response.status_code == HTTPStatus.NOT_FOUND:
                response = client.get("/api/agent-runs")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("connected") is True


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_agent_room_process(pid: int) -> bool:
    if os.name == "nt":
        return False
    try:
        process = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    command = process.stdout.replace("\\", "/")
    return process.returncode == 0 and "web/agent-room/server.py" in command


def _remove_discovery_for_pid(pid: int) -> None:
    discovery = _read_agent_room_discovery()
    if discovery is None or discovery.get("pid") != pid:
        return
    try:
        (_vibe_home() / AGENT_ROOM_DISCOVERY_FILE).unlink(missing_ok=True)
    except OSError:
        pass


def _wait_for_pid_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not _pid_exists(pid)


def _terminate_agent_room_process(pid: int) -> bool:
    if not _is_agent_room_process(pid):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    if _wait_for_pid_exit(pid, BACKEND_SHUTDOWN_TIMEOUT_SECONDS):
        return True
    if not _is_agent_room_process(pid):
        return False
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return False
    return _wait_for_pid_exit(pid, 1.0)


def _stop_unresponsive_owner(discovery: dict[str, Any]) -> bool:
    pid = discovery.get("pid")
    if not isinstance(pid, int) or pid <= 1 or pid == os.getpid():
        return False
    if _pid_exists(pid) and not _terminate_agent_room_process(pid):
        return False
    _remove_discovery_for_pid(pid)
    return True


def ensure_agent_room_backend(
    workdir: Path,
    *,
    port: int = 4173,
    network_mode: str = "auto",
    startup_timeout: float = BACKEND_STARTUP_TIMEOUT_SECONDS,
) -> str:
    """Return a healthy shared backend URL, starting one when necessary."""
    configured = os.environ.get("VIBE_AGENT_ROOM_URL")
    discovery = _read_agent_room_discovery()
    discovered = discover_agent_room()
    if discovered is not None and _agent_room_reachable(discovered):
        return discovered

    if (
        configured is None
        and discovery is not None
        and discovery.get("url") == discovered
        and not _stop_unresponsive_owner(discovery)
    ):
        raise AgentRoomUnavailable(
            "The discovered Agent Room is unresponsive and could not be stopped"
        )

    target = f"http://127.0.0.1:{port}"
    if target != discovered and _agent_room_reachable(target):
        return target
    process = _spawn_agent_room_backend(
        workdir, port=port, network_mode=network_mode, force=True
    )
    if process is None:
        raise AgentRoomUnavailable(
            "Agent Room server is unavailable from this Vibe installation"
        )

    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if _agent_room_reachable(target):
            return target
        if process.poll() is not None:
            raise AgentRoomUnavailable(
                f"Agent Room exited during startup with status {process.returncode}"
            )
        time.sleep(0.1)
    raise AgentRoomUnavailable(
        f"Agent Room did not become ready at {target} within "
        f"{startup_timeout:g} seconds"
    )


class AgentRoomClient:
    def __init__(
        self,
        base_url: str,
        parent_session_id: str,
        *,
        timeout: float = 10.0,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.parent_session_id = parent_session_id
        self.poll_interval = poll_interval
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            trust_env=False,
        )
        self._snapshot = AgentRoomSnapshot(connected=False)
        self._listeners: set[AgentRoomListener] = set()
        self._sequences: dict[str, int] = {}

    @classmethod
    def discovered(cls, parent_session_id: str) -> AgentRoomClient | None:
        url = discover_agent_room()
        return cls(url, parent_session_id) if url else None

    @property
    def snapshot(self) -> AgentRoomSnapshot:
        return self._snapshot

    def add_listener(self, listener: AgentRoomListener) -> None:
        self._listeners.add(listener)

    def remove_listener(self, listener: AgentRoomListener) -> None:
        self._listeners.discard(listener)

    async def close(self) -> None:
        await self._http.aclose()

    async def refresh(self) -> AgentRoomSnapshot:
        payload = await self._request("GET", "/api/agent-runs")
        try:
            snapshot = AgentRoomSnapshot.model_validate(payload)
        except ValueError as error:
            raise AgentRoomUnavailable(
                f"Invalid Agent Room snapshot: {error}"
            ) from error
        changed = (
            snapshot.connected != self._snapshot.connected
            or snapshot.api_version != self._snapshot.api_version
            or snapshot.instance_id != self._snapshot.instance_id
            or snapshot.revision != self._snapshot.revision
            or snapshot.activities != self._snapshot.activities
        )
        self._snapshot = snapshot
        if changed:
            for listener in tuple(self._listeners):
                listener(snapshot)
        return snapshot

    async def start(
        self, profile: str, task: str, *, name: str | None = None
    ) -> ManagedAgentSnapshot:
        run = AgentRoomRun.model_validate(
            await self._request(
                "POST",
                "/api/agent-runs",
                {
                    "agent_name": profile,
                    "display_name": name or profile,
                    "task": task,
                    "group_id": "unassigned",
                    "auto_approve": True,
                    "client_message_id": f"cli-create-{uuid4().hex}",
                },
            )
        )
        await self.refresh()
        return run.managed_snapshot()

    def list(self) -> tuple[ManagedAgentSnapshot, ...]:
        return tuple(
            run.managed_snapshot()
            for run in self._snapshot.activities
            if not run.is_orchestrator
        )

    def available_profiles(self) -> tuple[str, ...]:
        return tuple(
            profile.name
            for profile in self._snapshot.profiles
            if profile.name != "orchestrator"
        )

    async def message(self, agent_id: str, message: str) -> ManagedAgentSnapshot:
        payload = await self._request(
            "POST",
            f"/api/agent-runs/{agent_id}/messages",
            {"content": message, "client_message_id": f"cli-{uuid4().hex}"},
        )
        run = AgentRoomRun.model_validate(payload["run"])
        await self.refresh()
        return run.managed_snapshot()

    def output(self, agent_id: str) -> ManagedAgentSnapshot:
        run = self._run(agent_id)
        return run.managed_snapshot()

    async def stop(self, agent_id: str) -> ManagedAgentSnapshot:
        run = AgentRoomRun.model_validate(
            await self._request("POST", f"/api/agent-runs/{agent_id}/stop", {})
        )
        await self.refresh()
        return run.managed_snapshot()

    async def cancel(self, agent_id: str) -> AgentRoomRun:
        run = AgentRoomRun.model_validate(
            await self._request("POST", f"/api/agent-runs/{agent_id}/cancel", {})
        )
        await self.refresh()
        return run

    async def resolve_approval(
        self, agent_id: str, approval_id: str, decision: str
    ) -> None:
        await self._request(
            "POST",
            f"/api/agent-runs/{agent_id}/approvals/{approval_id}",
            {"decision": decision},
        )
        await self.refresh()

    async def answer_question(
        self, agent_id: str, question_id: str, answers: list[dict[str, Any]]
    ) -> None:
        await self._request(
            "POST",
            f"/api/agent-runs/{agent_id}/questions/{question_id}",
            {"answers": answers},
        )
        await self.refresh()

    async def subscribe_events(
        self,
    ) -> AsyncGenerator[ManagedAgentLifecycleEvent, None]:
        known: dict[str, tuple[str, float, int]] = {}
        while True:
            try:
                snapshot = await self.refresh()
            except AgentRoomUnavailable:
                if self._snapshot.connected:
                    self._snapshot = self._snapshot.model_copy(
                        update={"connected": False}
                    )
                    for listener in tuple(self._listeners):
                        listener(self._snapshot)
                await asyncio.sleep(self.poll_interval)
                continue
            for run in snapshot.activities:
                if run.is_orchestrator:
                    continue
                signature = (run.state, run.updated_at, run.queued_messages)
                if known.get(run.tool_call_id) == signature:
                    continue
                known[run.tool_call_id] = signature
                sequence = self._sequences.get(run.tool_call_id, 0) + 1
                self._sequences[run.tool_call_id] = sequence
                managed = run.managed_snapshot()
                yield ManagedAgentLifecycleEvent(
                    sequence=sequence,
                    agent_id=managed.agent_id,
                    profile=managed.profile,
                    agent_display_name=run.agent_display_name,
                    parent_session_id=self.parent_session_id,
                    child_session_id=managed.child_session_id,
                    task=managed.task,
                    state=managed.state,
                    current_activity=managed.current_activity,
                    queued_messages=managed.queued_messages,
                    error=managed.error,
                    last_response=managed.last_response,
                    usage=LLMUsage(
                        prompt_tokens=managed.prompt_tokens,
                        completion_tokens=managed.completion_tokens,
                    ),
                )
            await asyncio.sleep(self.poll_interval)

    def _run(self, agent_id: str) -> AgentRoomRun:
        try:
            return next(
                run for run in self._snapshot.activities if run.tool_call_id == agent_id
            )
        except StopIteration as error:
            raise ValueError(f"Unknown Agent Room run: {agent_id}") from error

    async def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        try:
            response = await self._http.request(method, path, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as error:
            try:
                detail = str(error.response.json().get("error") or "")
            except (AttributeError, ValueError):
                detail = ""
            raise AgentRoomUnavailable(detail or str(error)) from error
        except (httpx.HTTPError, ValueError) as error:
            raise AgentRoomUnavailable(str(error)) from error
