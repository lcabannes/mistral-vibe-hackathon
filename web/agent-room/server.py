from __future__ import annotations

import argparse
import base64
import binascii
from copy import deepcopy
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
from threading import Event, RLock, Thread
import time
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx

if os.name == "nt":
    import msvcrt
else:
    import fcntl

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vibe.core.agents.manager import AgentManager
from vibe.core.agents.models import BuiltinAgentName, ManagedAgentState
from vibe.core.config import VibeConfig, resolve_api_key
from vibe.core.config.harness_files import init_harness_files_manager
from vibe.core.config.orchestrator_legacy import LegacyConfigOrchestrator
from vibe.core.team_workspace import TeamWorkspaceSnapshot, discover_workspace_identity
from vibe.core.tools.manager import ToolManager
from vibe.core.utils.platform import is_windows
from vibe.core.worktree import PreparedWorktree, WorktreeError, prepare_worktree_session

WORKER_PATH = Path(__file__).with_name("worker.py")
DEFAULT_GROUP = "unassigned"
ORCHESTRATOR_ID = "orchestrator"
COATS = ("orange", "mint", "rose", "blue", "violet", "charcoal", "sunny")
TERMINAL_STATES = {"completed", "cancelled", "stopped"}
LIVE_STATES = {"requested", "running", "working", "attention", "idle", "failed"}
MIN_JSON_BODY_BYTES = 2
MAX_JSON_BODY_BYTES = 24_000_000
MAX_STORED_RUNS = 100
MAX_ACTIVE_AGENTS = 8
MAX_QUEUED_MESSAGES = 20
MAX_MESSAGE_CHARS = 8_000
MAX_CONVERSATION_ITEMS = 250
MAX_EVENTS = 180
MAX_TEAM_AGENT_LINKS = 100
TEAM_WORKSPACE_STALE_SECONDS = 45.0
MAX_ROOM_IMAGES = 4
MAX_ROOM_IMAGE_BYTES = 4 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/gif", "image/jpeg", "image/png", "image/webp"}
LOOPBACK_HOST = "127.0.0.1"
API_VERSION = 1
DISCOVERY_FILE = "agent-room/server.json"
OWNER_LOCK_FILE = "agent-room/server.lock"
ALLOWED_STATIC_PATHS = {
    "/web/agent-room/",
    "/web/agent-room/index.html",
    "/web/agent-room/styles.css",
    "/web/agent-room/app.js",
    "/web/agent-room/agents.json",
    "/distribution/zed/icons/mistral_vibe.svg",
}
RUN_ACTION_PARTS = 4
RUN_DETAIL_PARTS = 5
CONTROL_COMMAND_PARTS = 2
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
NETWORK_PROBE_URL = "https://api.mistral.ai/v1/models"
WORKTREE_STATUS_REFRESH_SECONDS = 5.0
WORKTREE_STATUS_TIMEOUT_SECONDS = 5.0


class AgentWorker:
    def __init__(
        self,
        store: AgentRoomStore,
        run_id: str,
        profile: str,
        worktree: PreparedWorktree,
        session_root: Path,
        worker_environment: dict[str, str],
        enabled_tools: tuple[str, ...] | None = None,
        resume_session_id: str | None = None,
        force_auto_approve: bool = False,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.profile = profile
        self.worktree = worktree
        self.session_root = session_root
        self.worker_environment = worker_environment
        self.enabled_tools = enabled_tools
        self.resume_session_id = resume_session_id
        self.force_auto_approve = force_auto_approve
        self.ready = Event()
        self.write_lock = RLock()
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        command = [
            sys.executable,
            str(WORKER_PATH),
            "--profile",
            self.profile,
            "--session-root",
            str(self.session_root),
        ]
        if self.enabled_tools is not None:
            command.append("--restrict-tools")
            for tool_name in self.enabled_tools:
                command.extend(("--enable-tool", tool_name))
        if self.resume_session_id:
            command.extend(("--resume-session", self.resume_session_id))
        if self.force_auto_approve:
            command.append("--auto-approve")
        creation_flags = int(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if is_windows() else 0
        )
        self.process = subprocess.Popen(
            command,
            cwd=self.worktree.path,
            env=self.worker_environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=not is_windows(),
            creationflags=creation_flags,
        )
        Thread(
            target=self._read_stdout, name=f"room-out-{self.run_id}", daemon=True
        ).start()
        Thread(
            target=self._read_stderr, name=f"room-err-{self.run_id}", daemon=True
        ).start()
        Thread(target=self._wait, name=f"room-wait-{self.run_id}", daemon=True).start()

    def send(self, payload: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.poll() is not None or process.stdin is None:
            raise RuntimeError("Agent worker is not running")
        line = json.dumps(payload, ensure_ascii=False)
        with self.write_lock:
            process.stdin.write(f"{line}\n")
            process.stdin.flush()

    def stop(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        with suppress_errors():
            self.send({"type": "shutdown"})
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        terminate_process_tree(process)

    def _read_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self.store.observe_worker_log(self.run_id, line.strip())
                continue
            if isinstance(payload, dict):
                self.store.observe_worker_event(self, payload)

    def _read_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            self.store.observe_worker_log(self.run_id, line.strip())

    def _wait(self) -> None:
        process = self.process
        if process is None:
            return
        return_code = process.wait()
        self.ready.set()
        self.store.observe_worker_exit(self, return_code)


class AgentRoomStore:
    def __init__(self, workdir: Path, *, network_mode: str = "auto") -> None:
        self._workdir = workdir
        self._lock = RLock()
        self._instance_id = uuid4().hex
        self._revision = 0
        self._lifecycle_locks: dict[str, RLock] = {}
        self._workers: dict[str, AgentWorker] = {}
        self._worktree_refreshing: set[str] = set()
        self._worktree_refreshed_at: dict[str, float] = {}
        vibe_home = Path(os.environ.get("VIBE_HOME", "~/.vibe")).expanduser()
        self._session_root = vibe_home / "logs" / "session"
        self._registry_path = vibe_home / "agent-room" / "runs.json"
        self._runs = self._load_registry()
        self._team_workspace: dict[str, Any] | None = None
        self._worker_environment, self._network = resolve_worker_network(network_mode)
        self._profiles = self._load_profiles()
        self._tools = self._load_tools()
        self._integration_branch = git_output(
            self._workdir, "branch", "--show-current"
        ).strip()
        self._mark_interrupted_runs()
        self._launch_orchestrator()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            for run_id, run in self._runs.items():
                self._schedule_worktree_status_refresh_locked(run_id, run)
            team_workspace = deepcopy(getattr(self, "_team_workspace", None))
            if (
                team_workspace is not None
                and time.time() - float(team_workspace.get("published_at") or 0)
                > TEAM_WORKSPACE_STALE_SECONDS
            ):
                team_workspace["snapshot"]["connection_state"] = "degraded"
            return {
                "api_version": API_VERSION,
                "instance_id": self._instance_id,
                "revision": self._revision,
                "connected": True,
                "workspace": {
                    "workdir": str(self._workdir),
                    "integration_branch": self._integration_branch,
                },
                "activities": [self._public_run(run) for run in self._runs.values()],
                "profiles": deepcopy(self._profiles),
                "tools": deepcopy(self._tools),
                "coordination": self._coordination_locked(),
                "network": deepcopy(self._network),
                "team_workspace": team_workspace,
            }

    def update_team_workspace(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_snapshot = payload.get("snapshot")
        if not isinstance(raw_snapshot, dict):
            raise ValueError("snapshot must be an object")
        try:
            snapshot = TeamWorkspaceSnapshot.model_validate_json(
                json.dumps(raw_snapshot, ensure_ascii=False)
            )
        except ValueError as error:
            raise ValueError(f"Invalid team workspace snapshot: {error}") from error

        expected = discover_workspace_identity(self._workdir)
        if (
            snapshot.identity.workspace_id != expected.workspace_id
            or snapshot.identity.project_fingerprint != expected.project_fingerprint
        ):
            raise ValueError("Team workspace snapshot belongs to another project")

        local_member_id = self.required_text(payload, "local_member_id", 80)
        member_ids = {member.member_id for member in snapshot.members}
        if not re.fullmatch(r"member_[a-f0-9]{32}", local_member_id) or (
            member_ids and local_member_id not in member_ids
        ):
            raise ValueError("local_member_id is not present in the team snapshot")
        raw_links = payload.get("local_agent_links", {})
        if not isinstance(raw_links, dict) or len(raw_links) > MAX_TEAM_AGENT_LINKS:
            raise ValueError("local_agent_links must be a bounded object")
        team_run_ids = {run.run_id for run in snapshot.runs}
        with self._lock:
            links: dict[str, str] = {}
            for team_run_id, local_run_id in raw_links.items():
                if (
                    not isinstance(team_run_id, str)
                    or not isinstance(local_run_id, str)
                    or team_run_id not in team_run_ids
                    or local_run_id not in self._runs
                ):
                    raise ValueError("local_agent_links contains an unknown run")
                links[team_run_id] = local_run_id
            self._team_workspace = {
                "snapshot": snapshot.model_dump(mode="json"),
                "local_member_id": local_member_id,
                "local_agent_links": links,
                "published_at": time.time(),
            }
            self._revision += 1
            return deepcopy(self._team_workspace)

    def health(self) -> dict[str, Any]:
        return {
            "api_version": API_VERSION,
            "instance_id": self._instance_id,
            "connected": True,
        }

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile = (
            self._optional_text(payload, "agent_name", 80) or BuiltinAgentName.DEFAULT
        )
        if profile == BuiltinAgentName.ORCHESTRATOR:
            raise ValueError("The room already has one orchestrator")
        if profile not in {item["name"] for item in self._profiles}:
            raise ValueError(f"Unknown agent profile: {profile}")
        task = self.required_text(payload, "task", MAX_MESSAGE_CHARS)
        display_name = self._optional_text(payload, "display_name", 50) or profile
        group_id = self._optional_text(payload, "group_id", 80) or DEFAULT_GROUP
        enabled_tools = self._enabled_tools(payload)
        auto_approve = self._optional_bool(payload, "auto_approve", default=True)
        client_message_id = (
            self._optional_text(payload, "client_message_id", 100)
            or f"create-{uuid4().hex}"
        )
        run = self._launch_worker(
            profile=profile,
            display_name=display_name,
            group_id=group_id,
            task=task,
            is_orchestrator=False,
            client_message_id=client_message_id,
            enabled_tools=enabled_tools,
            auto_approve=auto_approve,
        )
        return self._public_run(run)

    def send_message(
        self, run_id: str, payload: dict[str, Any], *, interpret_commands: bool = True
    ) -> dict[str, Any]:
        content = self.required_text(payload, "content", MAX_MESSAGE_CHARS)
        client_message_id = (
            self._optional_text(payload, "client_message_id", 100)
            or f"web-{uuid4().hex}"
        )
        images = self._validated_images(payload)
        if interpret_commands and content.startswith("//"):
            content = content[1:]
        elif interpret_commands and content.startswith("/"):
            return self._chat_command(run_id, content, client_message_id)
        with self._lifecycle_lock(run_id):
            return self._send_prompt_message(run_id, content, client_message_id, images)

    def _send_prompt_message(
        self,
        run_id: str,
        content: str,
        client_message_id: str,
        images: list[dict[str, Any]],
    ) -> dict[str, Any]:
        with self._lock:
            run = self._get_run_locked(run_id)
            existing = self._find_client_message(run, client_message_id)
            if existing is not None:
                return {"message": deepcopy(existing), "run": self._public_run(run)}
        worker = self._ensure_worker(run_id)
        with self._lock:
            run = self._get_run_locked(run_id)
            existing = self._find_client_message(run, client_message_id)
            if existing is not None:
                return {"message": deepcopy(existing), "run": self._public_run(run)}
            queued = sum(
                item.get("status") in {"queued", "running"}
                for item in run.get("conversation", [])
                if item.get("role") == "user"
            )
            if queued >= MAX_QUEUED_MESSAGES:
                raise ValueError(
                    f"This agent already has {MAX_QUEUED_MESSAGES} queued messages"
                )
            message = self._message("user", content, client_message_id, "queued")
            message["attachments"] = [
                {"alias": image["alias"], "mime_type": image["mime_type"]}
                for image in images
            ]
            run["conversation"].append(message)
            run["queued_messages"] = int(run.get("queued_messages") or 0) + 1
            run["updated_at"] = time.time()
            self._trim_conversation(run)
            try:
                worker.send({
                    "type": "prompt",
                    "message_id": message["id"],
                    "content": content,
                    "images": images,
                })
            except Exception as error:
                message["status"] = "failed"
                message["error_code"] = "worker_offline"
                run["error"] = safe_error(error)
                self._persist_locked()
                raise
            self._persist_locked()
        self._broadcast_management_snapshot()
        return {"message": deepcopy(message), "run": self._public_run(run)}

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._get_run_locked(run_id)
            worker = self._workers.get(run_id)
            if worker is None:
                return self._public_run(run)
            worker.send({"type": "cancel"})
            run["current_activity"] = "Cancellation requested"
            self._append_event(run, "cancel", "Response cancellation requested")
            self._persist_locked()
            return self._public_run(run)

    def stop(self, run_id: str) -> dict[str, Any]:
        with self._lifecycle_lock(run_id):
            return self._stop_worker(run_id)

    def _stop_worker(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._get_run_locked(run_id)
            worker = self._workers.pop(run_id, None)
            run["runtime_live"] = False
            run["resumable"] = True
            run["state"] = "stopped"
            run["current_activity"] = "Agent stopped"
            for message in run.get("conversation", []):
                if message.get("status") in {"queued", "running"}:
                    message["status"] = "cancelled"
            self._expire_interactions(run)
            self._append_event(run, "stopped", "Agent stopped")
            self._persist_locked()
        if worker is not None:
            worker.stop()
        self._broadcast_management_snapshot()
        return self._public_run(run)

    def update_group(self, run_id: str, group_id: str) -> dict[str, Any]:
        normalized = group_id.strip()[:80]
        if not normalized:
            raise ValueError("group_id is required")
        with self._lock:
            run = self._get_run_locked(run_id)
            if run.get("is_orchestrator"):
                raise ValueError("The orchestrator stays in the Coordination group")
            run["group_id"] = normalized
            run["updated_at"] = time.time()
            self._persist_locked()
            return self._public_run(run)

    def resolve_approval(
        self, run_id: str, approval_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        decision = self.required_text(payload, "decision", 30)
        if decision not in {"approve_once", "deny"}:
            raise ValueError("decision must be approve_once or deny")
        feedback = self._optional_text(payload, "feedback", 1_000)
        with self._lock:
            run = self._get_run_locked(run_id)
            approval = next(
                (
                    item
                    for item in run.get("approvals", [])
                    if item["id"] == approval_id
                ),
                None,
            )
            worker = self._workers.get(run_id)
            if approval is None or worker is None:
                raise KeyError(approval_id)
            if approval.get("status") != "pending":
                raise RuntimeError("This approval has already been resolved")
            approval["status"] = "approved" if decision == "approve_once" else "denied"
            approval["resolved_at"] = time.time()
            approval["feedback"] = feedback
            worker.send({
                "type": "approval_response",
                "request_id": approval_id,
                "decision": decision,
                "feedback": feedback,
            })
            self._append_event(
                run, "approval_resolved", f"Approval {approval['status']}"
            )
            self._persist_locked()
            return deepcopy(approval)

    def answer_question(
        self, run_id: str, question_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        answers = payload.get("answers")
        if not isinstance(answers, list) or not answers:
            raise ValueError("answers is required")
        with self._lock:
            run = self._get_run_locked(run_id)
            question = next(
                (
                    item
                    for item in run.get("questions", [])
                    if item["id"] == question_id
                ),
                None,
            )
            worker = self._workers.get(run_id)
            if question is None or worker is None:
                raise KeyError(question_id)
            if question.get("status") != "pending":
                raise RuntimeError("This question has already been answered")
            question["status"] = "answered"
            question["answers"] = answers
            question["answered_at"] = time.time()
            worker.send({
                "type": "question_response",
                "request_id": question_id,
                "answers": answers,
            })
            self._append_event(run, "question_answered", "Question answered")
            self._persist_locked()
            return deepcopy(question)

    def merge(self, run_id: str) -> dict[str, Any]:
        with self._lifecycle_lock(run_id):
            return self._merge_stopped_worker(run_id)

    def _merge_stopped_worker(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._get_run_locked(run_id)
            if run.get("is_orchestrator"):
                raise ValueError(
                    "The orchestrator worktree cannot be merged from the room"
                )
            if run.get("runtime_live"):
                raise ValueError(
                    "Stop the agent before validating and merging its work"
                )
            worktree_path = Path(str(run.get("worktree_path") or ""))
            branch = str(run.get("branch") or "")
            base_commit = str(run.get("base_commit") or "")
            if not worktree_path.is_dir() or not branch or not base_commit:
                raise ValueError("Worktree metadata is unavailable")
            run["merge_status"] = "validating"
            run["merge_error"] = None
            self._append_event(run, "merge", "Validating merge candidate")
            self._persist_locked()
        try:
            result = self._validate_and_merge(worktree_path, branch, base_commit)
        except Exception as error:
            with self._lock:
                run["merge_status"] = "failed"
                run["merge_error"] = safe_error(error)
                self._append_event(
                    run, "merge_failed", "Merge validation failed", run["merge_error"]
                )
                self._persist_locked()
            raise ValueError(safe_error(error)) from error
        with self._lock:
            run["merge_status"] = "merged"
            run["merge_commit"] = result["merge_commit"]
            run["validation_summary"] = result["validation_summary"]
            self._append_event(
                run, "merged", "Validated work merged", result["merge_commit"]
            )
            self._persist_locked()
            return self._public_run(run)

    def observe_worker_event(  # noqa: PLR0912, PLR0915
        self, worker: AgentWorker, payload: dict[str, Any]
    ) -> None:
        event_type = payload.get("type")
        if event_type == "remote_request":
            with self._lock:
                if self._workers.get(worker.run_id) is not worker:
                    return
            Thread(
                target=self._handle_remote_request,
                args=(worker, payload),
                name=f"room-remote-{worker.run_id}",
                daemon=True,
            ).start()
            return
        with self._lock:
            if self._workers.get(worker.run_id) is not worker:
                return
            run = self._runs.get(worker.run_id)
            if run is None:
                return
            now = time.time()
            if event_type == "ready":
                run.update({
                    "child_session_id": payload.get("session_id"),
                    "parent_session_id": payload.get("parent_session_id"),
                    "model": payload.get("model"),
                    "context_limit": payload.get("context_limit"),
                    "state": "idle",
                    "current_activity": None,
                    "runtime_live": True,
                    "resumable": True,
                    "updated_at": now,
                })
                resume_requested = bool(payload.get("resume_requested"))
                resumed = bool(payload.get("resumed"))
                if resumed:
                    self._append_event(run, "resumed", "Conversation resumed")
                else:
                    self._append_event(
                        run,
                        "ready",
                        "Worker connected",
                        str(payload.get("resume_error") or "")
                        if resume_requested
                        else None,
                    )
                worker.ready.set()
            elif event_type == "state":
                self._observe_state_locked(run, payload)
            elif event_type == "assistant_delta":
                run["current_activity"] = "Responding"
                run["updated_at"] = now
            elif event_type == "assistant_final":
                content = str(payload.get("content") or "").strip()
                if content:
                    run["conversation"].append(
                        self._message("assistant", content, None, "succeeded")
                    )
                    self._trim_conversation(run)
            elif event_type == "tool_started":
                tool_name = str(payload.get("tool_name") or "tool")
                run["state"] = "working"
                run["current_activity"] = f"Running {tool_name}"
                self._append_event(run, "tool", f"Running {tool_name}")
            elif event_type == "tool_finished":
                error = payload.get("error")
                self._append_event(
                    run,
                    "tool_result",
                    "Tool failed" if error else "Tool completed",
                    str(error) if error else None,
                )
            elif event_type == "approval_requested":
                approval = {
                    "id": payload.get("request_id"),
                    "tool_call_id": payload.get("tool_call_id"),
                    "tool_name": payload.get("tool_name"),
                    "arguments": payload.get("arguments") or {},
                    "permissions": payload.get("permissions") or [],
                    "status": "pending",
                    "created_at": now,
                }
                run["approvals"].append(approval)
                run["state"] = "attention"
                run["current_activity"] = f"Approval needed for {approval['tool_name']}"
                self._append_event(
                    run, "approval", f"Approval needed: {approval['tool_name']}"
                )
            elif event_type == "question_requested":
                question = {
                    "id": payload.get("request_id"),
                    "questions": payload.get("questions") or [],
                    "footer_note": payload.get("footer_note"),
                    "status": "pending",
                    "created_at": now,
                }
                run["questions"].append(question)
                run["state"] = "attention"
                run["current_activity"] = "Waiting for your answer"
                self._append_event(run, "question", "Agent asked a question")
            elif event_type == "usage":
                run.update({
                    "turns_used": payload.get("turns_used", 0),
                    "usage": {
                        "prompt_tokens": payload.get("prompt_tokens", 0),
                        "completion_tokens": payload.get("completion_tokens", 0),
                    },
                    "context_tokens": payload.get("context_tokens", 0),
                    "context_limit": payload.get("context_limit"),
                    "estimated_cost_usd": payload.get("estimated_cost_usd", 0.0),
                    "model": payload.get("model"),
                    "child_session_id": payload.get("session_id")
                    or run.get("child_session_id"),
                    "parent_session_id": payload.get("parent_session_id"),
                })
            elif event_type == "prompt_failed":
                message = self._message_by_id(run, payload.get("message_id"))
                if message is not None:
                    message["status"] = "failed"
                    message["error_code"] = "queue_full"
                run["error"] = str(payload.get("error") or "Prompt failed")
            run["updated_at"] = now
            self._schedule_worktree_status_refresh_locked(worker.run_id, run)
            self._persist_locked()
        if event_type in {"ready", "state", "usage"}:
            self._broadcast_management_snapshot()

    def observe_worker_log(self, run_id: str, line: str) -> None:
        if not line:
            return
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            logs = run.setdefault("worker_logs", [])
            logs.append(line[:1_000])
            run["worker_logs"] = logs[-30:]

    def observe_worker_exit(self, worker: AgentWorker, return_code: int) -> None:
        run_id = worker.run_id
        with self._lock:
            run = self._runs.get(run_id)
            if (
                run is None
                or not run.get("runtime_live")
                or self._workers.get(run_id) is not worker
            ):
                return
            self._workers.pop(run_id, None)
            run["runtime_live"] = False
            run["resumable"] = True
            run["state"] = "failed" if return_code else "stopped"
            run["current_activity"] = (
                f"Worker exited with status {return_code}"
                if return_code
                else "Worker stopped"
            )
            run["error"] = run["current_activity"] if return_code else None
            for message in run.get("conversation", []):
                if message.get("status") in {"queued", "running"}:
                    message["status"] = "failed" if return_code else "cancelled"
            self._expire_interactions(run)
            self._append_event(run, "worker_exit", run["current_activity"])
            self._persist_locked()
        self._broadcast_management_snapshot()

    def close(self) -> None:
        with self._lock:
            workers = tuple(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop()

    def _launch_orchestrator(self) -> None:
        with self._lock:
            existing = self._runs.get(ORCHESTRATOR_ID)
            if existing is not None:
                existing["auto_approve"] = True
                existing["enabled_tools"] = None
                existing["tool_policy"] = "all"
        if existing is not None:
            self._ensure_worker(ORCHESTRATOR_ID)
        else:
            self._launch_worker(
                profile=BuiltinAgentName.ORCHESTRATOR,
                display_name="Orchestrator",
                group_id="coordination",
                task="Coordinate and control the room's isolated agents",
                is_orchestrator=True,
                client_message_id=None,
                fixed_run_id=ORCHESTRATOR_ID,
            )
        with self._lock:
            run = self._runs[ORCHESTRATOR_ID]
            self._append_event(
                run,
                "network",
                f"Mistral {self._network['selected_mode']} connection ready",
                "Authenticated" if self._network["authenticated"] else "Unavailable",
            )
            self._persist_locked()

    def _launch_worker(
        self,
        *,
        profile: str,
        display_name: str,
        group_id: str,
        task: str,
        is_orchestrator: bool,
        client_message_id: str | None,
        fixed_run_id: str | None = None,
        enabled_tools: list[str] | None = None,
        auto_approve: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            active_count = sum(
                bool(run.get("runtime_live")) for run in self._runs.values()
            )
            if active_count >= MAX_ACTIVE_AGENTS:
                raise ValueError(f"At most {MAX_ACTIVE_AGENTS} agents can be active")
        run_id = fixed_run_id or f"agent-{uuid4().hex}"
        worktree_name = self._worktree_name(display_name)
        try:
            prepared = prepare_worktree_session(worktree_name, self._workdir)
        except WorktreeError as error:
            raise ValueError(str(error)) from error
        now = time.time()
        conversation = []
        if client_message_id is not None:
            conversation.append(
                self._message("user", task, client_message_id, "queued")
            )
        run = {
            "tool_call_id": run_id,
            "parent_session_id": None,
            "child_session_id": None,
            "agent_name": profile,
            "agent_display_name": display_name,
            "task": task,
            "state": "requested",
            "started_at": now,
            "updated_at": now,
            "current_activity": "Preparing isolated worktree",
            "turns_used": 0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "context_tokens": 0,
            "context_limit": None,
            "estimated_cost_usd": 0.0,
            "model": None,
            "is_primary": is_orchestrator,
            "is_orchestrator": is_orchestrator,
            "group_id": group_id,
            "coat": "charcoal"
            if is_orchestrator
            else COATS[len(self._runs) % len(COATS)],
            "source": "live",
            "runtime_live": True,
            "events": [],
            "conversation": conversation,
            "approvals": [],
            "questions": [],
            "error": None,
            "resumable": True,
            "queued_messages": len(conversation),
            "enabled_tools": enabled_tools,
            "tool_policy": "all" if enabled_tools is None else "selected",
            "auto_approve": auto_approve or is_orchestrator,
            "worktree_name": prepared.name,
            "worktree_path": str(prepared.path),
            "worktree_root": str(prepared.root),
            "branch": prepared.branch,
            "base_commit": prepared.base_commit,
            "worktree_dirty": False,
            "uncommitted_files": 0,
            "new_commit_count": 0,
            "merge_status": "not_ready",
            "merge_error": None,
        }
        worker = AgentWorker(
            self,
            run_id,
            profile,
            prepared,
            self._session_root,
            self._worker_environment,
            enabled_tools=tuple(enabled_tools) if enabled_tools is not None else None,
            force_auto_approve=auto_approve or is_orchestrator,
        )
        with self._lock:
            if fixed_run_id is not None:
                old = self._runs.pop(fixed_run_id, None)
                if old:
                    run["events"] = old.get("events", [])
                    run["conversation"] = old.get("conversation", [])
            self._runs[run_id] = run
            self._workers[run_id] = worker
            self._append_event(
                run, "worktree", "Isolated worktree ready", str(prepared.path)
            )
            self._prune_runs_locked()
            self._persist_locked()
        try:
            worker.start()
            if not worker.ready.wait(timeout=25):
                raise RuntimeError("Agent worker did not become ready")
            if worker.process is None or worker.process.poll() is not None:
                raise RuntimeError(run.get("error") or "Agent worker failed to start")
            if client_message_id is not None:
                message = run["conversation"][-1]
                worker.send({
                    "type": "prompt",
                    "message_id": message["id"],
                    "content": task,
                })
        except Exception:
            worker.stop()
            raise
        self._broadcast_management_snapshot()
        return run

    def _ensure_worker(self, run_id: str) -> AgentWorker:  # noqa: PLR0915
        leader = False
        with self._lock:
            run = self._get_run_locked(run_id)
            worker = self._workers.get(run_id)
            if worker is not None and run.get("runtime_live") and worker.ready.is_set():
                return worker
            if worker is None or not run.get("runtime_live"):
                active_count = sum(
                    bool(item.get("runtime_live")) for item in self._runs.values()
                )
                if active_count >= MAX_ACTIVE_AGENTS:
                    raise ValueError(
                        f"At most {MAX_ACTIVE_AGENTS} agents can be active"
                    )
                worktree_name = str(run.get("worktree_name") or "")
                if not worktree_name or run.get("merge_status") == "merged":
                    worktree_name = self._worktree_name(run["agent_display_name"])
                try:
                    prepared = prepare_worktree_session(worktree_name, self._workdir)
                except WorktreeError as error:
                    raise ValueError(str(error)) from error
                if prepared.name != run.get("worktree_name"):
                    run.update({
                        "worktree_name": prepared.name,
                        "worktree_path": str(prepared.path),
                        "worktree_root": str(prepared.root),
                        "branch": prepared.branch,
                        "base_commit": prepared.base_commit,
                        "merge_status": "not_ready",
                        "merge_error": None,
                    })
                enabled_tools = run.get("enabled_tools")
                if "tool_policy" not in run:
                    available = {item["name"] for item in self._tools}
                    if (
                        isinstance(enabled_tools, list)
                        and set(enabled_tools) == available
                    ):
                        enabled_tools = None
                    run["enabled_tools"] = enabled_tools
                    run["tool_policy"] = "all" if enabled_tools is None else "selected"
                if enabled_tools is not None and not isinstance(enabled_tools, list):
                    enabled_tools = None
                    run["enabled_tools"] = None
                worker = AgentWorker(
                    self,
                    run_id,
                    run["agent_name"],
                    prepared,
                    self._session_root,
                    self._worker_environment,
                    enabled_tools=(
                        tuple(enabled_tools) if enabled_tools is not None else None
                    ),
                    resume_session_id=run.get("child_session_id"),
                    force_auto_approve=bool(
                        run.get("auto_approve") or run.get("is_orchestrator")
                    ),
                )
                self._workers[run_id] = worker
                run.update({
                    "runtime_live": True,
                    "resumable": True,
                    "state": "requested",
                    "current_activity": "Restarting in isolated worktree",
                    "error": None,
                    "updated_at": time.time(),
                })
                self._append_event(run, "restart", "Restarting retained conversation")
                self._persist_locked()
                leader = True
        try:
            if leader:
                worker.start()
            if not worker.ready.wait(timeout=25):
                raise RuntimeError("Agent worker did not become ready")
            if worker.process is None or worker.process.poll() is not None:
                raise RuntimeError("Agent worker failed to restart")
            return worker
        except Exception as error:
            worker.stop()
            with self._lock:
                if self._workers.get(run_id) is worker:
                    self._workers.pop(run_id, None)
                run = self._get_run_locked(run_id)
                run.update({
                    "runtime_live": False,
                    "resumable": True,
                    "state": "failed",
                    "current_activity": "Restart failed",
                    "error": safe_error(error),
                    "updated_at": time.time(),
                })
                self._append_event(
                    run, "restart_failed", "Restart failed", run["error"]
                )
                self._persist_locked()
            raise ValueError(safe_error(error)) from error

    def _observe_state_locked(
        self, run: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        state = str(payload.get("state") or "idle")
        message = self._message_by_id(run, payload.get("message_id"))
        if state == "running" and message is not None:
            message["status"] = "running"
            message["updated_at"] = time.time()
        elif state == "idle" and message is not None:
            message["status"] = "cancelled" if payload.get("cancelled") else "succeeded"
            message["updated_at"] = time.time()
        elif state == "failed" and message is not None:
            message["status"] = "failed"
            message["error_code"] = "agent_error"
            message["updated_at"] = time.time()
        run["state"] = state
        run["current_activity"] = payload.get("activity")
        run["queued_messages"] = int(payload.get("queued_messages") or 0)
        run["error"] = payload.get("error")
        label = run["current_activity"] or state.title()
        self._append_event(run, state, label, run["error"])

    def _handle_remote_request(
        self, worker: AgentWorker, payload: dict[str, Any]
    ) -> None:
        with self._lifecycle_lock(worker.run_id):
            with self._lock:
                if self._workers.get(worker.run_id) is not worker:
                    return
            self._handle_current_remote_request(worker, payload)

    def _handle_current_remote_request(
        self, worker: AgentWorker, payload: dict[str, Any]
    ) -> None:
        request_id = str(payload.get("request_id") or "")
        operation = str(payload.get("operation") or "")
        args = payload.get("payload")
        if not isinstance(args, dict):
            args = {}
        try:
            if worker.run_id != ORCHESTRATOR_ID:
                raise ValueError("Only the orchestrator can control other agents")
            if operation == "start":
                created = self.create({
                    "agent_name": args.get("profile"),
                    "task": args.get("task"),
                    "display_name": args.get("name") or args.get("profile"),
                    "group_id": DEFAULT_GROUP,
                    "client_message_id": f"orchestrator-{uuid4().hex}",
                })
                result = self._managed_snapshot(created)
            elif operation == "message":
                response = self.send_message(
                    str(args.get("agent_id") or ""),
                    {
                        "content": args.get("message"),
                        "client_message_id": f"orchestrator-{uuid4().hex}",
                    },
                    interpret_commands=False,
                )
                result = self._managed_snapshot(response["run"])
            elif operation == "stop":
                stopped = self.stop(str(args.get("agent_id") or ""))
                result = self._managed_snapshot(stopped)
            elif operation == "control":
                self._record_control_request(args.get("request"))
                result = {"accepted": True}
            else:
                raise ValueError(f"Unsupported remote operation: {operation}")
            worker.send({
                "type": "remote_response",
                "request_id": request_id,
                "ok": True,
                "result": result,
            })
        except Exception as error:
            with suppress_errors():
                worker.send({
                    "type": "remote_response",
                    "request_id": request_id,
                    "ok": False,
                    "error": safe_error(error),
                })

    def _broadcast_management_snapshot(self) -> None:
        with self._lock:
            orchestrator = self._workers.get(ORCHESTRATOR_ID)
            if orchestrator is None or not orchestrator.ready.is_set():
                return
            agents = [
                self._managed_snapshot(run)
                for run_id, run in self._runs.items()
                if run_id != ORCHESTRATOR_ID
            ]
            profiles = [item["name"] for item in self._profiles]
        with suppress_errors():
            orchestrator.send({
                "type": "management_snapshot",
                "agents": agents,
                "profiles": profiles,
            })

    def _record_control_request(self, request: Any) -> None:
        if not isinstance(request, dict):
            raise ValueError("Invalid control request")
        with self._lock:
            run = self._runs[ORCHESTRATOR_ID]
            run["ui_directive"] = {**request, "id": uuid4().hex, "at": time.time()}
            self._append_event(
                run,
                "control",
                f"Room control: {request.get('action', 'command')}",
                json.dumps(request, ensure_ascii=False),
            )
            self._persist_locked()
        if request.get("action") != "command":
            return
        command = request.get("command")
        if not isinstance(command, str):
            raise ValueError("Invalid room command")
        try:
            parts = shlex.split(command)
        except ValueError as error:
            raise ValueError("Invalid room command") from error
        if len(parts) != CONTROL_COMMAND_PARTS:
            raise ValueError("Room commands require exactly one agent id")
        action, agent_id = parts
        if action == "/cancel":
            self.cancel(agent_id)
        elif action == "/stop":
            self.stop(agent_id)
        elif action == "/merge":
            self.merge(agent_id)
        else:
            raise ValueError(
                "Supported Agent Room commands are /cancel, /stop, and /merge"
            )

    def _chat_command(
        self, run_id: str, text: str, client_message_id: str
    ) -> dict[str, Any]:
        command = text.partition(" ")[0].lower()
        allowed = {
            "/help",
            "/status",
            "/history",
            "/queue",
            "/cancel",
            "/retry",
            "/stop",
        }
        if command not in allowed:
            raise ValueError(
                f"Unknown command: {command}. Use /help for available commands"
            )
        if command == "/cancel":
            return {"command": command, "run": self.cancel(run_id)}
        if command == "/stop":
            return {"command": command, "run": self.stop(run_id)}
        if command == "/retry":
            with self._lock:
                run = self._get_run_locked(run_id)
                failed = next(
                    (
                        item
                        for item in reversed(run.get("conversation", []))
                        if item.get("role") == "user"
                        and item.get("status") in {"failed", "cancelled"}
                    ),
                    None,
                )
            if failed is None:
                raise ValueError("There is no failed or cancelled message to retry")
            return self.send_message(
                run_id,
                {"content": failed["content"], "client_message_id": client_message_id},
            )
        with self._lock:
            run = self._get_run_locked(run_id)
            if command == "/help":
                response = "Commands: /status, /history, /queue, /cancel, /stop, /retry, /help. Use // for a literal slash."
            elif command == "/status":
                usage = run.get("usage") or {}
                total = int(usage.get("prompt_tokens") or 0) + int(
                    usage.get("completion_tokens") or 0
                )
                response = f"{run['state']} · {run.get('current_activity') or 'ready'} · {total:,} tokens · {run.get('queued_messages', 0)} queued"
            elif command == "/queue":
                response = f"{run.get('queued_messages', 0)} message(s) queued"
            else:
                response = f"{len(run.get('events', []))} lifecycle event(s); open History for details"
            user_message = self._message("user", text, client_message_id, "succeeded")
            system_message = self._message("system", response, None, "succeeded")
            run["conversation"].extend([user_message, system_message])
            self._trim_conversation(run)
            self._persist_locked()
            return {
                "command": command,
                "message": system_message,
                "run": self._public_run(run),
            }

    def _validate_and_merge(
        self, worktree_path: Path, branch: str, base_commit: str
    ) -> dict[str, str]:
        dirty = git_output(
            worktree_path, "status", "--porcelain", "--untracked-files=all"
        )
        if dirty.strip():
            raise ValueError(
                "Agent worktree has uncommitted files; ask the agent to commit first"
            )
        commit_count = int(
            git_output(
                worktree_path, "rev-list", "--count", f"{base_commit}..{branch}"
            ).strip()
        )
        if commit_count < 1:
            raise ValueError("Agent branch has no commits to merge")
        integration_status = git_output(
            self._workdir, "status", "--porcelain", "--untracked-files=all"
        )
        if integration_status.strip():
            raise ValueError("Integration worktree must be clean before merge")
        integration_head = git_output(self._workdir, "rev-parse", "HEAD").strip()
        validation_name = f"room-validate-{uuid4().hex[:10]}"
        validation_root = self._workdir.parent / validation_name
        subprocess.run(
            [
                "git",
                "-C",
                str(self._workdir),
                "worktree",
                "add",
                "--detach",
                str(validation_root),
                integration_head,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            merge = subprocess.run(
                [
                    "git",
                    "-C",
                    str(validation_root),
                    "merge",
                    "--no-commit",
                    "--no-ff",
                    branch,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if merge.returncode != 0:
                raise ValueError(
                    f"Merge conflict in validation worktree: {(merge.stderr or merge.stdout)[-1200:]}"
                )
            uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
            validation = subprocess.run(
                [uv, "run", "pytest", "-q"],
                cwd=validation_root,
                capture_output=True,
                text=True,
                timeout=15 * 60,
            )
            summary = (validation.stdout + validation.stderr)[-2_000:]
            if validation.returncode != 0:
                raise ValueError(f"Validation tests failed: {summary}")
        finally:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self._workdir),
                    "worktree",
                    "remove",
                    "--force",
                    str(validation_root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        if git_output(self._workdir, "rev-parse", "HEAD").strip() != integration_head:
            raise ValueError(
                "Integration branch changed during validation; retry the merge"
            )
        subprocess.run(
            [
                "git",
                "-C",
                str(self._workdir),
                "merge",
                "--no-ff",
                branch,
                "-m",
                f"Merge Agent Room branch {branch}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        subprocess.run(
            ["git", "-C", str(self._workdir), "diff", "--check", "HEAD^", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            "merge_commit": git_output(self._workdir, "rev-parse", "HEAD").strip(),
            "validation_summary": summary,
        }

    def _schedule_worktree_status_refresh_locked(
        self, run_id: str, run: dict[str, Any]
    ) -> None:
        path_value = run.get("worktree_path")
        base_commit = run.get("base_commit")
        branch = run.get("branch")
        if not path_value or not base_commit or not branch:
            return
        now = time.monotonic()
        if run_id in self._worktree_refreshing or (
            now - self._worktree_refreshed_at.get(run_id, 0)
            < WORKTREE_STATUS_REFRESH_SECONDS
        ):
            return
        self._worktree_refreshing.add(run_id)
        self._worktree_refreshed_at[run_id] = now
        Thread(
            target=self._refresh_worktree_status,
            args=(run_id, str(path_value), str(base_commit), str(branch)),
            name=f"room-git-{run_id}",
            daemon=True,
        ).start()

    def _refresh_worktree_status(
        self, run_id: str, path_value: str, base_commit: str, branch: str
    ) -> None:
        path = Path(path_value)
        updates: dict[str, Any]
        if not path.is_dir():
            updates = {"worktree_missing": True}
        else:
            updates = self._read_worktree_status(path, base_commit, branch)
        with self._lock:
            self._worktree_refreshing.discard(run_id)
            run = self._runs.get(run_id)
            if run is None or (
                str(run.get("worktree_path")) != path_value
                or str(run.get("base_commit")) != base_commit
                or str(run.get("branch")) != branch
            ):
                return
            if "new_commit_count" in updates and run.get("merge_status") not in {
                "validating",
                "merged",
            }:
                updates["merge_status"] = (
                    "ready"
                    if not updates["worktree_dirty"]
                    and updates["new_commit_count"] > 0
                    and not run.get("runtime_live")
                    else "not_ready"
                )
            changed = any(run.get(key) != value for key, value in updates.items())
            run.update(updates)
            if changed:
                self._persist_locked()

    @staticmethod
    def _read_worktree_status(
        path: Path, base_commit: str, branch: str
    ) -> dict[str, Any]:
        try:
            status = git_output(
                path,
                "status",
                "--porcelain",
                "--untracked-files=all",
                timeout=WORKTREE_STATUS_TIMEOUT_SECONDS,
            ).splitlines()
            new_commit_count = int(
                git_output(
                    path,
                    "rev-list",
                    "--count",
                    f"{base_commit}..{branch}",
                    timeout=WORKTREE_STATUS_TIMEOUT_SECONDS,
                ).strip()
            )
            return {
                "worktree_missing": False,
                "worktree_status_error": False,
                "uncommitted_files": len(status),
                "worktree_dirty": bool(status),
                "new_commit_count": new_commit_count,
            }
        except (OSError, subprocess.SubprocessError, ValueError):
            return {"worktree_status_error": True}

    def _load_profiles(self) -> list[dict[str, str]]:
        config = VibeConfig.load()
        manager = AgentManager(LegacyConfigOrchestrator(config))
        profiles = []
        for name, profile in manager.available_agents.items():
            if name == BuiltinAgentName.ORCHESTRATOR:
                continue
            profiles.append({
                "name": profile.name,
                "display_name": profile.display_name,
                "description": profile.description,
                "safety": profile.safety.value,
            })
        return profiles

    @staticmethod
    def _load_tools() -> list[dict[str, str]]:
        config = VibeConfig.load()
        manager = ToolManager(lambda: config, defer_mcp=True)
        return [
            {"name": name, "display_name": name.replace("_", " ").title()}
            for name in sorted(manager.available_tools)
            if name != "exit_plan_mode"
        ]

    def _enabled_tools(self, payload: dict[str, Any]) -> list[str] | None:
        available = {item["name"] for item in self._tools}
        raw = payload.get("enabled_tools")
        if raw is None:
            return None
        if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
            raise ValueError("enabled_tools must be a list of tool names")
        enabled = list(dict.fromkeys(item.strip() for item in raw if item.strip()))
        unknown = sorted(set(enabled) - available)
        if unknown:
            raise ValueError(f"Unknown tools: {', '.join(unknown)}")
        return None if set(enabled) == available else enabled

    @staticmethod
    def _validated_images(payload: dict[str, Any]) -> list[dict[str, Any]]:
        raw = payload.get("images")
        if raw is None:
            return []
        if not isinstance(raw, list) or len(raw) > MAX_ROOM_IMAGES:
            raise ValueError(f"images must contain at most {MAX_ROOM_IMAGES} items")
        images = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("Each image must be an object")
            alias = item.get("alias")
            mime_type = item.get("mime_type")
            data = item.get("data")
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError("Each image requires a name")
            if mime_type not in ALLOWED_IMAGE_TYPES:
                raise ValueError("Unsupported image type")
            if not isinstance(data, str):
                raise ValueError("Each image requires base64 data")
            try:
                decoded = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError) as error:
                raise ValueError("Image data is not valid base64") from error
            if len(decoded) > MAX_ROOM_IMAGE_BYTES:
                raise ValueError(
                    f"Each image must be at most {MAX_ROOM_IMAGE_BYTES // 1024 // 1024} MB"
                )
            images.append({
                "source": {"kind": "inline", "data": data},
                "alias": alias.strip()[:120],
                "mime_type": mime_type,
            })
        return images

    def _coordination_locked(self) -> dict[str, Any]:
        live = [run for run in self._runs.values() if run.get("runtime_live")]
        context_tokens = sum(int(run.get("context_tokens") or 0) for run in live)
        context_limit = sum(int(run.get("context_limit") or 0) for run in live)
        return {
            "orchestrator_id": ORCHESTRATOR_ID,
            "agents": len(live),
            "working": sum(run.get("state") in {"running", "working"} for run in live),
            "attention": sum(
                run.get("state") in {"attention", "failed"} for run in live
            ),
            "queued_messages": sum(
                int(run.get("queued_messages") or 0) for run in live
            ),
            "context_tokens": context_tokens,
            "context_limit": context_limit or None,
            "memory_percent": round(context_tokens / context_limit * 100, 1)
            if context_limit
            else 0,
            "workdir": str(self._workdir),
            "integration_branch": self._integration_branch,
            "isolation": "one Git worktree and process per agent",
        }

    def _mark_interrupted_runs(self) -> None:
        with self._lock:
            for run in self._runs.values():
                run["runtime_live"] = False
                run["resumable"] = True
                for approval in run.get("approvals", []):
                    if approval.get("status") == "pending":
                        approval["status"] = "expired"
                for question in run.get("questions", []):
                    if question.get("status") == "pending":
                        question["status"] = "expired"
                if run.get("state") in LIVE_STATES:
                    run["state"] = "stopped"
                    run["current_activity"] = "Interrupted when Agent Room stopped"
                    for message in run.get("conversation", []):
                        if message.get("status") in {"queued", "running"}:
                            message["status"] = "cancelled"
                    self._append_event(run, "interrupted", "Room process restarted")
            self._persist_locked()

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        try:
            payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, list):
            return {}
        runs = {}
        for item in payload:
            if not isinstance(item, dict) or not isinstance(
                item.get("tool_call_id"), str
            ):
                continue
            item.setdefault("approvals", [])
            item.setdefault("questions", [])
            item.setdefault("conversation", [])
            item.setdefault("events", [])
            item.setdefault("resumable", True)
            runs[item["tool_call_id"]] = item
        return runs

    def _persist_locked(self) -> None:
        self._revision += 1
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._registry_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(list(self._runs.values()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self._registry_path)

    def _prune_runs_locked(self) -> None:
        while len(self._runs) > MAX_STORED_RUNS:
            expired = next(
                (
                    run_id
                    for run_id, run in self._runs.items()
                    if run_id != ORCHESTRATOR_ID and not run.get("runtime_live")
                ),
                None,
            )
            if expired is None:
                return
            del self._runs[expired]

    @staticmethod
    def _managed_snapshot(run: dict[str, Any]) -> dict[str, Any]:
        state_name = {
            "requested": ManagedAgentState.STARTING,
            "running": ManagedAgentState.RUNNING,
            "working": ManagedAgentState.WORKING,
            "attention": ManagedAgentState.ATTENTION,
            "idle": ManagedAgentState.IDLE,
            "failed": ManagedAgentState.FAILED,
            "completed": ManagedAgentState.STOPPED,
            "cancelled": ManagedAgentState.STOPPED,
            "stopped": ManagedAgentState.STOPPED,
        }.get(str(run.get("state") or ""), ManagedAgentState.FAILED)
        usage = run.get("usage") or {}
        return {
            "agent_id": run["tool_call_id"],
            "child_session_id": run.get("child_session_id") or run["tool_call_id"],
            "profile": run["agent_name"],
            "state": state_name.value,
            "task": run.get("task") or "Agent task",
            "current_activity": run.get("current_activity"),
            "last_response": next(
                (
                    item.get("content", "")
                    for item in reversed(run.get("conversation", []))
                    if item.get("role") == "assistant"
                ),
                "",
            ),
            "error": run.get("error"),
            "queued_messages": int(run.get("queued_messages") or 0),
            "started_at": float(run.get("started_at") or 0),
            "updated_at": float(run.get("updated_at") or 0),
            "turns_used": int(run.get("turns_used") or 0),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "context_tokens": int(run.get("context_tokens") or 0),
            "context_limit": run.get("context_limit"),
            "estimated_cost_usd": float(run.get("estimated_cost_usd") or 0),
            "model": run.get("model"),
        }

    @staticmethod
    def _message(
        role: str, content: str, client_message_id: str | None, status: str
    ) -> dict[str, Any]:
        now = time.time()
        return {
            "id": f"message-{uuid4().hex}",
            "client_message_id": client_message_id,
            "role": role,
            "content": content[:MAX_MESSAGE_CHARS],
            "status": status,
            "created_at": now,
            "updated_at": now,
            "error_code": None,
        }

    @staticmethod
    def _find_client_message(
        run: dict[str, Any], client_id: str
    ) -> dict[str, Any] | None:
        return next(
            (
                item
                for item in run.get("conversation", [])
                if item.get("client_message_id") == client_id
            ),
            None,
        )

    @staticmethod
    def _message_by_id(run: dict[str, Any], message_id: Any) -> dict[str, Any] | None:
        return next(
            (
                item
                for item in run.get("conversation", [])
                if item.get("id") == message_id
            ),
            None,
        )

    @staticmethod
    def _trim_conversation(run: dict[str, Any]) -> None:
        run["conversation"] = run["conversation"][-MAX_CONVERSATION_ITEMS:]

    @staticmethod
    def _expire_interactions(run: dict[str, Any]) -> None:
        for approval in run.get("approvals", []):
            if approval.get("status") == "pending":
                approval["status"] = "expired"
        for question in run.get("questions", []):
            if question.get("status") == "pending":
                question["status"] = "expired"

    @staticmethod
    def _append_event(
        run: dict[str, Any], kind: str, label: str, detail: str | None = None
    ) -> None:
        run.setdefault("events", []).append({
            "id": f"event-{uuid4().hex}",
            "at": time.time(),
            "kind": kind,
            "label": label,
            "detail": detail[:1_000] if detail else None,
        })
        run["events"] = run["events"][-MAX_EVENTS:]

    @staticmethod
    def _worktree_name(display_name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", display_name.lower()).strip("-") or "agent"
        return f"room-{slug[:24]}-{uuid4().hex[:8]}"

    @staticmethod
    def _public_run(run: dict[str, Any]) -> dict[str, Any]:
        return json.loads(
            json.dumps({
                key: value for key, value in run.items() if key != "worker_logs"
            })
        )

    def _get_run_locked(self, run_id: str) -> dict[str, Any]:
        try:
            return self._runs[run_id]
        except KeyError as error:
            raise KeyError(run_id) from error

    def _lifecycle_lock(self, run_id: str) -> RLock:
        with self._lock:
            return self._lifecycle_locks.setdefault(run_id, RLock())

    @staticmethod
    def required_text(payload: dict[str, Any], name: str, limit: int) -> str:
        value = payload.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required")
        return value.strip()[:limit]

    @staticmethod
    def _optional_text(payload: dict[str, Any], name: str, limit: int) -> str | None:
        value = payload.get(name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        return value.strip()[:limit] or None

    @staticmethod
    def _optional_bool(payload: dict[str, Any], name: str, *, default: bool) -> bool:
        value = payload.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        return value


class suppress_errors:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> bool:
        return True


def git_output(path: Path, *args: str, timeout: float = 60) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout


def safe_error(error: Exception) -> str:
    return (str(error).strip() or type(error).__name__)[:1_000]


def resolve_worker_network(mode: str) -> tuple[dict[str, str], dict[str, Any]]:
    if mode not in {"auto", "inherit", "direct"}:
        raise ValueError("network mode must be auto, inherit, or direct")
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proxy_configured = any(environment.get(key) for key in PROXY_ENV_KEYS)
    proxied = probe_mistral_transport(trust_env=True) if proxy_configured else None
    direct = probe_mistral_transport(trust_env=False)
    selected = mode
    if mode == "auto":
        selected = (
            "direct"
            if proxy_configured
            and proxied is not None
            and not proxied["reachable"]
            and direct["reachable"]
            else "inherit"
        )
    if selected == "direct":
        for key in PROXY_ENV_KEYS:
            environment.pop(key, None)

    config = VibeConfig.load()
    provider = config.get_provider_for_model(config.get_active_model())
    credential = resolve_api_key(provider.api_key_env_var)
    auth_probe = probe_mistral_transport(
        trust_env=selected != "direct", credential=credential
    )
    status = {
        "requested_mode": mode,
        "selected_mode": selected,
        "proxy_configured": proxy_configured,
        "proxy_reachable": proxied["reachable"] if proxied is not None else None,
        "direct_reachable": direct["reachable"],
        "api_reachable": auth_probe["reachable"],
        "credential_resolved": bool(credential),
        "credential_source": (
            "environment"
            if os.environ.get(provider.api_key_env_var)
            else ("keyring" if credential else "none")
        ),
        "authenticated": auth_probe["status"] == HTTPStatus.OK,
        "http_status": auth_probe["status"],
        "error": auth_probe["error"],
    }
    return environment, status


def probe_mistral_transport(
    *, trust_env: bool, credential: str | None = None
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {credential}"} if credential else None
    try:
        response = httpx.get(
            NETWORK_PROBE_URL, headers=headers, timeout=10, trust_env=trust_env
        )
        return {"reachable": True, "status": response.status_code, "error": None}
    except httpx.HTTPError as error:
        return {"reachable": False, "status": None, "error": safe_error(error)}


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if is_windows():
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
        )
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress_errors():
            os.killpg(process.pid, signal.SIGKILL)


class AgentRoomHandler(SimpleHTTPRequestHandler):
    server: AgentRoomHTTPServer  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(REPOSITORY_ROOT), **kwargs)

    def do_GET(self) -> None:
        if not self._is_loopback_request():
            self._send_json({"error": "Loopback access only"}, HTTPStatus.FORBIDDEN)
            return
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(self.server.store.health())
            return
        if path == "/api/agent-runs":
            self._send_json(self.server.store.snapshot())
            return
        if path == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/web/agent-room/")
            self.end_headers()
            return
        if path not in ALLOWED_STATIC_PATHS:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if not self._is_loopback_request():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if urlparse(self.path).path not in ALLOWED_STATIC_PATHS:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        super().do_HEAD()

    def do_POST(self) -> None:
        if not self._is_loopback_request():
            self._send_json({"error": "Loopback access only"}, HTTPStatus.FORBIDDEN)
            return
        try:
            payload = self._read_json()
            path = urlparse(self.path).path
            if path == "/api/agent-runs":
                self._send_json(self.server.store.create(payload), HTTPStatus.ACCEPTED)
                return
            if path == "/api/team-workspace":
                self._send_json(self.server.store.update_team_workspace(payload))
                return
            parts = path.strip("/").split("/")
            if len(parts) < RUN_ACTION_PARTS or parts[:2] != ["api", "agent-runs"]:
                raise KeyError(path)
            run_id = parts[2]
            action = parts[3]
            if len(parts) == RUN_ACTION_PARTS and action == "messages":
                self._send_json(
                    self.server.store.send_message(run_id, payload), HTTPStatus.ACCEPTED
                )
            elif len(parts) == RUN_ACTION_PARTS and action == "cancel":
                self._send_json(self.server.store.cancel(run_id))
            elif len(parts) == RUN_ACTION_PARTS and action == "stop":
                self._send_json(self.server.store.stop(run_id))
            elif len(parts) == RUN_ACTION_PARTS and action == "group":
                group_id = AgentRoomStore.required_text(payload, "group_id", 80)
                self._send_json(self.server.store.update_group(run_id, group_id))
            elif len(parts) == RUN_ACTION_PARTS and action == "merge":
                self._send_json(self.server.store.merge(run_id))
            elif len(parts) == RUN_DETAIL_PARTS and action == "approvals":
                self._send_json(
                    self.server.store.resolve_approval(run_id, parts[4], payload)
                )
            elif len(parts) == RUN_DETAIL_PARTS and action == "questions":
                self._send_json(
                    self.server.store.answer_question(run_id, parts[4], payload)
                )
            else:
                raise KeyError(path)
        except KeyError:
            self._send_json(
                {"error": "Not found", "error_code": "not_found"}, HTTPStatus.NOT_FOUND
            )
        except RuntimeError as error:
            self._send_json(
                {"error": str(error), "error_code": "conflict"}, HTTPStatus.CONFLICT
            )
        except (ValueError, TimeoutError, subprocess.SubprocessError) as error:
            self._send_json(
                {"error": str(error), "error_code": "validation"},
                HTTPStatus.BAD_REQUEST,
            )

    def end_headers(self) -> None:
        if urlparse(self.path).path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        if not urlparse(self.path).path.startswith("/api/"):
            super().log_message(format, *args)

    def _is_loopback_request(self) -> bool:
        allowed_hosts = {"127.0.0.1", "localhost", "::1"}
        host = self.headers.get("Host", "").rsplit(":", 1)[0].strip("[]")
        if host not in allowed_hosts:
            return False
        origin = self.headers.get("Origin")
        return not origin or urlparse(origin).hostname in allowed_hosts

    def _read_json(self) -> dict[str, Any]:
        if "application/json" not in self.headers.get("Content-Type", ""):
            raise ValueError("Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length") from error
        if length < MIN_JSON_BODY_BYTES or length > MAX_JSON_BODY_BYTES:
            raise ValueError("Invalid request size")
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise ValueError("Invalid JSON body") from error
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _send_json(
        self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class AgentRoomHTTPServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], store: AgentRoomStore) -> None:
        super().__init__(address, AgentRoomHandler)
        self.store = store


class AgentRoomOwnerLock:
    def __init__(self) -> None:
        vibe_home = Path(os.environ.get("VIBE_HOME", "~/.vibe")).expanduser()
        self.path = vibe_home / OWNER_LOCK_FILE
        self._handle: Any | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            if os.name == "nt":
                handle.seek(0)
                if not handle.read(1):
                    handle.write("0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise RuntimeError(
                "Another Agent Room backend already owns this VIBE_HOME"
            ) from error
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _discovery_path() -> Path:
    vibe_home = Path(os.environ.get("VIBE_HOME", "~/.vibe")).expanduser()
    return vibe_home / DISCOVERY_FILE


def write_discovery(store: AgentRoomStore, port: int, workdir: Path) -> None:
    path = _discovery_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = store.snapshot()
    payload = {
        "api_version": API_VERSION,
        "instance_id": snapshot["instance_id"],
        "pid": os.getpid(),
        "url": f"http://{LOOPBACK_HOST}:{port}",
        "workdir": str(workdir),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def remove_discovery(instance_id: str) -> None:
    path = _discovery_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("instance_id") == instance_id:
            path.unlink(missing_ok=True)
    except (FileNotFoundError, OSError, json.JSONDecodeError, AttributeError):
        return


def assert_endpoint_available(port: int) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as endpoint:
            endpoint.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            endpoint.bind((LOOPBACK_HOST, port))
    except OSError as error:
        raise RuntimeError(
            f"Agent Room endpoint {LOOPBACK_HOST}:{port} is already in use"
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Vibe Agent Room")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--workdir", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--network-mode",
        choices=("auto", "inherit", "direct"),
        default="auto",
        help="Worker proxy policy; auto bypasses a broken inherited proxy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = args.workdir.expanduser().resolve()
    if not workdir.is_dir():
        raise SystemExit(f"Workdir does not exist: {workdir}")
    init_harness_files_manager("user", "project")
    owner_lock = AgentRoomOwnerLock()
    try:
        owner_lock.acquire()
        assert_endpoint_available(args.port)
    except RuntimeError as error:
        owner_lock.release()
        discovered = _discovery_path()
        raise SystemExit(f"{error}. Discovery: {discovered}") from error
    try:
        store = AgentRoomStore(workdir, network_mode=args.network_mode)
    except BaseException:
        owner_lock.release()
        raise
    try:
        server = AgentRoomHTTPServer((LOOPBACK_HOST, args.port), store)
    except BaseException:
        store.close()
        owner_lock.release()
        raise
    write_discovery(store, args.port, workdir)
    instance_id = store.snapshot()["instance_id"]
    print(f"Agent Room: http://{LOOPBACK_HOST}:{args.port}/web/agent-room/")
    print(f"Integration worktree: {workdir}")
    network = store.snapshot()["network"]
    print(
        f"Mistral network: {network['selected_mode']} "
        f"(authenticated={network['authenticated']})"
    )

    def stop_on_signal(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()
        remove_discovery(instance_id)
        owner_lock.release()


if __name__ == "__main__":
    main()
