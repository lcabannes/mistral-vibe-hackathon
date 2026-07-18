from __future__ import annotations

import argparse
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
import subprocess
import sys
from threading import Event, RLock, Thread
import time
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vibe.core.agents.manager import AgentManager
from vibe.core.agents.models import BuiltinAgentName, ManagedAgentState
from vibe.core.config import VibeConfig
from vibe.core.config.harness_files import init_harness_files_manager
from vibe.core.config.orchestrator_legacy import LegacyConfigOrchestrator
from vibe.core.utils.platform import is_windows
from vibe.core.worktree import PreparedWorktree, WorktreeError, prepare_worktree_session

WORKER_PATH = Path(__file__).with_name("worker.py")
DEFAULT_GROUP = "unassigned"
ORCHESTRATOR_ID = "orchestrator"
COATS = ("orange", "mint", "rose", "blue", "violet", "charcoal", "sunny")
TERMINAL_STATES = {"completed", "cancelled"}
LIVE_STATES = {"requested", "running", "working", "attention", "idle", "failed"}
MIN_JSON_BODY_BYTES = 2
MAX_JSON_BODY_BYTES = 64_000
MAX_STORED_RUNS = 100
MAX_ACTIVE_AGENTS = 8
MAX_QUEUED_MESSAGES = 20
MAX_MESSAGE_CHARS = 8_000
MAX_CONVERSATION_ITEMS = 250
MAX_EVENTS = 180
LOOPBACK_HOST = "127.0.0.1"
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


class AgentWorker:
    def __init__(
        self,
        store: AgentRoomStore,
        run_id: str,
        profile: str,
        worktree: PreparedWorktree,
        session_root: Path,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.profile = profile
        self.worktree = worktree
        self.session_root = session_root
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
        self.process = subprocess.Popen(
            command,
            cwd=self.worktree.path,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=not is_windows(),
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if is_windows() else 0),
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
        self.store.observe_worker_exit(self.run_id, return_code)


class AgentRoomStore:
    def __init__(self, workdir: Path) -> None:
        self._workdir = workdir
        self._lock = RLock()
        self._workers: dict[str, AgentWorker] = {}
        vibe_home = Path(os.environ.get("VIBE_HOME", "~/.vibe")).expanduser()
        self._session_root = vibe_home / "logs" / "session"
        self._registry_path = vibe_home / "agent-room" / "runs.json"
        self._runs = self._load_registry()
        self._profiles = self._load_profiles()
        self._integration_branch = git_output(
            self._workdir, "branch", "--show-current"
        ).strip()
        self._mark_interrupted_runs()
        self._launch_orchestrator()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            for run in self._runs.values():
                self._refresh_worktree_status_locked(run)
            return {
                "connected": True,
                "activities": [self._public_run(run) for run in self._runs.values()],
                "profiles": deepcopy(self._profiles),
                "coordination": self._coordination_locked(),
            }

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile = self.required_text(payload, "agent_name", 80)
        if profile == BuiltinAgentName.ORCHESTRATOR:
            raise ValueError("The room already has one orchestrator")
        if profile not in {item["name"] for item in self._profiles}:
            raise ValueError(f"Unknown agent profile: {profile}")
        task = self.required_text(payload, "task", MAX_MESSAGE_CHARS)
        display_name = self._optional_text(payload, "display_name", 50) or profile
        group_id = self._optional_text(payload, "group_id", 80) or DEFAULT_GROUP
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
        if interpret_commands and content.startswith("//"):
            content = content[1:]
        elif interpret_commands and content.startswith("/"):
            return self._chat_command(run_id, content, client_message_id)
        with self._lock:
            run = self._get_run_locked(run_id)
            existing = self._find_client_message(run, client_message_id)
            if existing is not None:
                return {"message": deepcopy(existing), "run": self._public_run(run)}
            worker = self._workers.get(run_id)
            if worker is None or not run.get("runtime_live"):
                raise ValueError("This agent is no longer running")
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
            run["conversation"].append(message)
            run["queued_messages"] = int(run.get("queued_messages") or 0) + 1
            run["updated_at"] = time.time()
            self._trim_conversation(run)
            self._persist_locked()
        try:
            worker.send({
                "type": "prompt",
                "message_id": message["id"],
                "content": content,
            })
        except Exception as error:
            with self._lock:
                message["status"] = "failed"
                message["error_code"] = "worker_offline"
                run["error"] = safe_error(error)
                self._persist_locked()
            raise
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
        with self._lock:
            run = self._get_run_locked(run_id)
            worker = self._workers.pop(run_id, None)
            run["runtime_live"] = False
            run["resumable"] = False
            run["state"] = "cancelled"
            run["current_activity"] = "Agent stopped"
            for message in run.get("conversation", []):
                if message.get("status") in {"queued", "running"}:
                    message["status"] = "cancelled"
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

    def observe_worker_event(  # noqa: PLR0915
        self, worker: AgentWorker, payload: dict[str, Any]
    ) -> None:
        event_type = payload.get("type")
        if event_type == "remote_request":
            Thread(
                target=self._handle_remote_request,
                args=(worker, payload),
                name=f"room-remote-{worker.run_id}",
                daemon=True,
            ).start()
            return
        with self._lock:
            run = self._runs.get(worker.run_id)
            if run is None:
                return
            now = time.time()
            if event_type == "ready":
                run.update({
                    "child_session_id": payload.get("session_id"),
                    "parent_session_id": payload.get("session_id"),
                    "model": payload.get("model"),
                    "context_limit": payload.get("context_limit"),
                    "state": "idle",
                    "current_activity": None,
                    "runtime_live": True,
                    "resumable": True,
                    "updated_at": now,
                })
                self._append_event(run, "ready", "Worker connected")
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
                })
            elif event_type == "prompt_failed":
                message = self._message_by_id(run, payload.get("message_id"))
                if message is not None:
                    message["status"] = "failed"
                    message["error_code"] = "queue_full"
                run["error"] = str(payload.get("error") or "Prompt failed")
            run["updated_at"] = now
            self._refresh_worktree_status_locked(run)
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

    def observe_worker_exit(self, run_id: str, return_code: int) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or not run.get("runtime_live"):
                return
            run["runtime_live"] = False
            run["resumable"] = False
            run["state"] = "failed" if return_code else "cancelled"
            run["current_activity"] = (
                f"Worker exited with status {return_code}"
                if return_code
                else "Worker stopped"
            )
            run["error"] = run["current_activity"] if return_code else None
            for message in run.get("conversation", []):
                if message.get("status") in {"queued", "running"}:
                    message["status"] = "failed" if return_code else "cancelled"
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
        self._launch_worker(
            profile=BuiltinAgentName.ORCHESTRATOR,
            display_name="Orchestrator",
            group_id="coordination",
            task="Coordinate and control the room's isolated agents",
            is_orchestrator=True,
            client_message_id=None,
            fixed_run_id=ORCHESTRATOR_ID,
        )

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
    ) -> dict[str, Any]:
        with self._lock:
            active_count = sum(run.get("runtime_live") for run in self._runs.values())
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
        worker = AgentWorker(self, run_id, profile, prepared, self._session_root)
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

    def _refresh_worktree_status_locked(self, run: dict[str, Any]) -> None:
        path_value = run.get("worktree_path")
        base_commit = run.get("base_commit")
        branch = run.get("branch")
        if not path_value or not base_commit or not branch:
            return
        path = Path(str(path_value))
        if not path.is_dir():
            run["worktree_missing"] = True
            return
        try:
            status = git_output(
                path, "status", "--porcelain", "--untracked-files=all"
            ).splitlines()
            run["uncommitted_files"] = len(status)
            run["worktree_dirty"] = bool(status)
            run["new_commit_count"] = int(
                git_output(
                    path, "rev-list", "--count", f"{base_commit}..{branch}"
                ).strip()
            )
            if run.get("merge_status") not in {"validating", "merged"}:
                run["merge_status"] = (
                    "ready"
                    if not status
                    and run["new_commit_count"] > 0
                    and not run.get("runtime_live")
                    else "not_ready"
                )
        except (OSError, subprocess.SubprocessError, ValueError):
            run["worktree_status_error"] = True

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
                run["resumable"] = False
                for approval in run.get("approvals", []):
                    if approval.get("status") == "pending":
                        approval["status"] = "expired"
                for question in run.get("questions", []):
                    if question.get("status") == "pending":
                        question["status"] = "expired"
                if run.get("state") in LIVE_STATES:
                    run["state"] = "cancelled"
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
            runs[item["tool_call_id"]] = item
        return runs

    def _persist_locked(self) -> None:
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
        }.get(run.get("state"), ManagedAgentState.FAILED)
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


class suppress_errors:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> bool:
        return True


def git_output(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout


def safe_error(error: Exception) -> str:
    return (str(error).strip() or type(error).__name__)[:1_000]


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
    server: AgentRoomHTTPServer

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(REPOSITORY_ROOT), **kwargs)

    def do_GET(self) -> None:
        if not self._is_loopback_request():
            self._send_json({"error": "Loopback access only"}, HTTPStatus.FORBIDDEN)
            return
        path = urlparse(self.path).path
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Vibe Agent Room")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--workdir", type=Path, default=REPOSITORY_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = args.workdir.expanduser().resolve()
    if not workdir.is_dir():
        raise SystemExit(f"Workdir does not exist: {workdir}")
    init_harness_files_manager("user", "project")
    store = AgentRoomStore(workdir)
    server = AgentRoomHTTPServer((LOOPBACK_HOST, args.port), store)
    print(f"Agent Room: http://{LOOPBACK_HOST}:{args.port}/web/agent-room/")
    print(f"Integration worktree: {workdir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()


if __name__ == "__main__":
    main()
