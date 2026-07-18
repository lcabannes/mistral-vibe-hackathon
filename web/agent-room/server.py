from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Lock, Thread, Timer
import time
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from vibe.core.utils.platform import is_windows

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUP = "unassigned"
AVAILABLE_PROFILES = (
    {"name": "default", "display_name": "Default"},
    {"name": "plan", "display_name": "Plan"},
)
PROFILE_NAMES = {profile["name"] for profile in AVAILABLE_PROFILES}
TERMINAL_STATES = {"failed", "completed", "cancelled"}
COATS = ("orange", "mint", "rose", "blue", "violet", "charcoal", "sunny")
CANCEL_PATH_PARTS = 4
MIN_JSON_BODY_BYTES = 2
MAX_JSON_BODY_BYTES = 16_384
MAX_CONCURRENT_RUNS = 3
MAX_STORED_RUNS = 100
MAX_TURNS = 16
MAX_TOKENS = 120_000
MAX_PRICE_DOLLARS = 1.0
RUN_TIMEOUT_SECONDS = 30 * 60
LOOPBACK_HOST = "127.0.0.1"
ALLOWED_STATIC_PATHS = {
    "/web/agent-room/",
    "/web/agent-room/index.html",
    "/web/agent-room/styles.css",
    "/web/agent-room/app.js",
    "/web/agent-room/agents.json",
    "/distribution/zed/icons/mistral_vibe.svg",
}


class AgentRunStore:
    def __init__(self, workdir: Path) -> None:
        self._workdir = workdir
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = Lock()
        vibe_home = Path(os.environ.get("VIBE_HOME", "~/.vibe")).expanduser()
        self._session_dir = vibe_home / "logs" / "session"
        self._registry_path = vibe_home / "agent-room" / "runs.json"
        self._runs = self._load_registry()
        changed = False
        for run in self._runs.values():
            if run.get("state") in TERMINAL_STATES:
                continue
            run["state"] = "cancelled"
            run["current_activity"] = "Interrupted when Agent Room stopped"
            run["updated_at"] = time.time()
            self._append_event(
                run,
                "interrupted",
                "Run interrupted",
                "The Agent Room process stopped before this run finished.",
            )
            changed = True
        if changed:
            self._persist_locked()

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public_run(run) for run in self._runs.values()]

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile = self.required_text(payload, "agent_name", 40)
        if profile not in PROFILE_NAMES:
            raise ValueError(f"Unknown agent profile: {profile}")
        task = self.required_text(payload, "task", 4_000)
        display_name = self._optional_text(payload, "display_name", 40) or profile
        group_id = self._optional_text(payload, "group_id", 80) or DEFAULT_GROUP
        run_id = f"web-{uuid4().hex}"
        now = time.time()
        run = {
            "tool_call_id": run_id,
            "parent_session_id": run_id,
            "child_session_id": None,
            "agent_name": profile,
            "agent_display_name": display_name,
            "task": task,
            "state": "requested",
            "started_at": now,
            "updated_at": now,
            "current_activity": "Queued by Agent Room",
            "turns_used": None,
            "usage": None,
            "estimated_cost_usd": None,
            "model": None,
            "is_primary": True,
            "group_id": group_id,
            "coat": COATS[0],
            "source": "live",
            "events": [
                {
                    "at": now,
                    "kind": "queued",
                    "label": "Queued from Agent Room",
                    "detail": task,
                }
            ],
            "conversation": [{"role": "user", "content": task, "at": now}],
            "error": None,
            "cancel_requested": False,
            "timed_out": False,
            "resumable": False,
        }
        with self._lock:
            active_count = sum(
                candidate.get("state") not in TERMINAL_STATES
                for candidate in self._runs.values()
            )
            if active_count >= MAX_CONCURRENT_RUNS:
                raise ValueError(
                    f"At most {MAX_CONCURRENT_RUNS} agent runs can be active at once"
                )
            while len(self._runs) >= MAX_STORED_RUNS:
                terminal = next(
                    (
                        candidate_id
                        for candidate_id, candidate in self._runs.items()
                        if candidate.get("state") in TERMINAL_STATES
                    ),
                    None,
                )
                if terminal is None:
                    break
                self._runs.pop(terminal)
            run["coat"] = COATS[len(self._runs) % len(COATS)]
            self._runs[run_id] = run
            self._persist_locked()
        Thread(target=self._execute, args=(run_id,), daemon=True).start()
        return self._public_run(run)

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            if run["state"] in TERMINAL_STATES:
                return self._public_run(run)
            process = self._processes.get(run_id)
            run["cancel_requested"] = True
            run["current_activity"] = "Cancellation requested"
            run["updated_at"] = time.time()
            self._append_event(run, "cancel", "Cancellation requested")
            self._persist_locked()
        if process is not None:
            self._terminate_process(process)
        return self._public_run(run)

    def update_group(self, run_id: str, group_id: str) -> dict[str, Any]:
        normalized_group = group_id.strip()[:80]
        if not normalized_group:
            raise ValueError("group_id is required")
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            run["group_id"] = normalized_group
            self._persist_locked()
            return self._public_run(run)

    def cancel_all(self) -> None:
        with self._lock:
            processes = tuple(self._processes.values())
        for process in processes:
            self._terminate_process(process)

    def _execute(self, run_id: str) -> None:
        run_input = self._mark_running(run_id)
        if run_input is None:
            return
        profile, task = run_input
        prefix = f"room-{run_id[4:12]}"
        env = self._run_environment(prefix)
        command = self._run_command(profile, task)
        process: subprocess.Popen[str] | None = None
        timeout: Timer | None = None
        try:
            process = self._start_process(run_id, command, env)
            if process is None:
                return
            timeout = Timer(RUN_TIMEOUT_SECONDS, self._timeout_run, args=(run_id,))
            timeout.daemon = True
            timeout.start()
            if process.stdout is not None:
                for line in process.stdout:
                    self._observe_output(run_id, line.rstrip())
            return_code = process.wait()
            self._finalize(run_id, prefix, return_code)
        except Exception as error:
            if process is not None:
                self._terminate_process(process)
            self._mark_launch_failed(run_id, error)
        finally:
            if timeout is not None:
                timeout.cancel()
            with self._lock:
                self._processes.pop(run_id, None)

    def _mark_running(self, run_id: str) -> tuple[str, str] | None:
        with self._lock:
            run = self._runs[run_id]
            if run.get("cancel_requested"):
                run["state"] = "cancelled"
                run["current_activity"] = "Run cancelled before launch"
                run["updated_at"] = time.time()
                self._append_event(run, "cancelled", "Run cancelled before launch")
                self._persist_locked()
                return None
            run["state"] = "running"
            run["current_activity"] = "Starting Vibe"
            run["updated_at"] = time.time()
            self._append_event(run, "running", "Vibe process started")
            self._persist_locked()
            profile = str(run["agent_name"])
            task = str(run["task"])
            return profile, task

    def _run_environment(self, prefix: str) -> dict[str, str]:
        env = os.environ.copy()
        env["VIBE_SESSION_LOGGING"] = json.dumps({
            "enabled": True,
            "save_dir": str(self._session_dir),
            "session_prefix": prefix,
        })
        return env

    def _run_command(self, profile: str, task: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "vibe.cli.entrypoint",
            "-p",
            task,
            "--agent",
            profile,
            "--output",
            "streaming",
            "--max-turns",
            str(MAX_TURNS),
            "--max-tokens",
            str(MAX_TOKENS),
            "--max-price",
            str(MAX_PRICE_DOLLARS),
            "--workdir",
            str(self._workdir),
            "--trust",
        ]

    def _start_process(
        self, run_id: str, command: list[str], env: dict[str, str]
    ) -> subprocess.Popen[str] | None:
        with self._lock:
            run = self._runs[run_id]
            if run.get("cancel_requested"):
                run["state"] = "cancelled"
                run["current_activity"] = "Run cancelled before launch"
                run["updated_at"] = time.time()
                self._append_event(run, "cancelled", "Run cancelled before launch")
                self._persist_locked()
                return None
            process = subprocess.Popen(
                command,
                cwd=self._workdir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=not is_windows(),
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if is_windows() else 0
                ),
            )
            self._processes[run_id] = process
            return process

    def _mark_launch_failed(self, run_id: str, error: Exception) -> None:
        with self._lock:
            run = self._runs[run_id]
            run["state"] = "failed"
            run["error"] = str(error)
            run["current_activity"] = str(error)
            run["updated_at"] = time.time()
            self._append_event(run, "failed", "Could not launch Vibe", str(error))
            self._persist_locked()

    def _observe_output(self, run_id: str, line: str) -> None:
        if not line:
            return
        now = time.time()
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            with self._lock:
                run = self._runs[run_id]
                self._append_event(run, "log", "Vibe output", line[:600])
                run["updated_at"] = now
                self._persist_locked()
            return
        if not isinstance(message, dict):
            return
        role = str(message.get("role") or "assistant")
        content = self._message_content(message.get("content"))
        if not content or role == "system":
            return
        with self._lock:
            run = self._runs[run_id]
            if (
                role == "user"
                and run["conversation"]
                and run["conversation"][-1]["role"] == "user"
                and run["conversation"][-1]["content"] == content
            ):
                return
            run["state"] = "working"
            run["current_activity"] = self._activity_label(role, content)
            run["updated_at"] = now
            run["conversation"].append({
                "role": role,
                "content": content[:4_000],
                "at": now,
            })
            run["conversation"] = run["conversation"][-80:]
            self._append_event(
                run,
                "message",
                f"{role.replace('_', ' ').title()} message",
                content[:600],
            )
            self._persist_locked()

    def _finalize(self, run_id: str, prefix: str, return_code: int) -> None:
        metadata = self._load_metadata(prefix)
        with self._lock:
            run = self._runs[run_id]
            was_cancelled = bool(run.get("cancel_requested"))
            if run.get("timed_out"):
                state = "failed"
                activity = "Run stopped after reaching the 30 minute limit"
            elif was_cancelled:
                state = "cancelled"
                activity = "Run cancelled"
            elif return_code == 0:
                state = "completed"
                activity = "Run completed"
            else:
                state = "failed"
                activity = f"Vibe exited with status {return_code}"
            run["state"] = state
            run["current_activity"] = activity
            run["updated_at"] = time.time()
            run["error"] = None if return_code == 0 or was_cancelled else activity
            self._apply_metadata(run, metadata)
            self._append_event(run, state, activity)
            self._persist_locked()

    def _timeout_run(self, run_id: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            process = self._processes.get(run_id)
            if run is None or process is None or process.poll() is not None:
                return
            run["timed_out"] = True
            run["current_activity"] = "Stopping after reaching the time limit"
            run["updated_at"] = time.time()
            self._append_event(run, "timeout", "Run reached the 30 minute limit")
            self._persist_locked()
        self._terminate_process(process)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if is_windows():
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    capture_output=True,
                )
            except OSError:
                process.terminate()
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        def kill_after_grace() -> None:
            if process.poll() is not None:
                return
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        escalation = Timer(5, kill_after_grace)
        escalation.daemon = True
        escalation.start()

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        try:
            payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, list):
            return {}
        runs: dict[str, dict[str, Any]] = {}
        for run in payload:
            if not isinstance(run, dict):
                continue
            run_id = run.get("tool_call_id")
            if not isinstance(run_id, str):
                continue
            run["source"] = "live"
            runs[run_id] = run
        return runs

    def _persist_locked(self) -> None:
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._registry_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(list(self._runs.values()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self._registry_path)

    def _load_metadata(self, prefix: str) -> dict[str, Any] | None:
        candidates = sorted(
            self._session_dir.glob(f"{prefix}_*/meta.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        try:
            data = json.loads(candidates[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _apply_metadata(run: dict[str, Any], metadata: dict[str, Any] | None) -> None:
        if metadata is None:
            return
        stats = metadata.get("stats")
        if isinstance(stats, dict):
            prompt = stats.get("session_prompt_tokens")
            completion = stats.get("session_completion_tokens")
            if isinstance(prompt, int) and isinstance(completion, int):
                run["usage"] = {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                }
                input_price = stats.get("input_price_per_million")
                output_price = stats.get("output_price_per_million")
                if isinstance(input_price, int | float) and isinstance(
                    output_price, int | float
                ):
                    run["estimated_cost_usd"] = (
                        prompt * input_price + completion * output_price
                    ) / 1_000_000
            steps = stats.get("steps")
            if isinstance(steps, int):
                run["turns_used"] = steps
        session_id = metadata.get("session_id")
        if isinstance(session_id, str):
            run["parent_session_id"] = session_id
            run["resumable"] = True
        config = metadata.get("config")
        if isinstance(config, dict) and isinstance(config.get("active_model"), str):
            run["model"] = config["active_model"]

    @staticmethod
    def _append_event(
        run: dict[str, Any], kind: str, label: str, detail: str | None = None
    ) -> None:
        run["events"].append({
            "at": time.time(),
            "kind": kind,
            "label": label,
            "detail": detail,
        })
        run["events"] = run["events"][-80:]

    @staticmethod
    def _message_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _activity_label(role: str, content: str) -> str:
        if role == "assistant":
            return f"Responding: {content[:90]}"
        if role == "tool":
            return f"Tool result: {content[:90]}"
        return f"Processing {role.replace('_', ' ')} message"

    @staticmethod
    def _public_run(run: dict[str, Any]) -> dict[str, Any]:
        public = {
            key: value
            for key, value in run.items()
            if key not in {"cancel_requested", "timed_out"}
        }
        return json.loads(json.dumps(public))

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


class AgentRoomHandler(SimpleHTTPRequestHandler):
    server: AgentRoomHTTPServer

    def do_GET(self) -> None:
        if not self._is_loopback_request():
            self._send_json({"error": "Loopback access only"}, HTTPStatus.FORBIDDEN)
            return
        path = urlparse(self.path).path
        if path == "/api/agent-runs":
            self._send_json({
                "connected": True,
                "activities": self.server.store.snapshot(),
                "profiles": AVAILABLE_PROFILES,
            })
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
        path = urlparse(self.path).path
        if path not in ALLOWED_STATIC_PATHS:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        super().do_HEAD()

    def do_POST(self) -> None:
        if not self._is_loopback_request():
            self._send_json({"error": "Loopback access only"}, HTTPStatus.FORBIDDEN)
            return
        path = urlparse(self.path).path
        if path == "/api/agent-runs":
            self._create_run()
            return

        parts = path.strip("/").split("/")
        if len(parts) != CANCEL_PATH_PARTS or parts[:2] != ["api", "agent-runs"]:
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        match parts[3]:
            case "cancel":
                self._cancel_run(parts[2])
            case "group":
                self._update_run_group(parts[2])
            case _:
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def _create_run(self) -> None:
        try:
            payload = self._read_json()
            run = self.server.store.create(payload)
        except ValueError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(run, HTTPStatus.ACCEPTED)

    def _cancel_run(self, run_id: str) -> None:
        try:
            run = self.server.store.cancel(run_id)
        except KeyError:
            self._send_json({"error": "Run not found"}, HTTPStatus.NOT_FOUND)
            return
        self._send_json(run)

    def _update_run_group(self, run_id: str) -> None:
        try:
            payload = self._read_json()
            group_id = AgentRunStore.required_text(payload, "group_id", 80)
            run = self.server.store.update_group(run_id, group_id)
        except ValueError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        except KeyError:
            self._send_json({"error": "Run not found"}, HTTPStatus.NOT_FOUND)
            return
        self._send_json(run)

    def end_headers(self) -> None:
        if urlparse(self.path).path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        if urlparse(self.path).path.startswith("/api/"):
            return
        super().log_message(format, *args)

    def _is_loopback_request(self) -> bool:
        allowed_hosts = {"127.0.0.1", "localhost", "::1"}
        host = self.headers.get("Host", "").rsplit(":", 1)[0].strip("[]")
        if host not in allowed_hosts:
            return False
        origin = self.headers.get("Origin")
        if not origin:
            return True
        return urlparse(origin).hostname in allowed_hosts

    def _read_json(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
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
    def __init__(self, address: tuple[str, int], store: AgentRunStore) -> None:
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
    store = AgentRunStore(workdir)
    os.chdir(REPOSITORY_ROOT)
    server = AgentRoomHTTPServer((LOOPBACK_HOST, args.port), store)
    print(f"Agent Room: http://{LOOPBACK_HOST}:{args.port}/web/agent-room/")
    print(f"Agent workdir: {workdir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        store.cancel_all()
        server.server_close()


if __name__ == "__main__":
    main()
