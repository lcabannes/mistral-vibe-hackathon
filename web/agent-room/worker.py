from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing, suppress
import json
import os
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vibe import __version__
from vibe.core.agent_loop import AgentLoop
from vibe.core.agents.management_port import AgentManagementPort
from vibe.core.agents.models import BuiltinAgentName, ManagedAgentSnapshot
from vibe.core.config import SessionLoggingConfig, VibeConfig
from vibe.core.config.harness_files import init_harness_files_manager
from vibe.core.config.orchestrator_legacy import LegacyConfigOrchestrator
from vibe.core.control_port import (
    CLIControlAction,
    CLIControlCapabilities,
    CLIControlRequest,
    CLIControlResult,
)
from vibe.core.hooks.config import load_hooks_from_fs
from vibe.core.session.session_loader import SessionLoader
from vibe.core.telemetry.build_metadata import build_launch_context
from vibe.core.tools.base import ToolPermission
from vibe.core.tools.builtins.ask_user_question import (
    Answer,
    AskUserQuestionArgs,
    AskUserQuestionResult,
)
from vibe.core.tools.permissions import RequiredPermission
from vibe.core.types import (
    ApprovalResponse,
    AssistantEvent,
    ImageAttachment,
    ToolCallEvent,
    ToolResultEvent,
)

MAX_TURNS = 48
MAX_TOKENS = 250_000
MAX_PRICE_DOLLARS = 5.0


class RemoteAgentManagement(AgentManagementPort):
    def __init__(self, bridge: WorkerBridge) -> None:
        self._bridge = bridge
        self._snapshots: dict[str, ManagedAgentSnapshot] = {}
        self._profiles: tuple[str, ...] = ()

    def apply_snapshot(self, payload: dict[str, Any]) -> None:
        raw_agents = payload.get("agents")
        if isinstance(raw_agents, list):
            snapshots = [
                ManagedAgentSnapshot.model_validate(item) for item in raw_agents
            ]
            self._snapshots = {item.agent_id: item for item in snapshots}
        raw_profiles = payload.get("profiles")
        if isinstance(raw_profiles, list):
            self._profiles = tuple(str(item) for item in raw_profiles)

    async def start(
        self, profile: str, task: str, *, name: str | None = None
    ) -> ManagedAgentSnapshot:
        result = await self._bridge.remote_request(
            "start", {"profile": profile, "task": task, "name": name}
        )
        snapshot = ManagedAgentSnapshot.model_validate(result)
        self._snapshots[snapshot.agent_id] = snapshot
        return snapshot

    def list(self) -> tuple[ManagedAgentSnapshot, ...]:
        return tuple(self._snapshots.values())

    def available_profiles(self) -> tuple[str, ...]:
        return self._profiles

    async def message(self, agent_id: str, message: str) -> ManagedAgentSnapshot:
        result = await self._bridge.remote_request(
            "message", {"agent_id": agent_id, "message": message}
        )
        snapshot = ManagedAgentSnapshot.model_validate(result)
        self._snapshots[snapshot.agent_id] = snapshot
        return snapshot

    def output(self, agent_id: str) -> ManagedAgentSnapshot:
        try:
            return self._snapshots[agent_id]
        except KeyError as error:
            raise ValueError(f"Unknown managed agent: {agent_id}") from error

    async def stop(self, agent_id: str) -> ManagedAgentSnapshot:
        result = await self._bridge.remote_request("stop", {"agent_id": agent_id})
        snapshot = ManagedAgentSnapshot.model_validate(result)
        self._snapshots[snapshot.agent_id] = snapshot
        return snapshot


class RemoteControlPort:
    capabilities = CLIControlCapabilities(actions=frozenset(CLIControlAction))

    def __init__(self, bridge: WorkerBridge) -> None:
        self._bridge = bridge

    async def defer(self, request: CLIControlRequest) -> CLIControlResult:
        await self._bridge.remote_request(
            "control", {"request": request.model_dump(mode="json")}
        )
        return CLIControlResult(message="Queued in Agent Room")


class WorkerBridge:
    def __init__(
        self,
        profile: str,
        session_root: Path,
        *,
        disabled_tools: tuple[str, ...] = (),
        resume_session_id: str | None = None,
        auto_approve: bool = False,
    ) -> None:
        self.profile = profile
        self.session_root = session_root
        self.disabled_tools = disabled_tools
        self.resume_session_id = resume_session_id
        self.auto_approve = auto_approve
        self.agent_loop: AgentLoop | None = None
        self.management = RemoteAgentManagement(self)
        self.prompt_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
            maxsize=20
        )
        self.pending_requests: dict[str, asyncio.Future[Any]] = {}
        self.pending_approvals: dict[
            str, asyncio.Future[tuple[ApprovalResponse, str | None]]
        ] = {}
        self.pending_questions: dict[str, asyncio.Future[AskUserQuestionResult]] = {}
        self.current_turn: asyncio.Task[None] | None = None
        self.shutdown_event = asyncio.Event()

    async def run(self) -> None:
        config = VibeConfig.load().model_copy(deep=True)
        config.session_logging = SessionLoggingConfig(
            save_dir=str(self.session_root),
            session_prefix=f"room-{self.profile}",
            enabled=True,
        )
        config.disabled_tools = list(
            dict.fromkeys([*config.disabled_tools, *self.disabled_tools])
        )
        agent_loop = AgentLoop(
            LegacyConfigOrchestrator(config),
            agent_name=self.profile,
            max_turns=MAX_TURNS,
            max_price=MAX_PRICE_DOLLARS,
            max_session_tokens=MAX_TOKENS,
            enable_streaming=True,
            defer_heavy_init=True,
            headless=True,
            launch_context=build_launch_context(
                agent_entrypoint="programmatic",
                agent_version=__version__,
                client_name="vibe_agent_room_worker",
                client_version=__version__,
            ),
            hook_config_result=load_hooks_from_fs(),
            force_bypass_tool_permissions=self.auto_approve,
        )
        self.agent_loop = agent_loop
        agent_loop.set_approval_callback(self._approval_callback)
        agent_loop.set_user_input_callback(self._question_callback)
        if self.profile == BuiltinAgentName.ORCHESTRATOR:
            agent_loop.enable_interactive_surface_capabilities()
            agent_loop.set_agent_management_port(self.management)
            agent_loop.set_cli_control_port(RemoteControlPort(self))
            await agent_loop.set_tool_permission("manage_agents", ToolPermission.ALWAYS)
            await agent_loop.set_tool_permission("control_cli", ToolPermission.ALWAYS)
            await agent_loop.inject_user_context(
                """
You are the Agent Room orchestrator. Every worker has its own Git worktree,
branch, persistent process, chat, and approval channel. Delegate implementation
with manage_agents; never ask two workers to edit the same files. Use
control_cli command /cancel AGENT_ID to cancel a turn, /stop AGENT_ID to stop a
worker, or /merge AGENT_ID only after that worker has committed and stopped.
The host validates the merge in a temporary worktree before touching the
integration branch. Your own worktree is for coordination, not implementation.
""".strip()
            )

        resumed = False
        resume_error: str | None = None
        if self.resume_session_id:
            try:
                session_path = SessionLoader.find_session_by_id(
                    self.resume_session_id, config.session_logging
                )
                if session_path is None:
                    raise ValueError("Saved Vibe session was not found")
                loaded_messages, metadata = SessionLoader.load_session(session_path)
                agent_loop.messages.extend(loaded_messages)
                loaded_session_id = metadata.get("session_id", self.resume_session_id)
                agent_loop.session_id = loaded_session_id
                agent_loop.parent_session_id = metadata.get("parent_session_id")
                agent_loop.session_logger.resume_existing_session(
                    loaded_session_id, session_path
                )
                await agent_loop.hydrate_experiments_from_session()
                resumed = True
            except (OSError, ValueError) as error:
                resume_error = self._safe_error(error)

        model = agent_loop.config.get_active_model()
        self.emit({
            "type": "ready",
            "session_id": agent_loop.session_id,
            "parent_session_id": agent_loop.parent_session_id,
            "profile": self.profile,
            "model": model.alias,
            "context_limit": model.auto_compact_threshold,
            "resume_requested": bool(self.resume_session_id),
            "resumed": resumed,
            "resume_error": resume_error,
        })
        input_task = asyncio.create_task(self._input_loop(), name="room-worker-input")
        prompt_task = asyncio.create_task(
            self._prompt_worker(), name="room-worker-prompts"
        )
        await self.shutdown_event.wait()
        await self.prompt_queue.put(None)
        if self.current_turn is not None:
            self.current_turn.cancel()
        for task in (input_task, prompt_task):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        for future in self.pending_approvals.values():
            if not future.done():
                future.set_result((ApprovalResponse.NO, "Agent stopped"))
        for future in self.pending_questions.values():
            if not future.done():
                future.set_result(AskUserQuestionResult(cancelled=True, answers=[]))
        with suppress(Exception):
            await agent_loop.aclose()
        with suppress(Exception):
            await agent_loop.telemetry_client.aclose()

    async def _input_loop(self) -> None:  # noqa: PLR0912
        while not self.shutdown_event.is_set():
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                self.shutdown_event.set()
                return
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            message_type = payload.get("type")
            if message_type == "prompt":
                try:
                    self.prompt_queue.put_nowait(payload)
                except asyncio.QueueFull:
                    self.emit({
                        "type": "prompt_failed",
                        "message_id": payload.get("message_id"),
                        "error": "Worker queue is full",
                    })
            elif message_type == "cancel":
                if self.current_turn is not None:
                    self.current_turn.cancel()
            elif message_type == "shutdown":
                self.shutdown_event.set()
            elif message_type == "approval_response":
                self._resolve_approval(payload)
            elif message_type == "question_response":
                self._resolve_question(payload)
            elif message_type == "remote_response":
                request_id = str(payload.get("request_id") or "")
                future = self.pending_requests.pop(request_id, None)
                if future is not None and not future.done():
                    if payload.get("ok"):
                        future.set_result(payload.get("result"))
                    else:
                        future.set_exception(
                            ValueError(
                                str(payload.get("error") or "Remote call failed")
                            )
                        )
            elif message_type == "management_snapshot":
                self.management.apply_snapshot(payload)

    async def _prompt_worker(self) -> None:
        while True:
            prompt = await self.prompt_queue.get()
            if prompt is None:
                return
            try:
                self.current_turn = asyncio.create_task(
                    self._run_turn(prompt), name=f"room-turn-{prompt.get('message_id')}"
                )
                await self.current_turn
            except asyncio.CancelledError:
                if self.shutdown_event.is_set():
                    return
            finally:
                self.current_turn = None
                self.prompt_queue.task_done()

    async def _run_turn(self, prompt: dict[str, Any]) -> None:
        assert self.agent_loop is not None
        message_id = str(prompt.get("message_id") or uuid4().hex)
        content = str(prompt.get("content") or "").strip()
        raw_images = prompt.get("images")
        images = (
            [ImageAttachment.model_validate(item) for item in raw_images]
            if isinstance(raw_images, list)
            else []
        )
        self.emit({
            "type": "state",
            "state": "running",
            "activity": "Thinking",
            "message_id": message_id,
            "queued_messages": self.prompt_queue.qsize(),
        })
        response = ""
        try:
            async with aclosing(
                self.agent_loop.act(content, images=images or None)
            ) as events:
                async for event in events:
                    if isinstance(event, AssistantEvent) and event.content:
                        response += event.content
                        self.emit({
                            "type": "assistant_delta",
                            "message_id": message_id,
                            "content": event.content,
                        })
                    elif isinstance(event, ToolCallEvent):
                        self.emit({
                            "type": "tool_started",
                            "message_id": message_id,
                            "tool_call_id": event.tool_call_id,
                            "tool_name": event.tool_name,
                        })
                    elif isinstance(event, ToolResultEvent):
                        self.emit({
                            "type": "tool_finished",
                            "message_id": message_id,
                            "tool_call_id": event.tool_call_id,
                            "error": event.error,
                        })
            self.emit({
                "type": "assistant_final",
                "message_id": message_id,
                "content": response.strip(),
            })
            self._emit_usage()
            self.emit({
                "type": "state",
                "state": "idle",
                "activity": None,
                "message_id": message_id,
                "queued_messages": self.prompt_queue.qsize(),
            })
        except asyncio.CancelledError:
            self.emit({
                "type": "state",
                "state": "idle",
                "activity": None,
                "message_id": message_id,
                "cancelled": True,
                "queued_messages": self.prompt_queue.qsize(),
            })
            raise
        except Exception as error:
            self._emit_usage()
            self.emit({
                "type": "state",
                "state": "failed",
                "activity": "Turn failed",
                "message_id": message_id,
                "error": self._safe_error(error),
                "queued_messages": self.prompt_queue.qsize(),
            })

    async def _approval_callback(
        self,
        tool_name: str,
        args: BaseModel,
        tool_call_id: str,
        required_permissions: list[RequiredPermission] | None,
    ) -> tuple[ApprovalResponse, str | None]:
        request_id = f"approval-{uuid4().hex}"
        future = asyncio.get_running_loop().create_future()
        self.pending_approvals[request_id] = future
        self.emit({
            "type": "approval_requested",
            "request_id": request_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": self._sanitize(args.model_dump(mode="json")),
            "permissions": [
                {
                    "label": item.label,
                    "scope": item.scope.value,
                    "pattern": item.invocation_pattern,
                }
                for item in required_permissions or []
            ],
        })
        try:
            return await future
        finally:
            self.pending_approvals.pop(request_id, None)

    async def _question_callback(self, args: BaseModel) -> BaseModel:
        if not isinstance(args, AskUserQuestionArgs):
            raise ValueError("Unsupported interactive question")
        request_id = f"question-{uuid4().hex}"
        future = asyncio.get_running_loop().create_future()
        self.pending_questions[request_id] = future
        self.emit({
            "type": "question_requested",
            "request_id": request_id,
            **args.model_dump(mode="json"),
        })
        try:
            return await future
        finally:
            self.pending_questions.pop(request_id, None)

    async def remote_request(self, operation: str, payload: dict[str, Any]) -> Any:
        request_id = f"remote-{uuid4().hex}"
        future = asyncio.get_running_loop().create_future()
        self.pending_requests[request_id] = future
        self.emit({
            "type": "remote_request",
            "request_id": request_id,
            "operation": operation,
            "payload": payload,
        })
        try:
            return await future
        finally:
            self.pending_requests.pop(request_id, None)

    def _resolve_approval(self, payload: dict[str, Any]) -> None:
        request_id = str(payload.get("request_id") or "")
        future = self.pending_approvals.get(request_id)
        if future is None or future.done():
            return
        if payload.get("decision") == "approve_once":
            future.set_result((ApprovalResponse.YES, None))
        else:
            feedback = str(payload.get("feedback") or "User denied this tool call")
            future.set_result((ApprovalResponse.NO, feedback))

    def _resolve_question(self, payload: dict[str, Any]) -> None:
        request_id = str(payload.get("request_id") or "")
        future = self.pending_questions.get(request_id)
        if future is None or future.done():
            return
        raw_answers = payload.get("answers")
        if not isinstance(raw_answers, list):
            future.set_result(AskUserQuestionResult(cancelled=True, answers=[]))
            return
        answers = [Answer.model_validate(item) for item in raw_answers]
        future.set_result(AskUserQuestionResult(cancelled=False, answers=answers))

    def _emit_usage(self) -> None:
        assert self.agent_loop is not None
        stats = self.agent_loop.stats
        model = self.agent_loop.config.get_active_model()
        self.emit({
            "type": "usage",
            "turns_used": stats.steps,
            "prompt_tokens": stats.session_prompt_tokens,
            "completion_tokens": stats.session_completion_tokens,
            "context_tokens": stats.context_tokens,
            "context_limit": model.auto_compact_threshold,
            "estimated_cost_usd": stats.session_cost,
            "model": model.alias,
            "session_id": self.agent_loop.session_id,
            "parent_session_id": self.agent_loop.parent_session_id,
        })

    @staticmethod
    def emit(payload: dict[str, Any]) -> None:
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    @classmethod
    def _sanitize(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: (
                    "[redacted]"
                    if any(
                        marker in key.lower()
                        for marker in ("token", "secret", "password", "api_key")
                    )
                    else cls._sanitize(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._sanitize(item) for item in value[:50]]
        if isinstance(value, str):
            return value[:2_000]
        return value

    @staticmethod
    def _safe_error(error: Exception) -> str:
        return (str(error).strip() or type(error).__name__)[:1_000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--disable-tool", action="append", default=[])
    parser.add_argument("--resume-session")
    parser.add_argument("--auto-approve", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    init_harness_files_manager("user", "project")
    bridge = WorkerBridge(
        args.profile,
        args.session_root,
        disabled_tools=tuple(args.disable_tool),
        resume_session_id=args.resume_session,
        auto_approve=args.auto_approve,
    )
    asyncio.run(bridge.run())


if __name__ == "__main__":
    main()
