from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing, suppress
from dataclasses import dataclass, field
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from vibe.core.agents.events import (
    MAX_MANAGED_AGENT_ACTIVITY_CHARS,
    MAX_MANAGED_AGENT_ERROR_CHARS,
    MAX_MANAGED_AGENT_ID_CHARS,
    MAX_MANAGED_AGENT_PROFILE_CHARS,
    MAX_MANAGED_AGENT_RESPONSE_CHARS,
    MAX_MANAGED_AGENT_TASK_CHARS,
    ManagedAgentCallbackContext,
    ManagedAgentLifecycleEvent,
    _reset_managed_agent_callback_context,
    _set_managed_agent_callback_context,
)
from vibe.core.agents.models import (
    AgentType,
    BuiltinAgentName,
    ManagedAgentSnapshot,
    ManagedAgentState,
)
from vibe.core.config import AnyVibeConfig, SessionLoggingConfig
from vibe.core.config.orchestrator_legacy import LegacyConfigOrchestrator
from vibe.core.types import AssistantEvent, LLMUsage, ToolCallEvent

if TYPE_CHECKING:
    from vibe.core.agents.manager import AgentManager
    from vibe.core.hooks.models import HookConfigResult
    from vibe.core.telemetry.types import LaunchContext
    from vibe.core.tools.permissions import PermissionStore, RequiredPermission
    from vibe.core.types import (
        ApprovalCallback,
        ApprovalResponse,
        BaseEvent,
        UserInputCallback,
    )

MAX_MANAGED_AGENTS = 8
MAX_QUEUED_MESSAGES = 20
MAX_STOPPED_HISTORY = 32
MANAGED_AGENT_EVENT_QUEUE_SIZE = 64


class ManagedSessionLogger(Protocol):
    def reset_session(
        self, session_id: str, *, parent_session_id: str | None = None
    ) -> None: ...


class ManagedTelemetryClient(Protocol):
    async def aclose(self) -> None: ...


class ManagedAgentStats(Protocol):
    steps: int
    session_prompt_tokens: int
    session_completion_tokens: int
    context_tokens: int

    @property
    def session_cost(self) -> float: ...


class ManagedAgentLoop(Protocol):
    session_id: str
    parent_session_id: str | None

    @property
    def config(self) -> AnyVibeConfig: ...

    @property
    def stats(self) -> ManagedAgentStats: ...

    @property
    def session_logger(self) -> ManagedSessionLogger: ...

    @property
    def telemetry_client(self) -> ManagedTelemetryClient: ...

    def act(self, msg: str, **kwargs: object) -> AsyncGenerator[BaseEvent, None]: ...

    def set_approval_callback(self, callback: ApprovalCallback) -> None: ...

    def set_user_input_callback(self, callback: UserInputCallback) -> None: ...

    async def wait_until_ready(self) -> None: ...

    async def aclose(self) -> None: ...


type AgentLoopFactory = Callable[
    [str, AgentType, SessionLoggingConfig], ManagedAgentLoop
]


@dataclass
class _ManagedAgent:
    agent_id: str
    profile: str
    profile_display_name: str
    parent_session_id: str
    task: str
    loop: ManagedAgentLoop
    queue: asyncio.Queue[str]
    runner: asyncio.Task[None] | None = None
    state: ManagedAgentState = ManagedAgentState.STARTING
    current_activity: str | None = None
    last_response: str = ""
    error: str | None = None
    sequence: int = 0
    terminal_emitted: bool = False
    closed: bool = False
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    attention_base_state: ManagedAgentState | None = None
    attention_base_activity: str | None = None
    pending_attention: dict[int, str] = field(default_factory=dict)
    next_attention_id: int = 1
    stop_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class _ManagedEventSubscriber:
    def __init__(self) -> None:
        self._pending: dict[str, ManagedAgentLifecycleEvent] = {}
        self._wake = asyncio.Event()
        self._closed = False

    def offer(self, event: ManagedAgentLifecycleEvent) -> bool:
        if (
            event.agent_id not in self._pending
            and len(self._pending) >= MANAGED_AGENT_EVENT_QUEUE_SIZE
        ):
            return False
        self._pending[event.agent_id] = event
        self._wake.set()
        return True

    def replace(self, events: list[ManagedAgentLifecycleEvent]) -> None:
        self._pending = {event.agent_id: event for event in events}
        self._wake.set()

    def close(self) -> None:
        self._closed = True
        self._wake.set()

    async def get(self) -> ManagedAgentLifecycleEvent | None:
        while not self._pending:
            if self._closed:
                return None
            self._wake.clear()
            await self._wake.wait()
        agent_id = next(iter(self._pending))
        return self._pending.pop(agent_id)


class AgentSupervisor:
    def __init__(
        self,
        *,
        base_config_getter: Callable[[], AnyVibeConfig],
        agent_manager: AgentManager,
        permission_store: PermissionStore,
        approval_callback_getter: Callable[[], ApprovalCallback | None],
        user_input_callback_getter: Callable[[], UserInputCallback | None],
        parent_session_id_getter: Callable[[], str],
        session_dir_getter: Callable[[], Path | None],
        launch_context: LaunchContext | None,
        hook_config_result: HookConfigResult | None,
        loop_factory: AgentLoopFactory | None = None,
    ) -> None:
        self._base_config_getter = base_config_getter
        self._agent_manager = agent_manager
        self._permission_store = permission_store
        self._approval_callback_getter = approval_callback_getter
        self._user_input_callback_getter = user_input_callback_getter
        self._parent_session_id_getter = parent_session_id_getter
        self._session_dir_getter = session_dir_getter
        self._launch_context = launch_context
        self._hook_config_result = hook_config_result
        self._loop_factory = loop_factory or self._create_loop
        self._agents: dict[str, _ManagedAgent] = {}
        self._stopped_ids: deque[str] = deque()
        self._subscribers: set[_ManagedEventSubscriber] = set()
        self._next_agent_sequence = 1
        self._closed = False

    async def start(
        self, profile: str, task: str, *, name: str | None = None
    ) -> ManagedAgentSnapshot:
        if self._closed:
            raise ValueError("Managed agent supervisor is closed")
        normalized_task = self._bounded(
            self._nonblank(task, "task"), MAX_MANAGED_AGENT_TASK_CHARS
        )
        profile_details = self._agent_manager.get_agent(profile)
        if profile_details.name == BuiltinAgentName.ORCHESTRATOR:
            raise ValueError("An orchestrator cannot launch another orchestrator")
        active_count = sum(
            agent.state is not ManagedAgentState.STOPPED
            for agent in self._agents.values()
        )
        if active_count >= MAX_MANAGED_AGENTS:
            raise ValueError(f"At most {MAX_MANAGED_AGENTS} managed agents can run")

        parent_session_id = self._parent_session_id_getter()
        agent_id = self._next_id(name or profile_details.name)
        loop = self._loop_factory(
            profile_details.name,
            profile_details.agent_type,
            self._session_logging(profile_details.name),
        )
        try:
            loop.parent_session_id = parent_session_id
            loop.session_logger.reset_session(
                loop.session_id, parent_session_id=parent_session_id
            )
            managed = _ManagedAgent(
                agent_id=agent_id,
                profile=self._bounded(
                    profile_details.name, MAX_MANAGED_AGENT_PROFILE_CHARS
                ),
                profile_display_name=(
                    self._bounded(
                        profile_details.display_name.strip() or profile_details.name,
                        MAX_MANAGED_AGENT_PROFILE_CHARS,
                    )
                ),
                parent_session_id=parent_session_id,
                task=normalized_task,
                loop=loop,
                queue=asyncio.Queue(maxsize=MAX_QUEUED_MESSAGES),
            )
            self._agents[agent_id] = managed
            self._set_tracked_callbacks(managed)
            managed.queue.put_nowait(normalized_task)
            self._emit(managed)
            managed.runner = asyncio.create_task(
                self._run(managed), name=f"vibe-managed-agent-{agent_id}"
            )
            return self._snapshot(managed)
        except BaseException:
            self._agents.pop(agent_id, None)
            await self._close_raw_loop(loop)
            raise

    def list(self) -> tuple[ManagedAgentSnapshot, ...]:
        return tuple(self._snapshot(agent) for agent in self._agents.values())

    def available_profiles(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self._agent_manager.available_agents
            if name != BuiltinAgentName.ORCHESTRATOR
        )

    async def message(self, agent_id: str, message: str) -> ManagedAgentSnapshot:
        agent = self._get(agent_id)
        normalized_message = self._bounded(
            self._nonblank(message, "message"), MAX_MANAGED_AGENT_TASK_CHARS
        )
        if agent.state is ManagedAgentState.STOPPED:
            raise ValueError(f"Managed agent '{agent_id}' is stopped")
        try:
            agent.queue.put_nowait(normalized_message)
        except asyncio.QueueFull as exc:
            raise ValueError(
                f"Managed agent '{agent_id}' has too many queued messages"
            ) from exc
        self._emit(agent)
        return self._snapshot(agent)

    def output(self, agent_id: str) -> ManagedAgentSnapshot:
        return self._snapshot(self._get(agent_id))

    async def stop(self, agent_id: str) -> ManagedAgentSnapshot:
        agent = self._get(agent_id)
        async with agent.stop_lock:
            if not agent.terminal_emitted:
                self._drain_queue(agent.queue)
                self._transition(
                    agent, ManagedAgentState.STOPPED, current_activity=None
                )
                self._remember_stopped(agent)
            if agent.runner is not None and not agent.runner.done():
                agent.runner.cancel()
                with suppress(asyncio.CancelledError):
                    await agent.runner
            await self._close_loop(agent)
        return self._snapshot(agent)

    async def stop_all(self) -> None:
        for agent_id in tuple(self._agents):
            agent = self._agents.get(agent_id)
            if agent is None or agent.state is ManagedAgentState.STOPPED:
                continue
            with suppress(Exception):
                await self.stop(agent_id)

    async def stop_for_session_change(self) -> None:
        await self.stop_all()
        self._agents.clear()
        self._stopped_ids.clear()

    async def subscribe_events(
        self,
    ) -> AsyncGenerator[ManagedAgentLifecycleEvent, None]:
        if self._closed:
            return
        subscriber = _ManagedEventSubscriber()
        self._subscribers.add(subscriber)
        for agent in self._agents.values():
            subscriber.offer(self._event(agent, advance=False))
        try:
            while (event := await subscriber.get()) is not None:
                yield event
        finally:
            self._subscribers.discard(subscriber)

    async def aclose(self) -> None:
        if self._closed:
            return
        await self.stop_all()
        self._closed = True
        subscribers = tuple(self._subscribers)
        self._subscribers.clear()
        for subscriber in subscribers:
            subscriber.close()

    async def _run(self, agent: _ManagedAgent) -> None:
        while True:
            prompt = await agent.queue.get()
            try:
                self._transition(
                    agent,
                    ManagedAgentState.RUNNING,
                    current_activity="Working",
                    error=None,
                )
                agent.last_response = ""
                async with aclosing(agent.loop.act(prompt)) as events:
                    async for event in events:
                        if isinstance(event, AssistantEvent) and event.content:
                            agent.last_response = self._bounded_tail(
                                agent.last_response + event.content,
                                MAX_MANAGED_AGENT_RESPONSE_CHARS,
                            )
                        elif isinstance(event, ToolCallEvent):
                            self._transition(
                                agent,
                                ManagedAgentState.WORKING,
                                current_activity=self._bounded(
                                    f"Running {event.tool_name}",
                                    MAX_MANAGED_AGENT_ACTIVITY_CHARS,
                                ),
                            )
                self._transition(
                    agent, ManagedAgentState.IDLE, current_activity=None, error=None
                )
            except asyncio.CancelledError:
                self._transition(
                    agent, ManagedAgentState.STOPPED, current_activity=None
                )
                raise
            except Exception as exc:
                self._transition(
                    agent,
                    ManagedAgentState.FAILED,
                    current_activity=None,
                    error=self._bounded(
                        str(exc) or type(exc).__name__, MAX_MANAGED_AGENT_ERROR_CHARS
                    ),
                )
            finally:
                agent.queue.task_done()

    def _set_tracked_callbacks(self, agent: _ManagedAgent) -> None:
        if self._approval_callback_getter() is not None:

            async def tracked_approval_callback(
                tool_name: str,
                callback_args: BaseModel,
                tool_call_id: str,
                required_permissions: list[RequiredPermission] | None,
            ) -> tuple[ApprovalResponse, str | None]:
                callback = self._approval_callback_getter()
                if callback is None:
                    raise RuntimeError("Approval callback is no longer available")
                attention_id = self._begin_attention(
                    agent, f"Approval needed for {tool_name}"
                )
                token = _set_managed_agent_callback_context(
                    ManagedAgentCallbackContext(
                        agent_id=agent.agent_id, profile=agent.profile
                    )
                )
                try:
                    return await callback(
                        tool_name, callback_args, tool_call_id, required_permissions
                    )
                finally:
                    _reset_managed_agent_callback_context(token)
                    self._end_attention(agent, attention_id)

            agent.loop.set_approval_callback(tracked_approval_callback)

        if self._user_input_callback_getter() is not None:

            async def tracked_user_input_callback(
                callback_args: BaseModel,
            ) -> BaseModel:
                callback = self._user_input_callback_getter()
                if callback is None:
                    raise RuntimeError("User input callback is no longer available")
                attention_id = self._begin_attention(agent, "Waiting for user input")
                token = _set_managed_agent_callback_context(
                    ManagedAgentCallbackContext(
                        agent_id=agent.agent_id, profile=agent.profile
                    )
                )
                try:
                    return await callback(callback_args)
                finally:
                    _reset_managed_agent_callback_context(token)
                    self._end_attention(agent, attention_id)

            agent.loop.set_user_input_callback(tracked_user_input_callback)

    def _begin_attention(self, agent: _ManagedAgent, activity: str) -> int | None:
        if agent.terminal_emitted:
            return None
        if not agent.pending_attention:
            agent.attention_base_state = agent.state
            agent.attention_base_activity = agent.current_activity
        attention_id = agent.next_attention_id
        agent.next_attention_id += 1
        bounded_activity = self._bounded(activity, MAX_MANAGED_AGENT_ACTIVITY_CHARS)
        agent.pending_attention[attention_id] = bounded_activity
        self._transition(
            agent,
            ManagedAgentState.ATTENTION,
            current_activity=self._active_attention_activity(agent),
        )
        return attention_id

    def _end_attention(self, agent: _ManagedAgent, attention_id: int | None) -> None:
        if attention_id is None:
            return
        agent.pending_attention.pop(attention_id, None)
        if agent.terminal_emitted:
            if not agent.pending_attention:
                agent.attention_base_state = None
                agent.attention_base_activity = None
            return
        if agent.pending_attention:
            self._transition(
                agent,
                ManagedAgentState.ATTENTION,
                current_activity=self._active_attention_activity(agent),
            )
            return
        base_state = agent.attention_base_state or ManagedAgentState.RUNNING
        base_activity = agent.attention_base_activity
        agent.attention_base_state = None
        agent.attention_base_activity = None
        self._transition(agent, base_state, current_activity=base_activity)

    def _transition(
        self,
        agent: _ManagedAgent,
        state: ManagedAgentState,
        *,
        current_activity: str | None,
        error: str | None | object = ...,
    ) -> None:
        if agent.terminal_emitted:
            return
        if (
            agent.pending_attention
            and state is not ManagedAgentState.ATTENTION
            and state is not ManagedAgentState.STOPPED
        ):
            agent.attention_base_state = state
            agent.attention_base_activity = current_activity
            state = ManagedAgentState.ATTENTION
            current_activity = self._active_attention_activity(agent)
        agent.state = state
        agent.current_activity = current_activity
        if error is not ...:
            agent.error = error if isinstance(error, str) else None
        if state is ManagedAgentState.STOPPED:
            agent.terminal_emitted = True
        self._emit(agent)

    @staticmethod
    def _active_attention_activity(agent: _ManagedAgent) -> str:
        return next(iter(agent.pending_attention.values()))

    def _emit(self, agent: _ManagedAgent) -> None:
        agent.updated_at = time.time()
        event = self._event(agent, advance=True)
        for subscriber in tuple(self._subscribers):
            if subscriber.offer(event):
                continue
            subscriber.replace([
                self._event(current, advance=False) for current in self._agents.values()
            ])

    def _event(
        self, agent: _ManagedAgent, *, advance: bool
    ) -> ManagedAgentLifecycleEvent:
        if advance:
            agent.sequence += 1
        return ManagedAgentLifecycleEvent(
            sequence=agent.sequence,
            agent_id=agent.agent_id,
            profile=agent.profile,
            agent_display_name=agent.profile_display_name,
            parent_session_id=agent.parent_session_id,
            child_session_id=agent.loop.session_id,
            task=agent.task,
            state=agent.state,
            current_activity=agent.current_activity,
            queued_messages=agent.queue.qsize(),
            error=agent.error,
            last_response=agent.last_response,
            usage=LLMUsage(
                prompt_tokens=agent.loop.stats.session_prompt_tokens,
                completion_tokens=agent.loop.stats.session_completion_tokens,
            ),
        )

    def _create_loop(
        self, profile: str, agent_type: AgentType, session_logging: SessionLoggingConfig
    ) -> ManagedAgentLoop:
        from vibe.core.agent_loop import AgentLoop

        base_config = self._base_config_getter().model_copy(deep=True)
        base_config.session_logging = session_logging
        base_config.enable_agent_management = False
        base_config.enable_cli_control = False
        base_config.enable_orchestrator_controls = False
        base_config.disabled_tools = list(
            dict.fromkeys([
                *base_config.disabled_tools,
                "control_cli",
                "manage_agents",
                "task",
            ])
        )
        return AgentLoop(
            config_orchestrator=LegacyConfigOrchestrator(base_config),
            agent_name=profile,
            launch_context=self._launch_context,
            is_subagent=agent_type is AgentType.SUBAGENT,
            defer_heavy_init=True,
            permission_store=self._permission_store,
            hook_config_result=self._hook_config_result,
        )

    def _session_logging(self, profile: str) -> SessionLoggingConfig:
        session_dir = self._session_dir_getter()
        return SessionLoggingConfig(
            save_dir=str(session_dir / "agents") if session_dir else "",
            session_prefix=profile,
            enabled=session_dir is not None,
        )

    def _next_id(self, requested: str) -> str:
        base = (re.sub(r"[^a-z0-9]+", "-", requested.lower()).strip("-") or "agent")[
            : MAX_MANAGED_AGENT_ID_CHARS - 12
        ]
        sequence = self._next_agent_sequence
        self._next_agent_sequence += 1
        return f"{base}-{sequence}"

    def _get(self, agent_id: str) -> _ManagedAgent:
        try:
            return self._agents[agent_id]
        except KeyError as exc:
            raise ValueError(f"Unknown managed agent: {agent_id}") from exc

    def _remember_stopped(self, agent: _ManagedAgent) -> None:
        self._stopped_ids.append(agent.agent_id)
        while len(self._stopped_ids) > MAX_STOPPED_HISTORY:
            expired_id = self._stopped_ids.popleft()
            expired = self._agents.get(expired_id)
            if expired is not None and expired.state is ManagedAgentState.STOPPED:
                del self._agents[expired_id]

    async def _close_loop(self, agent: _ManagedAgent) -> None:
        if agent.closed:
            return
        agent.closed = True
        await self._close_raw_loop(agent.loop)

    @staticmethod
    async def _close_raw_loop(loop: ManagedAgentLoop) -> None:
        with suppress(Exception):
            await loop.wait_until_ready()
        with suppress(Exception):
            await loop.aclose()
        with suppress(Exception):
            await loop.telemetry_client.aclose()

    @staticmethod
    def _snapshot(agent: _ManagedAgent) -> ManagedAgentSnapshot:
        stats = agent.loop.stats
        model = None
        context_limit = None
        with suppress(ValueError):
            active_model = agent.loop.config.get_active_model()
            model = active_model.alias
            context_limit = active_model.auto_compact_threshold
        return ManagedAgentSnapshot(
            agent_id=agent.agent_id,
            child_session_id=agent.loop.session_id,
            profile=agent.profile,
            state=agent.state,
            task=agent.task,
            current_activity=agent.current_activity,
            last_response=agent.last_response,
            error=agent.error,
            queued_messages=agent.queue.qsize(),
            started_at=agent.started_at,
            updated_at=agent.updated_at,
            turns_used=stats.steps,
            prompt_tokens=stats.session_prompt_tokens,
            completion_tokens=stats.session_completion_tokens,
            context_tokens=stats.context_tokens,
            context_limit=context_limit,
            estimated_cost_usd=stats.session_cost,
            model=model,
        )

    @staticmethod
    def _nonblank(value: str, field_name: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must not be blank")
        return normalized

    @staticmethod
    def _bounded(value: str, limit: int) -> str:
        return value[:limit]

    @staticmethod
    def _bounded_tail(value: str, limit: int) -> str:
        return value[-limit:]

    @staticmethod
    def _drain_queue(queue: asyncio.Queue[str]) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            queue.task_done()
