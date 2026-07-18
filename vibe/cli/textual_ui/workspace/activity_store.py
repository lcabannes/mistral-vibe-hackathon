from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
import time
from typing import Protocol, cast

from pydantic import BaseModel, ValidationError

from vibe.cli.textual_ui.workspace.models import (
    AgentActivity,
    AgentActivitySnapshot,
    AgentRunState,
    _TaskCall,
    _TaskOutcome,
)
from vibe.core.agents.events import ManagedAgentLifecycleEvent
from vibe.core.agents.models import ManagedAgentState
from vibe.core.types import (
    BaseEvent,
    LLMUsage,
    SubagentLifecycleEvent,
    ToolCallEvent,
    ToolResultEvent,
)

type AgentActivityListener = Callable[[AgentActivitySnapshot], None]


class _ManagedAgentEventPayload(Protocol):
    task: str
    last_response: str
    error: str | None
    usage: LLMUsage | None


def _project_model[T: BaseModel](
    model: BaseModel | None, projection: type[T]
) -> T | None:
    if model is None:
        return None
    try:
        return projection.model_validate(model.model_dump())
    except ValidationError:
        return None


class AgentActivityStore:
    def __init__(
        self,
        session_id: str,
        max_activities: int = 50,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_activities < 1:
            raise ValueError("max_activities must be at least 1")
        self._session_id = session_id
        self._max_activities = max_activities
        self._clock = clock
        self._activities: OrderedDict[str, AgentActivity] = OrderedDict()
        self._primary: AgentActivity | None = None
        self._listeners: list[AgentActivityListener] = []

    @property
    def snapshot(self) -> AgentActivitySnapshot:
        primary = (self._primary,) if self._primary is not None else ()
        return AgentActivitySnapshot(
            session_id=self._session_id,
            activities=primary + tuple(self._activities.values()),
        )

    def add_listener(self, listener: AgentActivityListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: AgentActivityListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def update_primary(
        self,
        agent_name: str,
        agent_display_name: str,
        state: AgentRunState,
        current_activity: str | None = None,
    ) -> bool:
        if state not in {
            AgentRunState.RUNNING,
            AgentRunState.WORKING,
            AgentRunState.ATTENTION,
            AgentRunState.IDLE,
        }:
            raise ValueError(f"Unsupported primary agent state: {state}")

        current = self._primary
        if current is not None and (
            current.agent_name,
            current.agent_display_name,
            current.state,
            current.current_activity,
        ) == (agent_name, agent_display_name, state, current_activity):
            return False

        now = self._clock()
        started_at = (
            now
            if current is None
            or (current.state is AgentRunState.IDLE and state is not AgentRunState.IDLE)
            else current.started_at
        )
        self._primary = AgentActivity(
            tool_call_id=f"primary:{self._session_id}",
            parent_session_id=self._session_id,
            agent_name=agent_name,
            agent_display_name=agent_display_name,
            task="Current conversation",
            state=state,
            started_at=started_at,
            updated_at=now,
            current_activity=current_activity,
            is_primary=True,
        )
        self._notify()
        return True

    def apply(self, event: BaseEvent) -> bool:
        match event:
            case ManagedAgentLifecycleEvent():
                changed = self._apply_managed_lifecycle(event)
            case SubagentLifecycleEvent():
                changed = self._apply_lifecycle(event)
            case ToolCallEvent() if event.tool_name == "task":
                changed = self._apply_tool_call(event)
            case ToolResultEvent() if event.tool_name == "task":
                changed = self._apply_tool_result(event)
            case _:
                return False

        if not changed:
            return False
        self._trim()
        self._notify()
        return True

    @staticmethod
    def _task_key(tool_call_id: str) -> str:
        return f"task:{tool_call_id}"

    @staticmethod
    def _managed_key(agent_id: str) -> str:
        return f"managed:{agent_id}"

    @staticmethod
    def _managed_state(state: ManagedAgentState) -> AgentRunState:
        if state is ManagedAgentState.STARTING:
            return AgentRunState.REQUESTED
        return AgentRunState(state.value)

    def _apply_managed_lifecycle(self, event: ManagedAgentLifecycleEvent) -> bool:
        if event.parent_session_id != self._session_id:
            return False
        key = self._managed_key(event.agent_id)
        current = self._activities.get(key)
        if current is not None:
            if current.state is AgentRunState.STOPPED:
                return False
            if (
                current.event_sequence is not None
                and event.sequence <= current.event_sequence
            ):
                return False

        now = self._clock()
        payload = cast(_ManagedAgentEventPayload, event)
        activity = AgentActivity(
            tool_call_id=key,
            parent_session_id=event.parent_session_id,
            agent_name=event.profile,
            agent_display_name=event.agent_display_name,
            task=payload.task,
            state=self._managed_state(event.state),
            started_at=current.started_at if current is not None else now,
            updated_at=now,
            child_session_id=event.child_session_id,
            current_activity=event.current_activity,
            usage=payload.usage,
            managed_agent_id=event.agent_id,
            event_sequence=event.sequence,
            queued_messages=event.queued_messages,
            last_response=payload.last_response,
            error=payload.error,
        )
        self._activities[key] = activity
        return True

    def _notify(self) -> None:
        snapshot = self.snapshot
        for listener in tuple(self._listeners):
            listener(snapshot)

    def _apply_tool_call(self, event: ToolCallEvent) -> bool:
        key = self._task_key(event.tool_call_id)
        current = self._activities.get(key)
        args = _project_model(event.args, _TaskCall)
        now = self._clock()
        if current is None:
            agent_name = args.agent if args else ""
            self._activities[key] = AgentActivity(
                tool_call_id=event.tool_call_id,
                parent_session_id=self._session_id,
                agent_name=agent_name,
                agent_display_name=agent_name,
                task=args.task if args else "",
                state=AgentRunState.REQUESTED,
                started_at=now,
                updated_at=now,
            )
            return True
        if current.state.is_terminal or args is None:
            return False
        return self._update(
            current,
            agent_name=args.agent,
            agent_display_name=args.agent,
            task=args.task,
        )

    def _apply_lifecycle(self, event: SubagentLifecycleEvent) -> bool:
        state = AgentRunState(event.state.value)
        key = self._task_key(event.tool_call_id)
        current = self._activities.get(key)
        now = self._clock()
        if current is None:
            self._activities[key] = AgentActivity(
                tool_call_id=event.tool_call_id,
                parent_session_id=self._session_id,
                agent_name=event.agent_name,
                agent_display_name=event.agent_display_name,
                task=event.task,
                state=state,
                started_at=now,
                updated_at=now,
                child_session_id=event.child_session_id,
                current_activity=event.current_activity,
                usage=event.terminal_usage,
            )
            return True
        if current.state.is_terminal and not state.is_terminal:
            return False
        return self._update(
            current,
            agent_name=event.agent_name,
            agent_display_name=event.agent_display_name,
            task=event.task,
            state=state,
            child_session_id=event.child_session_id,
            current_activity=event.current_activity or current.current_activity,
            usage=event.terminal_usage or current.usage,
        )

    def _apply_tool_result(self, event: ToolResultEvent) -> bool:
        key = self._task_key(event.tool_call_id)
        current = self._activities.get(key)
        if current is None:
            now = self._clock()
            current = AgentActivity(
                tool_call_id=event.tool_call_id,
                parent_session_id=self._session_id,
                agent_name="",
                agent_display_name="",
                task="",
                state=AgentRunState.REQUESTED,
                started_at=now,
                updated_at=now,
            )
            self._activities[key] = current

        result = _project_model(event.result, _TaskOutcome)
        if event.cancelled or event.skipped:
            state = AgentRunState.CANCELLED
        elif event.error:
            state = AgentRunState.FAILED
        elif result is not None and result.completed:
            state = AgentRunState.COMPLETED
        elif current.state is AgentRunState.FAILED:
            state = AgentRunState.FAILED
        else:
            state = AgentRunState.CANCELLED

        return self._update(
            current,
            state=state,
            current_activity=event.error
            or event.skip_reason
            or current.current_activity,
            turns_used=result.turns_used if result else current.turns_used,
        )

    def _update(self, current: AgentActivity, **changes: object) -> bool:
        meaningful = {
            name: value
            for name, value in changes.items()
            if getattr(current, name) != value
        }
        if not meaningful:
            return False
        meaningful["updated_at"] = self._clock()
        key = (
            self._managed_key(current.managed_agent_id)
            if current.managed_agent_id is not None
            else self._task_key(current.tool_call_id)
        )
        self._activities[key] = current.model_copy(update=meaningful)
        return True

    def _trim(self) -> None:
        while len(self._activities) > self._max_activities:
            terminal_id = next(
                (
                    tool_call_id
                    for tool_call_id, activity in self._activities.items()
                    if activity.state.is_terminal
                ),
                None,
            )
            self._activities.pop(terminal_id or next(iter(self._activities)))
