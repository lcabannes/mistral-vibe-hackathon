from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Callable
from contextlib import suppress
from datetime import UTC, datetime
import hashlib
from uuid import uuid4

from vibe.core.logger import logger
from vibe.core.task_center.execution_port import (
    TaskExecutionDisposition,
    TaskExecutionPort,
    TaskExecutionRequest,
    TaskExecutionResult,
)
from vibe.core.task_center.models import (
    AppStartTrigger,
    IntervalTrigger,
    ManualTrigger,
    SessionStartTrigger,
    TaskDefinition,
    TaskEventKind,
    TaskRunRecord,
    TaskRunState,
    TaskSourceEvent,
    TaskTriggeredEvent,
)
from vibe.core.task_center.schedule import next_occurrence
from vibe.core.task_center.store import TaskStore

TASK_EVENT_QUEUE_SIZE = 64
SEEN_SOURCE_EVENT_LIMIT = 256

type Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class TaskScheduler:
    def __init__(
        self,
        store: TaskStore,
        *,
        execution_port: TaskExecutionPort | None = None,
        clock: Clock = _utc_now,
    ) -> None:
        self._store = store
        self._execution_port = execution_port
        self._clock = clock
        self._timer: asyncio.TimerHandle | None = None
        self._dispatch_task: asyncio.Task[None] | None = None
        self._trigger_lock = asyncio.Lock()
        self._subscribers: set[asyncio.Queue[TaskTriggeredEvent | None]] = set()
        self._seen_source_events: set[str] = set()
        self._seen_source_order: deque[str] = deque()
        self._started = False

    @property
    def is_running(self) -> bool:
        return self._started

    @property
    def next_deadline(self) -> datetime | None:
        deadlines = [
            task.next_run_at
            for task in self._store.snapshot
            if task.enabled and task.next_run_at is not None
        ]
        return min(deadlines) if deadlines else None

    async def start(self) -> None:
        if self._started:
            return
        await self._store.load()
        self._started = True
        self._store.add_listener(self._on_store_changed)
        await self._dispatch_due(self._now())
        self._reschedule_timer()

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._store.remove_listener(self._on_store_changed)
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        task = self._dispatch_task
        self._dispatch_task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        subscribers = tuple(self._subscribers)
        self._subscribers.clear()
        for queue in subscribers:
            self._offer(queue, None)

    async def events(self) -> AsyncGenerator[TaskTriggeredEvent, None]:
        queue: asyncio.Queue[TaskTriggeredEvent | None] = asyncio.Queue(
            maxsize=TASK_EVENT_QUEUE_SIZE
        )
        self._subscribers.add(queue)
        try:
            while (event := await queue.get()) is not None:
                yield event
        finally:
            self._subscribers.discard(queue)

    async def submit_event(
        self, event: TaskSourceEvent
    ) -> tuple[TaskTriggeredEvent, ...]:
        if not self._started:
            raise RuntimeError("Task scheduler is not running")
        source_key = f"{event.kind.value}:{event.event_id}"
        if source_key in self._seen_source_events:
            return ()
        self._remember_source_event(source_key)
        expected = {
            TaskEventKind.APP_START: AppStartTrigger,
            TaskEventKind.SESSION_START: SessionStartTrigger,
        }[event.kind]
        triggered: list[TaskTriggeredEvent] = []
        for task in self._store.snapshot:
            if not task.enabled or not isinstance(task.trigger, expected):
                continue
            instance_id = _stable_instance_id("event", source_key, task.task_id)
            result = await self._trigger_task(
                task,
                trigger_instance_id=instance_id,
                triggered_at=self._now(),
                scheduled_for=None,
                coalesced=False,
            )
            if result is not None:
                triggered.append(result)
        return tuple(triggered)

    async def trigger_manual(
        self, task_id: str, *, request_id: str | None = None
    ) -> TaskTriggeredEvent | None:
        if not self._started:
            raise RuntimeError("Task scheduler is not running")
        task = await self._store.get(task_id)
        if not isinstance(task.trigger, ManualTrigger):
            raise ValueError(f"Task '{task_id}' is not manually triggered")
        instance_id = _stable_instance_id(
            "manual", request_id or uuid4().hex, task.task_id
        )
        return await self._trigger_task(
            task,
            trigger_instance_id=instance_id,
            triggered_at=self._now(),
            scheduled_for=None,
            coalesced=False,
        )

    def _on_store_changed(self, _snapshot: tuple[TaskDefinition, ...]) -> None:
        if self._started:
            self._reschedule_timer()

    def _reschedule_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if not self._started or (deadline := self.next_deadline) is None:
            return
        delay = max(0.0, (deadline - self._now()).total_seconds())
        self._timer = asyncio.get_running_loop().call_later(
            delay, self._deadline_reached
        )

    def _deadline_reached(self) -> None:
        self._timer = None
        if not self._started:
            return
        if self._dispatch_task is not None and not self._dispatch_task.done():
            return
        self._dispatch_task = asyncio.create_task(
            self._run_deadline_dispatch(), name="vibe-task-center-deadline"
        )

    async def _run_deadline_dispatch(self) -> None:
        try:
            await self._dispatch_due(self._now())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Task Center deadline dispatch failed")
        finally:
            self._dispatch_task = None
            self._reschedule_timer()

    async def _dispatch_due(self, now: datetime) -> None:
        due = sorted(
            (
                task
                for task in self._store.snapshot
                if task.enabled
                and task.next_run_at is not None
                and task.next_run_at <= now
            ),
            key=lambda item: (item.next_run_at or now, item.task_id),
        )
        for task in due:
            scheduled_for = task.next_run_at
            assert scheduled_for is not None
            await self._trigger_task(
                task,
                trigger_instance_id=_stable_instance_id(
                    "schedule", task.task_id, scheduled_for.isoformat()
                ),
                triggered_at=now,
                scheduled_for=scheduled_for,
                coalesced=scheduled_for < now,
            )

    async def _trigger_task(
        self,
        task: TaskDefinition,
        *,
        trigger_instance_id: str,
        triggered_at: datetime,
        scheduled_for: datetime | None,
        coalesced: bool,
    ) -> TaskTriggeredEvent | None:
        async with self._trigger_lock:
            current = await self._store.get(task.task_id)
            if not current.enabled:
                return None
            next_run_at = (
                next_occurrence(
                    current.trigger, after=triggered_at, created_at=current.created_at
                )
                if isinstance(current.trigger, IntervalTrigger)
                or current.next_run_at is not None
                else None
            )
            run_id = f"run_{uuid4().hex}"
            run = TaskRunRecord(
                run_id=run_id,
                trigger_instance_id=trigger_instance_id,
                trigger_kind=current.trigger.kind,
                state=TaskRunState.READY,
                scheduled_for=scheduled_for,
                triggered_at=triggered_at,
                coalesced=coalesced,
            )
            recorded = await self._store.record_trigger(
                current.task_id, run, next_run_at=next_run_at
            )
            if not recorded.created:
                return None
            event = TaskTriggeredEvent(
                task_id=current.task_id,
                run_id=run_id,
                trigger_instance_id=trigger_instance_id,
                trigger_kind=current.trigger.kind,
                title=current.title,
                details=current.details,
                assigned_profile=current.assigned_profile,
                scheduled_for=scheduled_for,
                triggered_at=triggered_at,
                coalesced=coalesced,
            )
            self._emit(event)
            handoff = await self._handoff(current, run_id)
            await self._record_handoff(current.task_id, run_id, handoff)
            return event

    async def _handoff(self, task: TaskDefinition, run_id: str) -> TaskExecutionResult:
        port = self._execution_port
        if port is None:
            return TaskExecutionResult(
                disposition=TaskExecutionDisposition.QUEUED_FOR_APPROVAL
            )
        if task.assigned_profile is not None:
            try:
                available = port.is_profile_available(task.assigned_profile)
            except Exception as error:
                return _blocked_result(str(error) or type(error).__name__)
            if not available:
                return _blocked_result(
                    f"Assigned profile '{task.assigned_profile}' is unavailable"
                )
        request = TaskExecutionRequest(
            task_id=task.task_id,
            run_id=run_id,
            title=task.title,
            details=task.details,
            assigned_profile=task.assigned_profile,
            trigger_kind=task.trigger.kind,
        )
        try:
            return await port.handoff(request)
        except Exception as error:
            return _blocked_result(str(error) or type(error).__name__)

    async def _record_handoff(
        self, task_id: str, run_id: str, result: TaskExecutionResult
    ) -> None:
        match result.disposition:
            case TaskExecutionDisposition.QUEUED_FOR_APPROVAL:
                state = TaskRunState.QUEUED_FOR_APPROVAL
            case TaskExecutionDisposition.STARTED:
                state = TaskRunState.RUNNING
            case TaskExecutionDisposition.BLOCKED:
                state = TaskRunState.BLOCKED
        await self._store.record_handoff(
            task_id,
            run_id,
            state=state,
            authorization=result.authorization,
            error=result.error,
            managed_agent_id=result.managed_agent_id,
        )

    def _emit(self, event: TaskTriggeredEvent) -> None:
        for queue in tuple(self._subscribers):
            self._offer(queue, event)

    @staticmethod
    def _offer(
        queue: asyncio.Queue[TaskTriggeredEvent | None],
        event: TaskTriggeredEvent | None,
    ) -> None:
        if queue.full():
            with suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        queue.put_nowait(event)

    def _remember_source_event(self, key: str) -> None:
        self._seen_source_events.add(key)
        self._seen_source_order.append(key)
        while len(self._seen_source_order) > SEEN_SOURCE_EVENT_LIMIT:
            expired = self._seen_source_order.popleft()
            self._seen_source_events.discard(expired)

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Task scheduler clock must return an aware datetime")
        return now.astimezone(UTC)


def _stable_instance_id(*parts: str) -> str:
    payload = "\0".join(parts).encode()
    return f"trigger_{hashlib.sha256(payload).hexdigest()}"


def _blocked_result(error: str) -> TaskExecutionResult:
    return TaskExecutionResult(
        disposition=TaskExecutionDisposition.BLOCKED, error=error[:2_000]
    )
