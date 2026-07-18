from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import suppress
from datetime import UTC, datetime, timedelta
import hashlib
from typing import Any, cast
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
    TaskTrigger,
    TaskTriggeredEvent,
)
from vibe.core.task_center.schedule import next_occurrence
from vibe.core.task_center.store import TaskNotFoundError, TaskStore

TASK_EVENT_QUEUE_SIZE = 64
INTERRUPTED_DISPATCH_ERROR = "Task dispatch interrupted; retry pending"
DEFAULT_TASK_RUN_CLAIM_LEASE_SECONDS = 300.0

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
        claim_lease_seconds: float = DEFAULT_TASK_RUN_CLAIM_LEASE_SECONDS,
    ) -> None:
        if claim_lease_seconds <= 0:
            raise ValueError("claim lease must be positive")
        self._store = store
        self._execution_port = execution_port
        self._clock = clock
        self._timer: asyncio.TimerHandle | None = None
        self._deadline_dispatch: asyncio.Task[None] | None = None
        self._inflight: set[asyncio.Task[object]] = set()
        self._dispatching_run_ids: set[str] = set()
        self._trigger_lock = asyncio.Lock()
        self._deadline_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._startup_task: asyncio.Task[None] | None = None
        self._subscribers: set[asyncio.Queue[TaskTriggeredEvent | None]] = set()
        self._owner_token = f"scheduler_{uuid4().hex}"
        self._claim_lease_seconds = claim_lease_seconds
        self._claim_renew_interval = max(0.05, claim_lease_seconds / 3)
        self._started = False

    @property
    def is_running(self) -> bool:
        return self._started

    @property
    def next_deadline(self) -> datetime | None:
        deadlines: list[datetime] = [
            task.next_run_at
            for task in self._store.snapshot
            if task.enabled and task.next_run_at is not None
        ]
        deadlines.extend(
            run.claim_expires_at
            for task in self._store.snapshot
            for run in task.run_history
            if run.state in {TaskRunState.READY, TaskRunState.RETRY_PENDING}
            and run.claim_owner not in {None, self._owner_token}
            and run.claim_expires_at is not None
        )
        return min(deadlines) if deadlines else None

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._started and self._startup_task is None:
                return
            if self._startup_task is None:
                self._started = True
                self._startup_task = asyncio.create_task(
                    self._start_impl(), name="vibe-task-center-start"
                )
            startup = self._startup_task
        try:
            await asyncio.shield(startup)
        finally:
            async with self._lifecycle_lock:
                if self._startup_task is startup and startup.done():
                    self._startup_task = None

    async def _start_impl(self) -> None:
        await self._store.load()
        if not self._started:
            return
        self._store.add_listener(self._on_store_changed)
        try:
            await self._await_tracked(
                self._run_deadline_dispatch(), name="vibe-task-center-startup"
            )
        except BaseException:
            self._started = False
            self._store.remove_listener(self._on_store_changed)
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            raise
        if self._started:
            self._reschedule_timer()

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            startup = self._startup_task
            if not self._started and not self._inflight and startup is None:
                return
            self._started = False
            self._store.remove_listener(self._on_store_changed)
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        while self._inflight:
            inflight = tuple(self._inflight)
            for task in inflight:
                task.cancel()
            for task in inflight:
                with suppress(asyncio.CancelledError, Exception):
                    await task
        if startup is not None and startup is not asyncio.current_task():
            with suppress(asyncio.CancelledError, Exception):
                await startup
        self._deadline_dispatch = None
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
        self._require_running()
        expected = {
            TaskEventKind.APP_START: AppStartTrigger,
            TaskEventKind.SESSION_START: SessionStartTrigger,
        }[event.kind]
        triggered: list[TaskTriggeredEvent] = []
        for task in self._store.snapshot:
            if not task.enabled or not isinstance(task.trigger, expected):
                continue
            instance_id = _stable_instance_id(
                "event", event.kind.value, event.event_id, task.task_id
            )
            result = await self._await_tracked(
                self._trigger_task(
                    task,
                    expected_trigger=task.trigger,
                    trigger_instance_id=instance_id,
                    triggered_at=self._now(),
                    scheduled_for=None,
                    coalesced=False,
                ),
                name=f"vibe-task-center-event-{task.task_id}",
            )
            if result is not None:
                triggered.append(result)
        return tuple(triggered)

    async def trigger_manual(
        self, task_id: str, *, request_id: str | None = None
    ) -> TaskTriggeredEvent | None:
        self._require_running()
        task = await self._store.get(task_id)
        if not isinstance(task.trigger, ManualTrigger):
            raise ValueError(f"Task '{task_id}' is not manually triggered")
        instance_id = _stable_instance_id(
            "manual", request_id or uuid4().hex, task.task_id
        )
        return await self._await_tracked(
            self._trigger_task(
                task,
                expected_trigger=task.trigger,
                trigger_instance_id=instance_id,
                triggered_at=self._now(),
                scheduled_for=None,
                coalesced=False,
            ),
            name=f"vibe-task-center-manual-{task.task_id}",
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
        if self._deadline_dispatch is not None and not self._deadline_dispatch.done():
            return
        task = self._create_tracked(
            self._run_deadline_dispatch(), name="vibe-task-center-deadline"
        )
        self._deadline_dispatch = task
        task.add_done_callback(self._deadline_finished)

    def _deadline_finished(self, task: asyncio.Task[None]) -> None:
        if self._deadline_dispatch is task:
            self._deadline_dispatch = None
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Task Center deadline dispatch failed")
        if self._started:
            self._reschedule_timer()

    async def _run_deadline_dispatch(self) -> None:
        async with self._deadline_lock:
            await self._recover_pending()
            await self._dispatch_due(self._now())

    async def _recover_pending(self) -> None:
        for task in self._store.snapshot:
            pending = next(
                (
                    run
                    for run in reversed(task.run_history)
                    if run.state in {TaskRunState.READY, TaskRunState.RETRY_PENDING}
                ),
                None,
            )
            if pending is None or pending.run_id in self._dispatching_run_ids:
                continue
            now = self._now()
            claim = await self._store.claim_run(
                task.task_id,
                pending.run_id,
                owner_token=self._owner_token,
                claimed_at=now,
                claim_expires_at=self._claim_expiry(now),
            )
            if not claim.claimed or claim.run is None:
                continue
            self._dispatching_run_ids.add(pending.run_id)
            try:
                await self._resume_run(claim.task, claim.run)
            finally:
                self._dispatching_run_ids.discard(pending.run_id)

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
                expected_trigger=task.trigger,
                trigger_instance_id=_stable_instance_id(
                    "schedule", task.task_id, scheduled_for.isoformat()
                ),
                triggered_at=now,
                scheduled_for=scheduled_for,
                coalesced=_has_missed_recurrence(task, scheduled_for, now),
            )

    async def _trigger_task(
        self,
        task: TaskDefinition,
        *,
        expected_trigger: TaskTrigger,
        trigger_instance_id: str,
        triggered_at: datetime,
        scheduled_for: datetime | None,
        coalesced: bool,
    ) -> TaskTriggeredEvent | None:
        run_id: str | None = None
        try:
            async with self._trigger_lock:
                current = await self._store.get(task.task_id)
                if not current.enabled or not self._started:
                    return None
                if current.trigger != expected_trigger:
                    return None
                if scheduled_for is not None and current.next_run_at != scheduled_for:
                    return None
                next_run_at = (
                    next_occurrence(
                        current.trigger,
                        after=triggered_at,
                        created_at=current.created_at,
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
                    claim_owner=self._owner_token,
                    claim_expires_at=self._claim_expiry(triggered_at),
                )
                recorded = await self._store.record_trigger(
                    current.task_id,
                    run,
                    next_run_at=next_run_at,
                    expected_trigger=expected_trigger,
                    expected_next_run_at=scheduled_for,
                    require_next_run_match=scheduled_for is not None,
                )
                if not recorded.created:
                    return None
                event = _event_from_run(current, run)
                self._emit(event)
            self._dispatching_run_ids.add(run_id)
            try:
                handoff = await self._handoff_with_claim(current, run_id)
                await self._persist_handoff(current.task_id, run_id, handoff)
            finally:
                self._dispatching_run_ids.discard(run_id)
            return event
        except asyncio.CancelledError:
            if run_id is not None:
                await self._reconcile_interrupted(task.task_id, run_id)
            raise

    async def _resume_run(
        self, task: TaskDefinition, run: TaskRunRecord
    ) -> TaskTriggeredEvent:
        try:
            event = _event_from_run(task, run)
            self._emit(event)
            handoff = await self._handoff_with_claim(task, run.run_id)
            await self._persist_handoff(task.task_id, run.run_id, handoff)
            return event
        except asyncio.CancelledError:
            await self._reconcile_interrupted(task.task_id, run.run_id)
            raise

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
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return _blocked_result(str(error) or type(error).__name__)

    async def _handoff_with_claim(
        self, task: TaskDefinition, run_id: str
    ) -> TaskExecutionResult:
        renewal = asyncio.create_task(
            self._renew_claim(task.task_id, run_id),
            name=f"vibe-task-center-claim-{run_id}",
        )
        try:
            return await self._handoff(task, run_id)
        finally:
            renewal.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await renewal

    async def _renew_claim(self, task_id: str, run_id: str) -> None:
        while True:
            await asyncio.sleep(self._claim_renew_interval)
            now = self._now()
            renewed = await self._store.renew_run_claim(
                task_id,
                run_id,
                owner_token=self._owner_token,
                renewed_at=now,
                claim_expires_at=self._claim_expiry(now),
            )
            if not renewed:
                return

    async def _persist_handoff(
        self, task_id: str, run_id: str, result: TaskExecutionResult
    ) -> None:
        match result.disposition:
            case TaskExecutionDisposition.QUEUED_FOR_APPROVAL:
                state = TaskRunState.QUEUED_FOR_APPROVAL
            case TaskExecutionDisposition.STARTED:
                state = TaskRunState.RUNNING
            case TaskExecutionDisposition.BLOCKED:
                state = TaskRunState.BLOCKED
        operation = self._store.record_handoff(
            task_id,
            run_id,
            state=state,
            authorization=result.authorization,
            error=result.error,
            managed_agent_id=result.managed_agent_id,
            expected_claim_owner=self._owner_token,
        )
        await _complete_before_cancellation(operation)

    async def _reconcile_interrupted(self, task_id: str, run_id: str) -> None:
        operation = asyncio.create_task(
            self._store.mark_retry_pending(
                task_id,
                run_id,
                owner_token=self._owner_token,
                error=INTERRUPTED_DISPATCH_ERROR,
            )
        )
        while True:
            try:
                await asyncio.shield(operation)
                return
            except asyncio.CancelledError:
                continue
            except TaskNotFoundError:
                return

    async def _await_tracked[T](
        self, operation: Coroutine[Any, Any, T], *, name: str
    ) -> T:
        task = self._create_tracked(operation, name=name)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
            raise

    def _create_tracked[T](
        self, operation: Coroutine[Any, Any, T], *, name: str
    ) -> asyncio.Task[T]:
        task = asyncio.create_task(operation, name=name)
        erased = cast(asyncio.Task[object], task)
        self._inflight.add(erased)
        task.add_done_callback(self._remove_inflight)
        return task

    def _remove_inflight(self, task: asyncio.Task[object]) -> None:
        self._inflight.discard(task)

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

    def _require_running(self) -> None:
        if not self._started:
            raise RuntimeError("Task scheduler is not running")

    def _claim_expiry(self, now: datetime) -> datetime:
        return now + timedelta(seconds=self._claim_lease_seconds)

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Task scheduler clock must return an aware datetime")
        return now.astimezone(UTC)


async def _complete_before_cancellation[T](operation: Coroutine[Any, Any, T]) -> T:
    task = asyncio.create_task(operation)
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


def _event_from_run(task: TaskDefinition, run: TaskRunRecord) -> TaskTriggeredEvent:
    return TaskTriggeredEvent(
        task_id=task.task_id,
        run_id=run.run_id,
        trigger_instance_id=run.trigger_instance_id,
        trigger_kind=run.trigger_kind,
        title=task.title,
        details=task.details,
        assigned_profile=task.assigned_profile,
        scheduled_for=run.scheduled_for,
        triggered_at=run.triggered_at,
        coalesced=run.coalesced,
    )


def _has_missed_recurrence(
    task: TaskDefinition, scheduled_for: datetime, now: datetime
) -> bool:
    following = next_occurrence(
        task.trigger, after=scheduled_for, created_at=task.created_at
    )
    return following is not None and following <= now


def _stable_instance_id(*parts: str) -> str:
    payload = "\0".join(parts).encode()
    return f"trigger_{hashlib.sha256(payload).hexdigest()}"


def _blocked_result(error: str) -> TaskExecutionResult:
    return TaskExecutionResult(
        disposition=TaskExecutionDisposition.BLOCKED, error=error[:2_000]
    )
