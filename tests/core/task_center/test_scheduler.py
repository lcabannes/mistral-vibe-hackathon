from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
import threading

import pytest

from vibe.core.task_center import (
    AppStartTrigger,
    IntervalTrigger,
    ManualTrigger,
    SessionStartTrigger,
    TaskCreate,
    TaskEventKind,
    TaskExecutionAuthorization,
    TaskExecutionDisposition,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskRunRecord,
    TaskRunState,
    TaskScheduler,
    TaskSourceEvent,
    TaskState,
    TaskStore,
    TaskTriggerKind,
    TaskUpdate,
)
from vibe.core.task_center._process_lock import process_file_lock


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class FakeExecutionPort:
    def __init__(
        self, result: TaskExecutionResult, *, unavailable: set[str] | None = None
    ) -> None:
        self.result = result
        self.unavailable = unavailable or set()
        self.requests: list[TaskExecutionRequest] = []

    def is_profile_available(self, profile: str) -> bool:
        return profile not in self.unavailable

    async def handoff(self, request: TaskExecutionRequest) -> TaskExecutionResult:
        self.requests.append(request)
        return self.result


class BlockingExecutionPort:
    def __init__(self, *, defer_cancellation: bool = False) -> None:
        self.defer_cancellation = defer_cancellation
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.requests: list[TaskExecutionRequest] = []

    def is_profile_available(self, profile: str) -> bool:
        del profile
        return True

    async def handoff(self, request: TaskExecutionRequest) -> TaskExecutionResult:
        self.requests.append(request)
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            if not self.defer_cancellation:
                raise
            await self.release.wait()
        return TaskExecutionResult(
            disposition=TaskExecutionDisposition.STARTED,
            authorization=TaskExecutionAuthorization.ALWAYS,
            managed_agent_id="worker-1",
        )


@pytest.fixture
def task_ids() -> Iterator[str]:
    return (f"task_{value:032x}" for value in range(1, 100))


def _store(tmp_path, task_ids, clock) -> TaskStore:
    return TaskStore(
        tmp_path / ".vibe" / "tasks.toml",
        clock=clock,
        id_factory=lambda: next(task_ids),
    )


async def _record_pending(store: TaskStore, task_id: str, now: datetime) -> str:
    run = TaskRunRecord(
        run_id=f"run_{1:032x}",
        trigger_instance_id="pending",
        trigger_kind=TaskTriggerKind.MANUAL,
        state=TaskRunState.READY,
        triggered_at=now,
    )
    await store.record_trigger(task_id, run, next_run_at=None)
    return run.run_id


@pytest.mark.asyncio
async def test_app_and_session_events_fire_only_matching_tasks_once(
    tmp_path, task_ids
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    app_task = await store.create(
        TaskCreate(title="App task", trigger=AppStartTrigger())
    )
    await store.create(TaskCreate(title="Session task", trigger=SessionStartTrigger()))
    scheduler = TaskScheduler(store, clock=clock)
    await scheduler.start()

    event = TaskSourceEvent(
        event_id="app-1", kind=TaskEventKind.APP_START, occurred_at=clock()
    )
    first = await scheduler.submit_event(event)
    duplicate = await scheduler.submit_event(event)

    assert [item.task_id for item in first] == [app_task.task_id]
    assert duplicate == ()
    assert (await store.get(app_task.task_id)).state is TaskState.QUEUED_FOR_APPROVAL
    await scheduler.stop()


@pytest.mark.asyncio
async def test_trigger_emits_before_default_approval_queue(tmp_path, task_ids) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    task = await store.create(TaskCreate(title="Manual", trigger=ManualTrigger()))
    scheduler = TaskScheduler(store, clock=clock)
    await scheduler.start()
    events = scheduler.events()
    pending = asyncio.create_task(anext(events))

    emitted = await scheduler.trigger_manual(task.task_id, request_id="request-1")

    assert await pending == emitted
    persisted = await store.get(task.task_id)
    assert persisted.state is TaskState.QUEUED_FOR_APPROVAL
    assert persisted.run_history[-1].state is TaskRunState.QUEUED_FOR_APPROVAL
    await events.aclose()
    await scheduler.stop()


@pytest.mark.asyncio
async def test_manual_request_id_prevents_duplicate_run(tmp_path, task_ids) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    task = await store.create(TaskCreate(title="Manual"))
    scheduler = TaskScheduler(store, clock=clock)
    await scheduler.start()

    assert await scheduler.trigger_manual(task.task_id, request_id="same") is not None
    assert await scheduler.trigger_manual(task.task_id, request_id="same") is None
    assert len((await store.get(task.task_id)).run_history) == 1
    await scheduler.stop()


@pytest.mark.asyncio
async def test_manual_request_id_survives_terminal_history_pruning(
    tmp_path, task_ids
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    task = await store.create(TaskCreate(title="Manual"))
    scheduler = TaskScheduler(store, clock=clock)
    await scheduler.start()

    for index in range(25):
        event = await scheduler.trigger_manual(
            task.task_id, request_id=f"request-{index}"
        )
        assert event is not None
        await store.record_handoff(
            task.task_id,
            event.run_id,
            state=TaskRunState.COMPLETED,
            authorization=TaskExecutionAuthorization.ASK,
        )

    persisted = await store.get(task.task_id)
    assert len(persisted.run_history) == 20
    assert len(persisted.trigger_index) == 25
    assert await scheduler.trigger_manual(task.task_id, request_id="request-0") is None
    await scheduler.stop()


@pytest.mark.asyncio
async def test_source_event_id_survives_restart_and_history_pruning(
    tmp_path, task_ids
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    task = await store.create(TaskCreate(title="App", trigger=AppStartTrigger()))
    scheduler = TaskScheduler(store, clock=clock)
    await scheduler.start()

    for index in range(25):
        source = TaskSourceEvent(
            event_id=f"app-{index}", kind=TaskEventKind.APP_START, occurred_at=clock()
        )
        (event,) = await scheduler.submit_event(source)
        await store.record_handoff(
            task.task_id,
            event.run_id,
            state=TaskRunState.COMPLETED,
            authorization=TaskExecutionAuthorization.ASK,
        )
    await scheduler.stop()

    restarted = TaskScheduler(TaskStore(store.path, clock=clock), clock=clock)
    await restarted.start()
    replay = TaskSourceEvent(
        event_id="app-0", kind=TaskEventKind.APP_START, occurred_at=clock()
    )

    assert await restarted.submit_event(replay) == ()
    await restarted.stop()


@pytest.mark.asyncio
async def test_concurrent_start_calls_share_one_recovery_handoff(
    tmp_path, task_ids
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    task = await store.create(TaskCreate(title="Pending"))
    await _record_pending(store, task.task_id, clock())
    port = FakeExecutionPort(
        TaskExecutionResult(disposition=TaskExecutionDisposition.QUEUED_FOR_APPROVAL)
    )
    scheduler = TaskScheduler(store, execution_port=port, clock=clock)

    first = asyncio.create_task(scheduler.start())
    second = asyncio.create_task(scheduler.start())
    await first
    await second

    assert len(port.requests) == 1
    await scheduler.stop()


@pytest.mark.asyncio
async def test_independent_schedulers_atomically_claim_one_recovery_handoff(
    tmp_path, task_ids
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    task = await store.create(TaskCreate(title="Pending"))
    run_id = await _record_pending(store, task.task_id, clock())
    first_port = FakeExecutionPort(
        TaskExecutionResult(disposition=TaskExecutionDisposition.QUEUED_FOR_APPROVAL)
    )
    second_port = FakeExecutionPort(
        TaskExecutionResult(disposition=TaskExecutionDisposition.QUEUED_FOR_APPROVAL)
    )
    first = TaskScheduler(
        TaskStore(store.path, clock=clock), execution_port=first_port, clock=clock
    )
    second = TaskScheduler(
        TaskStore(store.path, clock=clock), execution_port=second_port, clock=clock
    )

    first_start = asyncio.create_task(first.start())
    second_start = asyncio.create_task(second.start())
    await first_start
    await second_start

    requests = [*first_port.requests, *second_port.requests]
    assert [request.run_id for request in requests] == [run_id]
    await first.stop()
    await second.stop()


@pytest.mark.asyncio
async def test_live_scheduler_renews_claim_during_blocking_handoff(
    tmp_path, task_ids
) -> None:
    store = TaskStore(
        tmp_path / ".vibe" / "tasks.toml", id_factory=lambda: next(task_ids)
    )
    task = await store.create(TaskCreate(title="Pending"))
    await _record_pending(store, task.task_id, datetime.now(UTC))
    blocking = BlockingExecutionPort()
    first = TaskScheduler(
        TaskStore(store.path), execution_port=blocking, claim_lease_seconds=0.15
    )
    first_start = asyncio.create_task(first.start())
    await asyncio.wait_for(blocking.entered.wait(), timeout=1)
    await asyncio.sleep(0.3)
    second_port = FakeExecutionPort(
        TaskExecutionResult(disposition=TaskExecutionDisposition.QUEUED_FOR_APPROVAL)
    )
    second = TaskScheduler(
        TaskStore(store.path), execution_port=second_port, claim_lease_seconds=0.15
    )

    await second.start()

    assert second_port.requests == []
    blocking.release.set()
    await first_start
    await first.stop()
    await second.stop()


@pytest.mark.asyncio
async def test_claim_lease_starts_after_cross_process_lock_wait(
    tmp_path, task_ids
) -> None:
    store = TaskStore(
        tmp_path / ".vibe" / "tasks.toml", id_factory=lambda: next(task_ids)
    )
    task = await store.create(TaskCreate(title="Pending"))
    await _record_pending(store, task.task_id, datetime.now(UTC))
    lock_entered = threading.Event()
    release_lock = threading.Event()

    def hold_transaction_lock() -> None:
        with process_file_lock(store._lock_path):
            lock_entered.set()
            release_lock.wait(timeout=2)

    holder = asyncio.create_task(asyncio.to_thread(hold_transaction_lock))
    assert await asyncio.to_thread(lock_entered.wait, 1)
    blocking = BlockingExecutionPort()
    first = TaskScheduler(
        TaskStore(store.path), execution_port=blocking, claim_lease_seconds=0.05
    )
    first_start = asyncio.create_task(first.start())
    await asyncio.sleep(0.12)

    release_lock.set()
    await holder
    await asyncio.wait_for(blocking.entered.wait(), timeout=1)
    second_port = FakeExecutionPort(
        TaskExecutionResult(disposition=TaskExecutionDisposition.QUEUED_FOR_APPROVAL)
    )
    second = TaskScheduler(
        TaskStore(store.path), execution_port=second_port, claim_lease_seconds=0.05
    )
    await second.start()
    await asyncio.sleep(0.12)

    assert second_port.requests == []
    blocking.release.set()
    await first_start
    persisted = (await TaskStore(store.path).load())[0]
    assert persisted.state is TaskState.RUNNING
    assert persisted.run_history[-1].claim_owner is None
    await first.stop()
    await second.stop()


@pytest.mark.asyncio
async def test_expired_claim_is_recovered_with_same_run_id(tmp_path, task_ids) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    task = await store.create(TaskCreate(title="Pending"))
    run = TaskRunRecord(
        run_id=f"run_{1:032x}",
        trigger_instance_id="pending",
        trigger_kind=TaskTriggerKind.MANUAL,
        state=TaskRunState.READY,
        triggered_at=clock(),
        claim_owner="dead-scheduler",
        claim_expires_at=clock() + timedelta(minutes=1),
    )
    await store.record_trigger(task.task_id, run, next_run_at=None)
    early_port = FakeExecutionPort(
        TaskExecutionResult(disposition=TaskExecutionDisposition.QUEUED_FOR_APPROVAL)
    )
    early = TaskScheduler(store, execution_port=early_port, clock=clock)
    await early.start()
    assert early_port.requests == []
    await early.stop()

    clock.value += timedelta(minutes=2)
    recovered_port = FakeExecutionPort(
        TaskExecutionResult(disposition=TaskExecutionDisposition.QUEUED_FOR_APPROVAL)
    )
    recovered = TaskScheduler(store, execution_port=recovered_port, clock=clock)
    await recovered.start()

    assert recovered_port.requests[0].run_id == run.run_id
    await recovered.stop()


@pytest.mark.asyncio
async def test_explicit_always_handoff_starts_and_runtime_id_is_not_durable(
    tmp_path, task_ids
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    task = await store.create(TaskCreate(title="Assigned", assigned_profile="explore"))
    port = FakeExecutionPort(
        TaskExecutionResult(
            disposition=TaskExecutionDisposition.STARTED,
            authorization=TaskExecutionAuthorization.ALWAYS,
            managed_agent_id="worker-1",
        )
    )
    scheduler = TaskScheduler(store, execution_port=port, clock=clock)
    await scheduler.start()

    await scheduler.trigger_manual(task.task_id)

    started = await store.get(task.task_id)
    assert started.state is TaskState.RUNNING
    assert started.managed_agent_id == "worker-1"
    assert port.requests[0].assigned_profile == "explore"
    assert (await TaskStore(store.path).load())[0].managed_agent_id is None
    await scheduler.stop()


@pytest.mark.asyncio
async def test_unavailable_assignment_blocks_without_reassignment(
    tmp_path, task_ids
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    task = await store.create(TaskCreate(title="Assigned", assigned_profile="missing"))
    port = FakeExecutionPort(
        TaskExecutionResult(disposition=TaskExecutionDisposition.QUEUED_FOR_APPROVAL),
        unavailable={"missing"},
    )
    scheduler = TaskScheduler(store, execution_port=port, clock=clock)
    await scheduler.start()

    await scheduler.trigger_manual(task.task_id)

    blocked = await store.get(task.task_id)
    assert blocked.state is TaskState.BLOCKED
    assert "missing" in (blocked.last_error or "")
    assert port.requests == []
    await scheduler.stop()


@pytest.mark.asyncio
async def test_unassigned_task_remains_unassigned_in_handoff(
    tmp_path, task_ids
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    task = await store.create(TaskCreate(title="Unassigned"))
    port = FakeExecutionPort(
        TaskExecutionResult(disposition=TaskExecutionDisposition.QUEUED_FOR_APPROVAL)
    )
    scheduler = TaskScheduler(store, execution_port=port, clock=clock)
    await scheduler.start()

    await scheduler.trigger_manual(task.task_id)

    assert port.requests[0].assigned_profile is None
    await scheduler.stop()


@pytest.mark.asyncio
async def test_missed_intervals_coalesce_once_and_advance_past_startup(
    tmp_path, task_ids
) -> None:
    initial = datetime(2026, 1, 1, tzinfo=UTC)
    clock = MutableClock(initial)
    store = _store(tmp_path, task_ids, clock)
    task = await store.create(
        TaskCreate(title="Interval", trigger=IntervalTrigger(interval_seconds=60))
    )
    clock.value = initial + timedelta(minutes=10, seconds=30)
    scheduler = TaskScheduler(store, clock=clock)

    await scheduler.start()

    updated = await store.get(task.task_id)
    assert len(updated.run_history) == 1
    assert updated.run_history[0].coalesced
    assert updated.run_history[0].scheduled_for == initial + timedelta(minutes=1)
    assert updated.next_run_at == initial + timedelta(minutes=11)
    await scheduler.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replacement", [ManualTrigger(), IntervalTrigger(interval_seconds=120)]
)
async def test_queued_deadline_revalidates_trigger_edit(
    tmp_path, task_ids, replacement
) -> None:
    initial = datetime(2026, 1, 1, tzinfo=UTC)
    clock = MutableClock(initial)
    store = _store(tmp_path, task_ids, clock)
    first = await store.create(
        TaskCreate(title="First", trigger=IntervalTrigger(interval_seconds=60))
    )
    second = await store.create(
        TaskCreate(title="Second", trigger=IntervalTrigger(interval_seconds=60))
    )
    clock.value += timedelta(minutes=2)
    blocking = BlockingExecutionPort()
    scheduler = TaskScheduler(store, execution_port=blocking, clock=clock)
    starting = asyncio.create_task(scheduler.start())
    await asyncio.wait_for(blocking.entered.wait(), timeout=1)

    await store.update(second.task_id, TaskUpdate(trigger=replacement))
    blocking.release.set()
    await starting

    assert [request.task_id for request in blocking.requests] == [first.task_id]
    assert (await store.get(second.task_id)).run_history == ()
    await scheduler.stop()


@pytest.mark.asyncio
async def test_queued_source_event_revalidates_trigger_edit(tmp_path, task_ids) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    first = await store.create(TaskCreate(title="First", trigger=AppStartTrigger()))
    second = await store.create(TaskCreate(title="Second", trigger=AppStartTrigger()))
    blocking = BlockingExecutionPort()
    scheduler = TaskScheduler(store, execution_port=blocking, clock=clock)
    await scheduler.start()
    dispatch = asyncio.create_task(
        scheduler.submit_event(
            TaskSourceEvent(
                event_id="app-1", kind=TaskEventKind.APP_START, occurred_at=clock()
            )
        )
    )
    await asyncio.wait_for(blocking.entered.wait(), timeout=1)

    await store.update(second.task_id, TaskUpdate(trigger=SessionStartTrigger()))
    blocking.release.set()
    await dispatch

    assert [request.task_id for request in blocking.requests] == [first.task_id]
    assert (await store.get(second.task_id)).run_history == ()
    await scheduler.stop()


@pytest.mark.asyncio
async def test_queued_manual_request_revalidates_trigger_edit(
    tmp_path, task_ids
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    task = await store.create(TaskCreate(title="Manual"))
    port = FakeExecutionPort(
        TaskExecutionResult(disposition=TaskExecutionDisposition.QUEUED_FOR_APPROVAL)
    )
    scheduler = TaskScheduler(store, execution_port=port, clock=clock)
    await scheduler.start()
    await scheduler._trigger_lock.acquire()
    dispatch = asyncio.create_task(scheduler.trigger_manual(task.task_id))
    await asyncio.sleep(0)

    await store.update(
        task.task_id, TaskUpdate(trigger=IntervalTrigger(interval_seconds=60))
    )
    scheduler._trigger_lock.release()

    assert await dispatch is None
    assert port.requests == []
    await scheduler.stop()


@pytest.mark.asyncio
async def test_timer_fires_and_reschedules_without_polling(tmp_path, task_ids) -> None:
    store = TaskStore(
        tmp_path / ".vibe" / "tasks.toml", id_factory=lambda: next(task_ids)
    )
    task = await store.create(
        TaskCreate(title="Fast", trigger=IntervalTrigger(interval_seconds=0.03))
    )
    scheduler = TaskScheduler(store)
    await scheduler.start()
    events = scheduler.events()

    fired = await asyncio.wait_for(anext(events), timeout=1)

    assert fired.task_id == task.task_id
    assert not fired.coalesced
    assert scheduler.next_deadline is not None
    await store.update(task.task_id, TaskUpdate(enabled=False))
    assert scheduler.next_deadline is None
    assert scheduler._timer is None
    await events.aclose()
    await scheduler.stop()


@pytest.mark.asyncio
async def test_manual_shutdown_marks_retry_and_restart_reuses_run_id(
    tmp_path, task_ids
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    task = await store.create(TaskCreate(title="Manual"))
    blocking = BlockingExecutionPort()
    scheduler = TaskScheduler(store, execution_port=blocking, clock=clock)
    await scheduler.start()
    dispatch = asyncio.create_task(
        scheduler.trigger_manual(task.task_id, request_id="request-1")
    )
    await asyncio.wait_for(blocking.entered.wait(), timeout=1)
    run_id = blocking.requests[0].run_id

    await scheduler.stop()

    with pytest.raises(asyncio.CancelledError):
        await dispatch
    assert blocking.cancelled.is_set()
    interrupted = await store.get(task.task_id)
    assert interrupted.active_run is not None
    assert interrupted.active_run.state is TaskRunState.RETRY_PENDING

    recovery = FakeExecutionPort(
        TaskExecutionResult(disposition=TaskExecutionDisposition.QUEUED_FOR_APPROVAL)
    )
    restarted = TaskScheduler(store, execution_port=recovery, clock=clock)
    await restarted.start()

    assert recovery.requests[0].run_id == run_id
    assert (await store.get(task.task_id)).state is TaskState.QUEUED_FOR_APPROVAL
    await restarted.stop()


@pytest.mark.asyncio
async def test_source_shutdown_marks_retry_pending(tmp_path, task_ids) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    task = await store.create(TaskCreate(title="App", trigger=AppStartTrigger()))
    blocking = BlockingExecutionPort()
    scheduler = TaskScheduler(store, execution_port=blocking, clock=clock)
    await scheduler.start()
    dispatch = asyncio.create_task(
        scheduler.submit_event(
            TaskSourceEvent(
                event_id="app-1", kind=TaskEventKind.APP_START, occurred_at=clock()
            )
        )
    )
    await asyncio.wait_for(blocking.entered.wait(), timeout=1)

    await scheduler.stop()

    with pytest.raises(asyncio.CancelledError):
        await dispatch
    interrupted = await store.get(task.task_id)
    assert interrupted.active_run is not None
    assert interrupted.active_run.state is TaskRunState.RETRY_PENDING


@pytest.mark.asyncio
async def test_timer_shutdown_marks_retry_pending(tmp_path, task_ids) -> None:
    store = TaskStore(
        tmp_path / ".vibe" / "tasks.toml", id_factory=lambda: next(task_ids)
    )
    task = await store.create(
        TaskCreate(title="Fast", trigger=IntervalTrigger(interval_seconds=0.03))
    )
    blocking = BlockingExecutionPort()
    scheduler = TaskScheduler(store, execution_port=blocking)
    await scheduler.start()
    await asyncio.wait_for(blocking.entered.wait(), timeout=1)

    await scheduler.stop()

    interrupted = await store.get(task.task_id)
    assert interrupted.active_run is not None
    assert interrupted.active_run.state is TaskRunState.RETRY_PENDING


@pytest.mark.asyncio
async def test_stop_waits_for_cancellation_deferring_handoff(
    tmp_path, task_ids
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    task = await store.create(TaskCreate(title="Manual"))
    blocking = BlockingExecutionPort(defer_cancellation=True)
    scheduler = TaskScheduler(store, execution_port=blocking, clock=clock)
    await scheduler.start()
    dispatch = asyncio.create_task(scheduler.trigger_manual(task.task_id))
    await asyncio.wait_for(blocking.entered.wait(), timeout=1)

    stopping = asyncio.create_task(scheduler.stop())
    await asyncio.wait_for(blocking.cancelled.wait(), timeout=1)
    assert not stopping.done()
    blocking.release.set()
    await asyncio.wait_for(stopping, timeout=1)
    await dispatch

    stopped_state = await store.get(task.task_id)
    assert stopped_state.state is TaskState.RUNNING
    assert stopped_state.managed_agent_id == "worker-1"
    await asyncio.sleep(0)
    assert await store.get(task.task_id) == stopped_state


@pytest.mark.asyncio
async def test_shutdown_cancels_timer_and_closes_subscribers(
    tmp_path, task_ids
) -> None:
    clock = MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    store = _store(tmp_path, task_ids, clock)
    await store.create(
        TaskCreate(title="Later", trigger=IntervalTrigger(interval_seconds=3600))
    )
    scheduler = TaskScheduler(store, clock=clock)
    await scheduler.start()
    events = scheduler.events()
    pending = asyncio.create_task(anext(events))
    await asyncio.sleep(0)

    await scheduler.stop()

    with pytest.raises(StopAsyncIteration):
        await pending
    assert not scheduler.is_running
    assert scheduler._timer is None
