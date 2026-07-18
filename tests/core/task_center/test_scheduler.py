from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

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
    TaskRunState,
    TaskScheduler,
    TaskSourceEvent,
    TaskState,
    TaskStore,
    TaskUpdate,
)


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


@pytest.fixture
def task_ids() -> Iterator[str]:
    return (f"task_{value:032x}" for value in range(1, 100))


def _store(tmp_path, task_ids, clock) -> TaskStore:
    return TaskStore(
        tmp_path / ".vibe" / "tasks.toml",
        clock=clock,
        id_factory=lambda: next(task_ids),
    )


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
    assert scheduler.next_deadline is not None
    await store.update(task.task_id, TaskUpdate(enabled=False))
    assert scheduler.next_deadline is None
    assert scheduler._timer is None
    await events.aclose()
    await scheduler.stop()


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
