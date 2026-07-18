from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
import tomllib

import pytest

from vibe.core.task_center import (
    IntervalTrigger,
    TaskConflictError,
    TaskCreate,
    TaskDefinition,
    TaskExecutionAuthorization,
    TaskNotFoundError,
    TaskRunRecord,
    TaskRunState,
    TaskState,
    TaskStore,
    TaskStoreReadError,
    TaskStoreVersionError,
    TaskStoreWriteError,
    TaskTriggerKind,
    TaskUpdate,
    store as store_module,
)


@pytest.fixture
def task_ids() -> Iterator[str]:
    return (f"task_{value:032x}" for value in range(1, 100))


def _store(tmp_path, task_ids: Iterator[str], now: datetime) -> TaskStore:
    return TaskStore(
        tmp_path / ".vibe" / "tasks.toml",
        clock=lambda: now,
        id_factory=lambda: next(task_ids),
    )


@pytest.mark.asyncio
async def test_crud_round_trip_and_assignment_clear(tmp_path, task_ids) -> None:
    now = datetime(2026, 1, 1, 12, tzinfo=UTC)
    store = _store(tmp_path, task_ids, now)
    created = await store.create(
        TaskCreate(
            title="Daily review",
            details="Inspect failures",
            assigned_profile="explore",
            trigger=IntervalTrigger(interval_seconds=300),
        )
    )

    assert created.task_id == "task_00000000000000000000000000000001"
    assert created.next_run_at == now + timedelta(minutes=5)
    assert await store.get(created.task_id) == created

    updated = await store.update(
        created.task_id,
        TaskUpdate(title="Review failures", assigned_profile=None, enabled=False),
    )
    assert updated.title == "Review failures"
    assert updated.assigned_profile is None
    assert updated.next_run_at is None

    reloaded = TaskStore(store.path)
    assert await reloaded.load() == (updated,)

    deleted = await store.delete(created.task_id)
    assert deleted.task_id == created.task_id
    assert await store.list() == ()
    with pytest.raises(TaskNotFoundError):
        await store.get(created.task_id)


@pytest.mark.asyncio
async def test_runtime_agent_id_is_never_persisted(tmp_path, task_ids) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = _store(tmp_path, task_ids, now)
    task = await store.create(TaskCreate(title="Task"))

    runtime = store.set_runtime_managed_agent(task.task_id, "worker-1")
    assert runtime.managed_agent_id == "worker-1"
    assert "managed_agent_id" not in store.path.read_text(encoding="utf-8")

    reloaded = TaskStore(store.path)
    assert (await reloaded.load())[0].managed_agent_id is None


@pytest.mark.asyncio
async def test_listeners_receive_coherent_runtime_assignment(
    tmp_path, task_ids
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = _store(tmp_path, task_ids, now)
    task = await store.create(TaskCreate(title="Task"))
    run = TaskRunRecord(
        run_id=f"run_{1:032x}",
        trigger_instance_id="manual-1",
        trigger_kind=TaskTriggerKind.MANUAL,
        state=TaskRunState.READY,
        triggered_at=now,
    )
    await store.record_trigger(task.task_id, run, next_run_at=None)
    snapshots: list[tuple[TaskDefinition, ...]] = []
    store.add_listener(snapshots.append)

    await store.record_handoff(
        task.task_id,
        run.run_id,
        state=TaskRunState.RUNNING,
        authorization=TaskExecutionAuthorization.ALWAYS,
        managed_agent_id="worker-1",
    )

    assert len(snapshots) == 1
    assert snapshots[-1][0].managed_agent_id == "worker-1"
    snapshots.clear()

    await store.update(task.task_id, TaskUpdate(assigned_profile="explore"))

    assert len(snapshots) == 1
    assert snapshots[-1][0].managed_agent_id is None


@pytest.mark.asyncio
async def test_store_uses_structured_versioned_toml(tmp_path, task_ids) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = _store(tmp_path, task_ids, now)
    await store.create(TaskCreate(title="Task"))

    with store.path.open("rb") as file:
        persisted = tomllib.load(file)

    assert persisted["schema_version"] == 1
    assert persisted["tasks"][0]["trigger"] == {"kind": "manual"}


@pytest.mark.asyncio
async def test_corruption_and_unknown_version_are_safe_errors(tmp_path) -> None:
    path = tmp_path / ".vibe" / "tasks.toml"
    path.parent.mkdir(parents=True)
    path.write_text("not = [valid", encoding="utf-8")

    with pytest.raises(TaskStoreReadError):
        await TaskStore(path).load()
    assert path.read_text(encoding="utf-8") == "not = [valid"

    path.write_text("schema_version = 99\ntasks = []\n", encoding="utf-8")
    with pytest.raises(TaskStoreVersionError):
        await TaskStore(path).load()


@pytest.mark.asyncio
async def test_atomic_replace_failure_preserves_previous_file(
    tmp_path, task_ids, monkeypatch
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = _store(tmp_path, task_ids, now)
    task = await store.create(TaskCreate(title="Original"))
    previous = store.path.read_bytes()

    def fail_replace(_source, _target) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(store_module.os, "replace", fail_replace)
    with pytest.raises(TaskStoreWriteError):
        await store.update(task.task_id, TaskUpdate(title="Changed"))

    assert store.path.read_bytes() == previous
    assert (await store.get(task.task_id)).title == "Original"


@pytest.mark.asyncio
async def test_path_lock_prevents_two_stores_from_losing_updates(tmp_path) -> None:
    path = tmp_path / ".vibe" / "tasks.toml"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    first = TaskStore(path, clock=lambda: now, id_factory=lambda: f"task_{1:032x}")
    second = TaskStore(path, clock=lambda: now, id_factory=lambda: f"task_{2:032x}")

    await asyncio.gather(
        first.create(TaskCreate(title="First")),
        second.create(TaskCreate(title="Second")),
    )

    loaded = await TaskStore(path).load()
    assert {task.title for task in loaded} == {"First", "Second"}


@pytest.mark.asyncio
async def test_duplicate_ids_are_rejected(tmp_path) -> None:
    path = tmp_path / ".vibe" / "tasks.toml"
    task_id = f"task_{1:032x}"
    store = TaskStore(path, id_factory=lambda: task_id)
    await store.create(TaskCreate(title="First"))

    with pytest.raises(TaskConflictError):
        await store.create(TaskCreate(title="Second"))


@pytest.mark.asyncio
async def test_run_history_is_bounded(tmp_path, task_ids) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = _store(tmp_path, task_ids, now)
    task = await store.create(TaskCreate(title="Task"))

    for index in range(25):
        run = TaskRunRecord(
            run_id=f"run_{index:032x}",
            trigger_instance_id=f"manual-{index}",
            trigger_kind=TaskTriggerKind.MANUAL,
            state=TaskRunState.READY,
            triggered_at=now + timedelta(seconds=index),
        )
        await store.record_trigger(task.task_id, run, next_run_at=None)

    history = (await store.get(task.task_id)).run_history
    assert len(history) == 20
    assert history[0].trigger_instance_id == "manual-5"
    assert history[-1].trigger_instance_id == "manual-24"
    assert (await store.get(task.task_id)).state is TaskState.READY
