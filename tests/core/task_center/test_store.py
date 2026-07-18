from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, time, timedelta
import tomllib

import pytest

from vibe.core.task_center import (
    AppStartTrigger,
    DailyTrigger,
    IntervalTrigger,
    ManualTrigger,
    SessionStartTrigger,
    TaskCenterDocument,
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
    TaskTriggerReceipt,
    TaskUpdate,
    Weekday,
    WeeklyTrigger,
    store as store_module,
)
from vibe.core.task_center.models import (
    MAX_TASK_TRIGGER_INDEX,
    TASK_TRIGGER_RETENTION_DAYS,
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
@pytest.mark.parametrize(
    "trigger",
    [
        ManualTrigger(),
        AppStartTrigger(),
        SessionStartTrigger(),
        IntervalTrigger(interval_seconds=90),
        DailyTrigger(at=time(9, 30), timezone="Europe/Paris"),
        WeeklyTrigger(
            at=time(10), timezone="UTC", weekdays=(Weekday.MONDAY, Weekday.FRIDAY)
        ),
    ],
)
async def test_every_trigger_kind_can_be_edited_and_reloaded(
    tmp_path, task_ids, trigger
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = _store(tmp_path, task_ids, now)
    task = await store.create(TaskCreate(title="Task"))

    updated = await store.update(task.task_id, TaskUpdate(trigger=trigger))
    reloaded = (await TaskStore(store.path).load())[0]

    assert updated.trigger == trigger
    assert reloaded.trigger == trigger


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
async def test_read_rejects_symlinked_task_center_parent(tmp_path, task_ids) -> None:
    external = tmp_path / "external"
    external_store = _store(external, task_ids, datetime(2026, 1, 1, tzinfo=UTC))
    await external_store.create(TaskCreate(title="External"))
    project = tmp_path / "project"
    project.mkdir()
    (project / ".vibe").symlink_to(external / ".vibe", target_is_directory=True)

    with pytest.raises(TaskStoreReadError, match="Unsafe Task Center directory"):
        await TaskStore(project_root=project).load()


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
async def test_oversized_serialization_is_rejected_before_replace(
    tmp_path, task_ids
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = _store(tmp_path, task_ids, now)
    await store.create(TaskCreate(title="Original"))
    previous = store.path.read_bytes()
    oversized = TaskCenterDocument(
        tasks=tuple(
            TaskDefinition(
                task_id=f"task_{index:032x}",
                title=f"Task {index}",
                details="x" * 10_000,
                created_at=now,
                updated_at=now,
            )
            for index in range(250)
        )
    )

    with pytest.raises(TaskStoreWriteError, match="size limit"):
        store._write_document(oversized)

    assert store.path.read_bytes() == previous
    assert list(store.path.parent.glob(f".{store.path.name}.*.tmp")) == []


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
async def test_duplicate_run_ids_are_rejected(tmp_path, task_ids) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = _store(tmp_path, task_ids, now)
    task = await store.create(TaskCreate(title="Task"))
    first = TaskRunRecord(
        run_id=f"run_{1:032x}",
        trigger_instance_id="first",
        trigger_kind=TaskTriggerKind.MANUAL,
        state=TaskRunState.READY,
        triggered_at=now,
    )
    await store.record_trigger(task.task_id, first, next_run_at=None)
    await store.record_handoff(
        task.task_id,
        first.run_id,
        state=TaskRunState.COMPLETED,
        authorization=TaskExecutionAuthorization.ASK,
    )
    duplicate = first.model_copy(update={"trigger_instance_id": "second"})

    with pytest.raises(TaskConflictError, match="run already exists"):
        await store.record_trigger(task.task_id, duplicate, next_run_at=None)


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
        await store.record_handoff(
            task.task_id,
            run.run_id,
            state=TaskRunState.COMPLETED,
            authorization=TaskExecutionAuthorization.ASK,
        )

    persisted = await store.get(task.task_id)
    history = persisted.run_history
    assert len(history) == 20
    assert history[0].trigger_instance_id == "manual-5"
    assert history[-1].trigger_instance_id == "manual-24"
    assert len(persisted.trigger_index) == 25
    assert persisted.state is TaskState.COMPLETED


@pytest.mark.asyncio
async def test_trigger_receipts_prune_by_count_and_age(tmp_path, task_ids) -> None:
    now = datetime(2026, 2, 1, tzinfo=UTC)
    store = _store(tmp_path, task_ids, now)
    task_id = next(task_ids)
    receipts = tuple(
        TaskTriggerReceipt(trigger_instance_id=f"receipt-{index}", recorded_at=now)
        for index in range(MAX_TASK_TRIGGER_INDEX)
    )
    definition = TaskDefinition(
        task_id=task_id,
        title="Receipts",
        created_at=now,
        updated_at=now,
        trigger_index=receipts,
    )
    store._write_document(TaskCenterDocument(tasks=(definition,)))
    await store.load()
    run = TaskRunRecord(
        run_id=f"run_{1:032x}",
        trigger_instance_id="new-receipt",
        trigger_kind=TaskTriggerKind.MANUAL,
        state=TaskRunState.READY,
        triggered_at=now,
    )

    await store.record_trigger(task_id, run, next_run_at=None)

    count_pruned = await store.get(task_id)
    count_ids = {receipt.trigger_instance_id for receipt in count_pruned.trigger_index}
    assert len(count_ids) == MAX_TASK_TRIGGER_INDEX
    assert "receipt-0" not in count_ids
    assert "new-receipt" in count_ids

    await store.record_handoff(
        task_id,
        run.run_id,
        state=TaskRunState.COMPLETED,
        authorization=TaskExecutionAuthorization.ASK,
    )
    old_receipt = TaskTriggerReceipt(
        trigger_instance_id="expired",
        recorded_at=now - timedelta(days=TASK_TRIGGER_RETENTION_DAYS + 1),
    )
    current = await store.get(task_id)
    store._write_document(
        TaskCenterDocument(
            tasks=(
                current.model_copy(
                    update={"trigger_index": (old_receipt, *current.trigger_index[-5:])}
                ),
            )
        )
    )
    await store.load()
    next_run = TaskRunRecord(
        run_id=f"run_{2:032x}",
        trigger_instance_id="after-expiry",
        trigger_kind=TaskTriggerKind.MANUAL,
        state=TaskRunState.READY,
        triggered_at=now,
    )

    await store.record_trigger(task_id, next_run, next_run_at=None)

    age_ids = {
        receipt.trigger_instance_id
        for receipt in (await store.get(task_id)).trigger_index
    }
    assert "expired" not in age_ids


@pytest.mark.asyncio
async def test_active_run_prevents_overlap_and_remains_completable(
    tmp_path, task_ids
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = _store(tmp_path, task_ids, now)
    task = await store.create(TaskCreate(title="Task"))
    active = TaskRunRecord(
        run_id=f"run_{1:032x}",
        trigger_instance_id="active",
        trigger_kind=TaskTriggerKind.MANUAL,
        state=TaskRunState.RUNNING,
        triggered_at=now,
    )
    assert (await store.record_trigger(task.task_id, active, next_run_at=None)).created
    await store.record_handoff(
        task.task_id,
        active.run_id,
        state=TaskRunState.RUNNING,
        authorization=TaskExecutionAuthorization.ALWAYS,
        managed_agent_id="worker-1",
    )

    for index in range(25):
        overlapping = TaskRunRecord(
            run_id=f"run_{index + 2:032x}",
            trigger_instance_id=f"overlap-{index}",
            trigger_kind=TaskTriggerKind.MANUAL,
            state=TaskRunState.READY,
            triggered_at=now + timedelta(seconds=index + 1),
        )
        result = await store.record_trigger(task.task_id, overlapping, next_run_at=None)
        assert not result.created
        assert result.blocked_by_active_run

    persisted = await store.get(task.task_id)
    assert persisted.active_run is not None
    assert persisted.active_run.run_id == active.run_id
    assert len(persisted.run_history) == 1

    completed = await store.record_handoff(
        task.task_id,
        active.run_id,
        state=TaskRunState.COMPLETED,
        authorization=TaskExecutionAuthorization.ALWAYS,
    )
    assert completed.run_history[-1].state is TaskRunState.COMPLETED


@pytest.mark.asyncio
async def test_legacy_active_runs_survive_terminal_pruning_until_completion(
    tmp_path, task_ids
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = _store(tmp_path, task_ids, now)
    task_id = next(task_ids)
    first_active = TaskRunRecord(
        run_id=f"run_{1:032x}",
        trigger_instance_id="active-1",
        trigger_kind=TaskTriggerKind.MANUAL,
        state=TaskRunState.RUNNING,
        triggered_at=now,
    )
    terminal = tuple(
        TaskRunRecord(
            run_id=f"run_{index + 10:032x}",
            trigger_instance_id=f"terminal-{index}",
            trigger_kind=TaskTriggerKind.MANUAL,
            state=TaskRunState.COMPLETED,
            triggered_at=now + timedelta(seconds=index + 1),
        )
        for index in range(25)
    )
    second_active = TaskRunRecord(
        run_id=f"run_{2:032x}",
        trigger_instance_id="active-2",
        trigger_kind=TaskTriggerKind.MANUAL,
        state=TaskRunState.RUNNING,
        triggered_at=now + timedelta(seconds=30),
    )
    runs = (first_active, *terminal, second_active)
    definition = TaskDefinition(
        task_id=task_id,
        title="Legacy overlaps",
        state=TaskState.RUNNING,
        created_at=now,
        updated_at=now,
        trigger_index=tuple(
            TaskTriggerReceipt(
                trigger_instance_id=run.trigger_instance_id,
                recorded_at=run.triggered_at,
            )
            for run in runs
        ),
        run_history=runs,
    )
    store._write_document(TaskCenterDocument(tasks=(definition,)))
    await store.load()

    after_first = await store.record_handoff(
        task_id,
        first_active.run_id,
        state=TaskRunState.COMPLETED,
        authorization=TaskExecutionAuthorization.ALWAYS,
    )

    assert any(run.run_id == second_active.run_id for run in after_first.run_history)
    assert len([run for run in after_first.run_history if run.state.is_terminal]) == 20
    assert after_first.state is TaskState.RUNNING

    after_second = await store.record_handoff(
        task_id,
        second_active.run_id,
        state=TaskRunState.COMPLETED,
        authorization=TaskExecutionAuthorization.ALWAYS,
    )
    assert after_second.active_run is None
