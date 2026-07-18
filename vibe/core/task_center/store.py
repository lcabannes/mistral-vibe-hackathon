from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import tempfile
import tomllib
from typing import cast
from uuid import uuid4

from pydantic import ValidationError
import tomli_w

from vibe.core.task_center._process_lock import process_file_lock
from vibe.core.task_center.models import (
    MAX_TASK_CLAIM_OWNER_LENGTH,
    MAX_TASK_ERROR_LENGTH,
    MAX_TASK_RUN_HISTORY,
    MAX_TASK_TRIGGER_INDEX,
    TASK_CENTER_SCHEMA_VERSION,
    TASK_TRIGGER_RETENTION_DAYS,
    TaskCenterDocument,
    TaskCreate,
    TaskDefinition,
    TaskExecutionAuthorization,
    TaskRunRecord,
    TaskRunState,
    TaskState,
    TaskTrigger,
    TaskTriggerReceipt,
    TaskUpdate,
)
from vibe.core.task_center.schedule import next_occurrence
from vibe.core.utils.io import file_write_lock, read_safe

TASK_CENTER_PATH = Path(".vibe") / "tasks.toml"
MAX_TASK_CENTER_FILE_BYTES = 2 * 1024 * 1024

type TaskStoreListener = Callable[[tuple[TaskDefinition, ...]], None]
type TaskIdFactory = Callable[[], str]
type Clock = Callable[[], datetime]


class TaskStoreError(Exception):
    pass


class TaskStoreReadError(TaskStoreError):
    pass


class TaskStoreWriteError(TaskStoreError):
    pass


class TaskStoreVersionError(TaskStoreReadError):
    pass


class TaskNotFoundError(TaskStoreError):
    pass


class TaskConflictError(TaskStoreError):
    pass


@dataclass(frozen=True, slots=True)
class TaskTriggerRecordResult:
    task: TaskDefinition
    created: bool
    blocked_by_active_run: bool = False


@dataclass(frozen=True, slots=True)
class TaskRunClaimResult:
    task: TaskDefinition
    run: TaskRunRecord | None
    claimed: bool


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_task_id() -> str:
    return f"task_{uuid4().hex}"


class TaskStore:
    def __init__(
        self,
        path: Path | None = None,
        *,
        project_root: Path | None = None,
        clock: Clock = _utc_now,
        id_factory: TaskIdFactory = _new_task_id,
    ) -> None:
        if path is not None and project_root is not None:
            raise ValueError("path and project_root are mutually exclusive")
        self.path = (
            path
            if path is not None
            else (project_root or Path.cwd()).resolve() / TASK_CENTER_PATH
        )
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")
        self._clock = clock
        self._id_factory = id_factory
        self._tasks: dict[str, TaskDefinition] = {}
        self._runtime_agent_ids: dict[str, str] = {}
        self._listeners: list[TaskStoreListener] = []
        self._loaded = False

    @property
    def snapshot(self) -> tuple[TaskDefinition, ...]:
        return tuple(
            self._with_runtime(task)
            for task in sorted(
                self._tasks.values(), key=lambda item: (item.created_at, item.task_id)
            )
        )

    def add_listener(self, listener: TaskStoreListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: TaskStoreListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def load(self) -> tuple[TaskDefinition, ...]:
        async with file_write_lock(self.path):
            document = await asyncio.to_thread(self._read_document)
            self._replace_memory(document, clear_runtime=True)
        return self.snapshot

    async def create(self, request: TaskCreate) -> TaskDefinition:
        now = self._now()
        created_id: str | None = None

        def mutate(document: TaskCenterDocument) -> TaskCenterDocument:
            nonlocal created_id
            existing = {task.task_id for task in document.tasks}
            task_id = self._id_factory()
            if task_id in existing:
                raise TaskConflictError(f"Task already exists: {task_id}")
            created_id = task_id
            task = TaskDefinition(
                task_id=task_id,
                title=request.title,
                details=request.details,
                enabled=request.enabled,
                assigned_profile=request.assigned_profile,
                trigger=request.trigger,
                created_at=now,
                updated_at=now,
                next_run_at=(
                    next_occurrence(request.trigger, after=now, created_at=now)
                    if request.enabled
                    else None
                ),
            )
            return TaskCenterDocument(tasks=(*document.tasks, task))

        await self._mutate(mutate)
        assert created_id is not None
        return await self.get(created_id)

    async def update(self, task_id: str, request: TaskUpdate) -> TaskDefinition:
        now = self._now()
        assignment_changed = "assigned_profile" in request.model_fields_set

        def mutate(document: TaskCenterDocument) -> TaskCenterDocument:
            current = self._find(document, task_id)
            changes = request.model_dump(exclude_unset=True)
            trigger_changed = "trigger" in changes
            if trigger_changed:
                assert request.trigger is not None
                changes["trigger"] = request.trigger.model_dump()
            enabled_changed = "enabled" in changes
            durable = current.model_dump(exclude={"managed_agent_id"})
            durable.update(changes)
            durable["updated_at"] = now
            updated = TaskDefinition.model_validate(durable)
            if trigger_changed or enabled_changed:
                next_run_at = (
                    next_occurrence(
                        updated.trigger, after=now, created_at=updated.created_at
                    )
                    if updated.enabled
                    else None
                )
                updated = updated.model_copy(update={"next_run_at": next_run_at})
            return self._replace_task(document, updated)

        await self._mutate(
            mutate,
            memory_update=(
                lambda: (
                    self._runtime_agent_ids.pop(task_id, None)
                    if assignment_changed
                    else None
                )
            ),
        )
        return await self.get(task_id)

    async def delete(self, task_id: str) -> TaskDefinition:
        deleted: TaskDefinition | None = None

        def mutate(document: TaskCenterDocument) -> TaskCenterDocument:
            nonlocal deleted
            deleted = self._find(document, task_id)
            return TaskCenterDocument(
                tasks=tuple(task for task in document.tasks if task.task_id != task_id)
            )

        await self._mutate(mutate)
        self._runtime_agent_ids.pop(task_id, None)
        assert deleted is not None
        return deleted

    async def get(self, task_id: str) -> TaskDefinition:
        await self._ensure_loaded()
        try:
            task = self._tasks[task_id]
        except KeyError as error:
            raise TaskNotFoundError(f"Unknown task: {task_id}") from error
        return self._with_runtime(task)

    async def list(self) -> tuple[TaskDefinition, ...]:
        await self._ensure_loaded()
        return self.snapshot

    async def record_trigger(
        self,
        task_id: str,
        run: TaskRunRecord,
        *,
        next_run_at: datetime | None,
        expected_trigger: TaskTrigger | None = None,
        expected_next_run_at: datetime | None = None,
        require_next_run_match: bool = False,
    ) -> TaskTriggerRecordResult:
        created = False
        blocked_by_active_run = False

        def mutate(document: TaskCenterDocument) -> TaskCenterDocument:
            nonlocal blocked_by_active_run, created
            current = self._find(document, task_id)
            if expected_trigger is not None and current.trigger != expected_trigger:
                return document
            if require_next_run_match and current.next_run_at != expected_next_run_at:
                return document
            if any(
                receipt.trigger_instance_id == run.trigger_instance_id
                for receipt in current.trigger_index
            ):
                return document
            if any(item.run_id == run.run_id for item in current.run_history):
                raise TaskConflictError(f"Task run already exists: {run.run_id}")
            if current.active_run is not None:
                blocked_by_active_run = True
                if next_run_at == current.next_run_at:
                    return document
                updated = current.model_copy(
                    update={"updated_at": run.triggered_at, "next_run_at": next_run_at}
                )
                return self._replace_task(document, updated)
            created = True
            history = _prune_run_history((*current.run_history, run))
            trigger_index = _prune_trigger_index(
                (
                    *current.trigger_index,
                    TaskTriggerReceipt(
                        trigger_instance_id=run.trigger_instance_id,
                        recorded_at=run.triggered_at,
                    ),
                ),
                now=run.triggered_at,
            )
            updated = current.model_copy(
                update={
                    "state": TaskState.READY,
                    "updated_at": run.triggered_at,
                    "last_run_at": run.triggered_at,
                    "next_run_at": next_run_at,
                    "last_error": None,
                    "trigger_index": trigger_index,
                    "run_history": history,
                }
            )
            return self._replace_task(document, updated)

        await self._mutate(mutate, write_when_unchanged=False)
        return TaskTriggerRecordResult(
            task=await self.get(task_id),
            created=created,
            blocked_by_active_run=blocked_by_active_run,
        )

    async def record_handoff(
        self,
        task_id: str,
        run_id: str,
        *,
        state: TaskRunState,
        authorization: TaskExecutionAuthorization,
        error: str | None = None,
        managed_agent_id: str | None = None,
        expected_claim_owner: str | None = None,
    ) -> TaskDefinition:
        now = self._now()
        bounded_error = error[:MAX_TASK_ERROR_LENGTH] if error else None

        def mutate(document: TaskCenterDocument) -> TaskCenterDocument:
            current = self._find(document, task_id)
            found = False
            history: list[TaskRunRecord] = []
            for run in current.run_history:
                if run.run_id != run_id:
                    history.append(run)
                    continue
                if (
                    expected_claim_owner is not None
                    and run.claim_owner != expected_claim_owner
                ):
                    raise TaskConflictError(
                        f"Task run claim is no longer owned: {run_id}"
                    )
                found = True
                history.append(
                    run.model_copy(
                        update={
                            "state": state,
                            "authorization": authorization,
                            "error": bounded_error,
                            "claim_owner": None,
                            "claim_expires_at": None,
                        }
                    )
                )
            if not found:
                raise TaskConflictError(f"Unknown task run: {run_id}")
            pruned_history = _prune_run_history(tuple(history))
            active = next(
                (run for run in reversed(pruned_history) if not run.state.is_terminal),
                None,
            )
            task_state = _task_state(active.state if active is not None else state)
            updated = current.model_copy(
                update={
                    "state": task_state,
                    "updated_at": now,
                    "last_error": bounded_error,
                    "run_history": pruned_history,
                }
            )
            return self._replace_task(document, updated)

        def update_runtime() -> None:
            if managed_agent_id is None:
                self._runtime_agent_ids.pop(task_id, None)
            else:
                self._runtime_agent_ids[task_id] = managed_agent_id

        await self._mutate(mutate, memory_update=update_runtime)
        return await self.get(task_id)

    async def claim_run(
        self,
        task_id: str,
        run_id: str,
        *,
        owner_token: str,
        claimed_at: datetime,
        claim_expires_at: datetime,
    ) -> TaskRunClaimResult:
        claimed = False

        def mutate(document: TaskCenterDocument) -> TaskCenterDocument:
            nonlocal claimed
            current = self._find(document, task_id)
            target = self._find_run(current, run_id)
            if target.state not in {TaskRunState.READY, TaskRunState.RETRY_PENDING}:
                return document
            if (
                target.claim_owner is not None
                and target.claim_owner != owner_token
                and target.claim_expires_at is not None
                and target.claim_expires_at > claimed_at
            ):
                return document
            claimed = True
            updated_run = target.model_copy(
                update={
                    "claim_owner": owner_token,
                    "claim_expires_at": claim_expires_at,
                }
            )
            updated = current.model_copy(
                update={
                    "state": TaskState.READY,
                    "updated_at": claimed_at,
                    "run_history": tuple(
                        updated_run if run.run_id == run_id else run
                        for run in current.run_history
                    ),
                }
            )
            return self._replace_task(document, updated)

        _validate_claim_window(claimed_at, claim_expires_at)
        _validate_owner_token(owner_token)
        await self._mutate(mutate, write_when_unchanged=False)
        task = await self.get(task_id)
        return TaskRunClaimResult(
            task=task,
            run=next((run for run in task.run_history if run.run_id == run_id), None),
            claimed=claimed,
        )

    async def renew_run_claim(
        self,
        task_id: str,
        run_id: str,
        *,
        owner_token: str,
        renewed_at: datetime,
        claim_expires_at: datetime,
    ) -> bool:
        renewed = False

        def mutate(document: TaskCenterDocument) -> TaskCenterDocument:
            nonlocal renewed
            current = self._find(document, task_id)
            target = self._find_run(current, run_id)
            if (
                target.state not in {TaskRunState.READY, TaskRunState.RETRY_PENDING}
                or target.claim_owner != owner_token
            ):
                return document
            renewed = True
            updated_run = target.model_copy(
                update={"claim_expires_at": claim_expires_at}
            )
            updated = current.model_copy(
                update={
                    "updated_at": renewed_at,
                    "run_history": tuple(
                        updated_run if run.run_id == run_id else run
                        for run in current.run_history
                    ),
                }
            )
            return self._replace_task(document, updated)

        _validate_claim_window(renewed_at, claim_expires_at)
        _validate_owner_token(owner_token)
        await self._mutate(mutate, write_when_unchanged=False)
        return renewed

    async def mark_retry_pending(
        self, task_id: str, run_id: str, *, owner_token: str, error: str
    ) -> TaskDefinition:
        _validate_owner_token(owner_token)
        now = self._now()
        bounded_error = error[:MAX_TASK_ERROR_LENGTH]

        def mutate(document: TaskCenterDocument) -> TaskCenterDocument:
            current = self._find(document, task_id)
            history: list[TaskRunRecord] = []
            changed = False
            for run in current.run_history:
                if (
                    run.run_id != run_id
                    or run.state not in {TaskRunState.READY, TaskRunState.RETRY_PENDING}
                    or run.claim_owner != owner_token
                ):
                    history.append(run)
                    continue
                changed = run.state is not TaskRunState.RETRY_PENDING or (
                    run.error != bounded_error
                )
                history.append(
                    run.model_copy(
                        update={
                            "state": TaskRunState.RETRY_PENDING,
                            "error": bounded_error,
                            "claim_owner": None,
                            "claim_expires_at": None,
                        }
                    )
                )
            if not changed:
                return document
            updated = current.model_copy(
                update={
                    "state": TaskState.READY,
                    "updated_at": now,
                    "last_error": bounded_error,
                    "run_history": tuple(history),
                }
            )
            return self._replace_task(document, updated)

        await self._mutate(mutate, write_when_unchanged=False)
        return await self.get(task_id)

    def set_runtime_managed_agent(
        self, task_id: str, managed_agent_id: str | None
    ) -> TaskDefinition:
        if task_id not in self._tasks:
            raise TaskNotFoundError(f"Unknown task: {task_id}")
        if managed_agent_id is None:
            self._runtime_agent_ids.pop(task_id, None)
        else:
            stripped = managed_agent_id.strip()
            if not stripped:
                raise ValueError("managed_agent_id must not be blank")
            self._runtime_agent_ids[task_id] = stripped
        self._notify_listeners()
        return self._with_runtime(self._tasks[task_id])

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            await self.load()

    async def _mutate(
        self,
        mutation: Callable[[TaskCenterDocument], TaskCenterDocument],
        *,
        write_when_unchanged: bool = True,
        memory_update: Callable[[], object] | None = None,
    ) -> None:
        async with file_write_lock(self.path):
            transaction = asyncio.create_task(
                asyncio.to_thread(self._run_transaction, mutation, write_when_unchanged)
            )
            cancelled = False
            while True:
                try:
                    updated = await asyncio.shield(transaction)
                    break
                except asyncio.CancelledError:
                    cancelled = True
            if memory_update is not None:
                memory_update()
            self._replace_memory(updated, force_notify=memory_update is not None)
            if cancelled:
                raise asyncio.CancelledError

    def _run_transaction(
        self,
        mutation: Callable[[TaskCenterDocument], TaskCenterDocument],
        write_when_unchanged: bool,
    ) -> TaskCenterDocument:
        self._ensure_safe_parent()
        try:
            with process_file_lock(self._lock_path):
                document = self._read_document()
                updated = mutation(document)
                if write_when_unchanged or updated != document:
                    self._write_document(updated)
                return updated
        except TaskStoreError:
            raise
        except OSError as error:
            raise TaskStoreWriteError(
                f"Failed to lock Task Center: {self.path}"
            ) from error

    def _read_document(self) -> TaskCenterDocument:
        self._validate_safe_parent_for_read()
        if not self.path.exists():
            return TaskCenterDocument()
        if self.path.is_symlink() or not self.path.is_file():
            raise TaskStoreReadError(f"Unsafe Task Center path: {self.path}")
        try:
            if self.path.stat().st_size > MAX_TASK_CENTER_FILE_BYTES:
                raise TaskStoreReadError("Task Center file exceeds the size limit")
            raw = tomllib.loads(read_safe(self.path, raise_on_error=True).text)
        except TaskStoreReadError:
            raise
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
            raise TaskStoreReadError(
                f"Invalid Task Center file: {self.path}"
            ) from error
        version = raw.get("schema_version")
        if version != TASK_CENTER_SCHEMA_VERSION:
            raise TaskStoreVersionError(
                f"Unsupported Task Center schema version: {version!r}"
            )
        try:
            return TaskCenterDocument.model_validate(raw)
        except ValidationError as error:
            raise TaskStoreReadError(
                f"Invalid Task Center data: {self.path}"
            ) from error

    def _write_document(self, document: TaskCenterDocument) -> None:
        self._ensure_safe_parent()
        temporary: Path | None = None
        try:
            data = cast(
                dict[str, object],
                _toml_ready(document.model_dump(mode="python", exclude_none=True)),
            )
            serialized = tomli_w.dumps(data).encode("utf-8")
            if len(serialized) > MAX_TASK_CENTER_FILE_BYTES:
                raise TaskStoreWriteError("Task Center file exceeds the size limit")
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            temporary = None
        except (OSError, TypeError) as error:
            raise TaskStoreWriteError(
                f"Failed to write Task Center: {self.path}"
            ) from error
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)

    def _ensure_safe_parent(self) -> None:
        parent = self.path.parent
        if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
            raise TaskStoreWriteError(f"Unsafe Task Center directory: {parent}")
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise TaskStoreWriteError(
                f"Failed to create Task Center directory: {parent}"
            ) from error
        if self.path.is_symlink():
            raise TaskStoreWriteError(f"Unsafe Task Center path: {self.path}")

    def _validate_safe_parent_for_read(self) -> None:
        parent = self.path.parent
        if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
            raise TaskStoreReadError(f"Unsafe Task Center directory: {parent}")

    def _replace_memory(
        self,
        document: TaskCenterDocument,
        *,
        clear_runtime: bool = False,
        force_notify: bool = False,
    ) -> None:
        next_tasks = {task.task_id: task for task in document.tasks}
        changed = next_tasks != self._tasks or force_notify
        self._tasks = next_tasks
        self._loaded = True
        if clear_runtime:
            changed = changed or bool(self._runtime_agent_ids)
            self._runtime_agent_ids.clear()
        else:
            self._runtime_agent_ids = {
                task_id: agent_id
                for task_id, agent_id in self._runtime_agent_ids.items()
                if task_id in next_tasks
            }
        if not changed:
            return
        self._notify_listeners()

    def _notify_listeners(self) -> None:
        snapshot = self.snapshot
        for listener in tuple(self._listeners):
            with suppress(Exception):
                listener(snapshot)

    def _with_runtime(self, task: TaskDefinition) -> TaskDefinition:
        return task.model_copy(
            update={"managed_agent_id": self._runtime_agent_ids.get(task.task_id)}
        )

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Task Center clock must return an aware datetime")
        return now.astimezone(UTC)

    @staticmethod
    def _find(document: TaskCenterDocument, task_id: str) -> TaskDefinition:
        task = next((item for item in document.tasks if item.task_id == task_id), None)
        if task is None:
            raise TaskNotFoundError(f"Unknown task: {task_id}")
        return task

    @staticmethod
    def _find_run(task: TaskDefinition, run_id: str) -> TaskRunRecord:
        run = next((item for item in task.run_history if item.run_id == run_id), None)
        if run is None:
            raise TaskConflictError(f"Unknown task run: {run_id}")
        return run

    @staticmethod
    def _replace_task(
        document: TaskCenterDocument, updated: TaskDefinition
    ) -> TaskCenterDocument:
        return TaskCenterDocument(
            tasks=tuple(
                updated if task.task_id == updated.task_id else task
                for task in document.tasks
            )
        )


def _toml_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _toml_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_toml_ready(item) for item in value]
    return value


def _prune_run_history(history: tuple[TaskRunRecord, ...]) -> tuple[TaskRunRecord, ...]:
    terminal_indexes = [
        index for index, run in enumerate(history) if run.state.is_terminal
    ]
    retained_terminal_indexes = set(terminal_indexes[-MAX_TASK_RUN_HISTORY:])
    return tuple(
        run
        for index, run in enumerate(history)
        if not run.state.is_terminal or index in retained_terminal_indexes
    )


def _task_state(run_state: TaskRunState) -> TaskState:
    return {
        TaskRunState.READY: TaskState.READY,
        TaskRunState.RETRY_PENDING: TaskState.READY,
        TaskRunState.QUEUED_FOR_APPROVAL: TaskState.QUEUED_FOR_APPROVAL,
        TaskRunState.RUNNING: TaskState.RUNNING,
        TaskRunState.BLOCKED: TaskState.BLOCKED,
        TaskRunState.COMPLETED: TaskState.COMPLETED,
        TaskRunState.FAILED: TaskState.FAILED,
    }[run_state]


def _prune_trigger_index(
    receipts: tuple[TaskTriggerReceipt, ...], *, now: datetime
) -> tuple[TaskTriggerReceipt, ...]:
    cutoff = now - timedelta(days=TASK_TRIGGER_RETENTION_DAYS)
    retained = tuple(receipt for receipt in receipts if receipt.recorded_at >= cutoff)
    return retained[-MAX_TASK_TRIGGER_INDEX:]


def _validate_claim_window(claimed_at: datetime, claim_expires_at: datetime) -> None:
    if (
        claimed_at.tzinfo is None
        or claimed_at.utcoffset() is None
        or claim_expires_at.tzinfo is None
        or claim_expires_at.utcoffset() is None
    ):
        raise ValueError("claim timestamps must include a UTC offset")
    if claim_expires_at <= claimed_at:
        raise ValueError("claim expiry must follow claim time")


def _validate_owner_token(owner_token: str) -> None:
    if not owner_token.strip() or len(owner_token) > MAX_TASK_CLAIM_OWNER_LENGTH:
        raise ValueError("claim owner token must be 1 to 200 nonblank characters")
