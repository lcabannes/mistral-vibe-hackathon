from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
import tempfile
import tomllib
from typing import cast
from uuid import uuid4

from pydantic import ValidationError
import tomli_w

from vibe.core.task_center.models import (
    MAX_TASK_ERROR_LENGTH,
    MAX_TASK_RUN_HISTORY,
    TASK_CENTER_SCHEMA_VERSION,
    TaskCenterDocument,
    TaskCreate,
    TaskDefinition,
    TaskExecutionAuthorization,
    TaskRunRecord,
    TaskRunState,
    TaskState,
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
        self, task_id: str, run: TaskRunRecord, *, next_run_at: datetime | None
    ) -> TaskTriggerRecordResult:
        created = False

        def mutate(document: TaskCenterDocument) -> TaskCenterDocument:
            nonlocal created
            current = self._find(document, task_id)
            if any(
                item.trigger_instance_id == run.trigger_instance_id
                for item in current.run_history
            ):
                return document
            created = True
            history = (*current.run_history, run)[-MAX_TASK_RUN_HISTORY:]
            updated = current.model_copy(
                update={
                    "state": TaskState.READY,
                    "updated_at": run.triggered_at,
                    "last_run_at": run.triggered_at,
                    "next_run_at": next_run_at,
                    "last_error": None,
                    "run_history": history,
                }
            )
            return self._replace_task(document, updated)

        await self._mutate(mutate, write_when_unchanged=False)
        return TaskTriggerRecordResult(task=await self.get(task_id), created=created)

    async def record_handoff(
        self,
        task_id: str,
        run_id: str,
        *,
        state: TaskRunState,
        authorization: TaskExecutionAuthorization,
        error: str | None = None,
        managed_agent_id: str | None = None,
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
                found = True
                history.append(
                    run.model_copy(
                        update={
                            "state": state,
                            "authorization": authorization,
                            "error": bounded_error,
                        }
                    )
                )
            if not found:
                raise TaskConflictError(f"Unknown task run: {run_id}")
            task_state = {
                TaskRunState.READY: TaskState.READY,
                TaskRunState.QUEUED_FOR_APPROVAL: TaskState.QUEUED_FOR_APPROVAL,
                TaskRunState.RUNNING: TaskState.RUNNING,
                TaskRunState.BLOCKED: TaskState.BLOCKED,
                TaskRunState.COMPLETED: TaskState.COMPLETED,
                TaskRunState.FAILED: TaskState.FAILED,
            }[state]
            updated = current.model_copy(
                update={
                    "state": task_state,
                    "updated_at": now,
                    "last_error": bounded_error,
                    "run_history": tuple(history),
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
            document = await asyncio.to_thread(self._read_document)
            updated = mutation(document)
            if write_when_unchanged or updated != document:
                await asyncio.to_thread(self._write_document, updated)
            if memory_update is not None:
                memory_update()
            self._replace_memory(updated, force_notify=memory_update is not None)

    def _read_document(self) -> TaskCenterDocument:
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
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                data = cast(
                    dict[str, object],
                    _toml_ready(document.model_dump(mode="python", exclude_none=True)),
                )
                tomli_w.dump(data, handle)
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
