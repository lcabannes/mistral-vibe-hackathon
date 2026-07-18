from __future__ import annotations

from datetime import UTC, datetime, time
from enum import IntEnum, StrEnum, auto
from typing import Annotated, Literal, final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vibe.core.types import BaseEvent

TASK_CENTER_SCHEMA_VERSION = 1
MAX_TASK_TITLE_LENGTH = 200
MAX_TASK_DETAILS_LENGTH = 10_000
MAX_TASK_ERROR_LENGTH = 2_000
MAX_TASK_CLAIM_OWNER_LENGTH = 200
MAX_TASK_RUN_HISTORY = 20
MAX_TASK_TRIGGER_INDEX = 1_024
TASK_TRIGGER_RETENTION_DAYS = 30
TASK_ID_PATTERN = r"^task_[a-f0-9]{32}$"
RUN_ID_PATTERN = r"^run_[a-f0-9]{32}$"
PROFILE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
EVENT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _nonblank(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("value must not be blank")
    return stripped


def _optional_nonblank(value: str | None) -> str | None:
    return None if value is None else _nonblank(value)


def _optional_details(value: str | None) -> str | None:
    return None if value is None else value.strip()


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    return value.astimezone(UTC)


def _optional_aware_utc(value: datetime | None) -> datetime | None:
    return None if value is None else _aware_utc(value)


def _local_time(value: time) -> time:
    if value.tzinfo is not None:
        raise ValueError("local schedule time must not include a UTC offset")
    return value


class TaskState(StrEnum):
    IDLE = auto()
    READY = auto()
    QUEUED_FOR_APPROVAL = auto()
    RUNNING = auto()
    BLOCKED = auto()
    COMPLETED = auto()
    FAILED = auto()


class TaskTriggerKind(StrEnum):
    MANUAL = auto()
    APP_START = auto()
    SESSION_START = auto()
    INTERVAL = auto()
    DAILY = auto()
    WEEKLY = auto()


class Weekday(IntEnum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


class _TriggerModel(_StrictModel):
    pass


@final
class ManualTrigger(_TriggerModel):
    kind: Literal[TaskTriggerKind.MANUAL] = TaskTriggerKind.MANUAL


@final
class AppStartTrigger(_TriggerModel):
    kind: Literal[TaskTriggerKind.APP_START] = TaskTriggerKind.APP_START


@final
class SessionStartTrigger(_TriggerModel):
    kind: Literal[TaskTriggerKind.SESSION_START] = TaskTriggerKind.SESSION_START


@final
class IntervalTrigger(_TriggerModel):
    kind: Literal[TaskTriggerKind.INTERVAL] = TaskTriggerKind.INTERVAL
    interval_seconds: float = Field(gt=0, le=31_536_000)
    anchor_at: datetime | None = None

    _anchor_at = field_validator("anchor_at")(_optional_aware_utc)


class _WallClockTrigger(_TriggerModel):
    at: time
    timezone: str = Field(min_length=1, max_length=128)

    _at = field_validator("at")(_local_time)

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        value = _nonblank(value)
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Unknown timezone: {value}") from error
        return value


@final
class DailyTrigger(_WallClockTrigger):
    kind: Literal[TaskTriggerKind.DAILY] = TaskTriggerKind.DAILY


@final
class WeeklyTrigger(_WallClockTrigger):
    kind: Literal[TaskTriggerKind.WEEKLY] = TaskTriggerKind.WEEKLY
    weekdays: tuple[Weekday, ...] = Field(min_length=1, max_length=7)

    @field_validator("weekdays")
    @classmethod
    def unique_weekdays(cls, value: tuple[Weekday, ...]) -> tuple[Weekday, ...]:
        if len(set(value)) != len(value):
            raise ValueError("weekdays must not contain duplicates")
        return tuple(sorted(value))


type TaskTrigger = Annotated[
    ManualTrigger
    | AppStartTrigger
    | SessionStartTrigger
    | IntervalTrigger
    | DailyTrigger
    | WeeklyTrigger,
    Field(discriminator="kind"),
]


class TaskRunState(StrEnum):
    READY = auto()
    RETRY_PENDING = auto()
    QUEUED_FOR_APPROVAL = auto()
    RUNNING = auto()
    BLOCKED = auto()
    COMPLETED = auto()
    FAILED = auto()

    @property
    def is_terminal(self) -> bool:
        return self in {
            TaskRunState.BLOCKED,
            TaskRunState.COMPLETED,
            TaskRunState.FAILED,
        }


class TaskExecutionAuthorization(StrEnum):
    ASK = auto()
    ALWAYS = auto()


class TaskRunRecord(_StrictModel):
    run_id: str = Field(pattern=RUN_ID_PATTERN)
    trigger_instance_id: str = Field(min_length=1, max_length=200)
    trigger_kind: TaskTriggerKind
    state: TaskRunState
    authorization: TaskExecutionAuthorization = TaskExecutionAuthorization.ASK
    scheduled_for: datetime | None = None
    triggered_at: datetime
    coalesced: bool = False
    error: str | None = Field(default=None, max_length=MAX_TASK_ERROR_LENGTH)
    claim_owner: str | None = Field(
        default=None, min_length=1, max_length=MAX_TASK_CLAIM_OWNER_LENGTH
    )
    claim_expires_at: datetime | None = None

    _scheduled_for = field_validator("scheduled_for")(_optional_aware_utc)
    _triggered_at = field_validator("triggered_at")(_aware_utc)
    _claim_expires_at = field_validator("claim_expires_at")(_optional_aware_utc)

    @model_validator(mode="after")
    def validate_claim(self) -> TaskRunRecord:
        if (self.claim_owner is None) != (self.claim_expires_at is None):
            raise ValueError("claim owner and expiry must be set together")
        if self.claim_owner is not None and self.state not in {
            TaskRunState.READY,
            TaskRunState.RETRY_PENDING,
        }:
            raise ValueError("only pending task runs can retain a dispatch claim")
        return self


class TaskTriggerReceipt(_StrictModel):
    trigger_instance_id: str = Field(min_length=1, max_length=200)
    recorded_at: datetime

    _recorded_at = field_validator("recorded_at")(_aware_utc)


class TaskDefinition(_StrictModel):
    task_id: str = Field(pattern=TASK_ID_PATTERN)
    title: str = Field(min_length=1, max_length=MAX_TASK_TITLE_LENGTH)
    details: str = Field(default="", max_length=MAX_TASK_DETAILS_LENGTH)
    state: TaskState = TaskState.IDLE
    enabled: bool = True
    assigned_profile: str | None = Field(default=None, pattern=PROFILE_PATTERN)
    managed_agent_id: str | None = Field(default=None, exclude=True, min_length=1)
    trigger: TaskTrigger = Field(default_factory=ManualTrigger)
    created_at: datetime
    updated_at: datetime
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    last_error: str | None = Field(default=None, max_length=MAX_TASK_ERROR_LENGTH)
    trigger_index: tuple[TaskTriggerReceipt, ...] = Field(
        default=(), max_length=MAX_TASK_TRIGGER_INDEX
    )
    run_history: tuple[TaskRunRecord, ...] = ()

    _title = field_validator("title")(_nonblank)
    _details = field_validator("details")(lambda value: value.strip())
    _assigned_profile = field_validator("assigned_profile")(_optional_nonblank)
    _created_at = field_validator("created_at")(_aware_utc)
    _updated_at = field_validator("updated_at")(_aware_utc)
    _last_run_at = field_validator("last_run_at")(_optional_aware_utc)
    _next_run_at = field_validator("next_run_at")(_optional_aware_utc)

    @model_validator(mode="before")
    @classmethod
    def populate_legacy_trigger_index(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        runs = value.get("run_history")
        fallback_timestamp = value.get("updated_at")
        history_timestamps: dict[str, object] = {}
        if isinstance(runs, (list, tuple)):
            for run in runs:
                if isinstance(run, TaskRunRecord):
                    history_timestamps[run.trigger_instance_id] = run.triggered_at
                    continue
                if isinstance(run, dict) and isinstance(
                    trigger_instance_id := run.get("trigger_instance_id"), str
                ):
                    history_timestamps[trigger_instance_id] = run.get(
                        "triggered_at", fallback_timestamp
                    )
        raw_index = value.get("trigger_index", tuple(history_timestamps))
        if not isinstance(raw_index, (list, tuple)):
            return value
        trigger_index: list[object] = []
        for receipt in raw_index[-MAX_TASK_TRIGGER_INDEX:]:
            if isinstance(receipt, str):
                trigger_index.append({
                    "trigger_instance_id": receipt,
                    "recorded_at": history_timestamps.get(receipt, fallback_timestamp),
                })
                continue
            trigger_index.append(receipt)
        updated = dict(value)
        updated["trigger_index"] = tuple(trigger_index)
        return updated

    @model_validator(mode="after")
    def validate_timestamps(self) -> TaskDefinition:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        trigger_ids = [receipt.trigger_instance_id for receipt in self.trigger_index]
        if len(trigger_ids) != len(set(trigger_ids)):
            raise ValueError("trigger index entries must be unique")
        run_ids = [run.run_id for run in self.run_history]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("task run ids must be unique")
        return self

    @property
    def active_run(self) -> TaskRunRecord | None:
        return next(
            (run for run in reversed(self.run_history) if not run.state.is_terminal),
            None,
        )


class TaskCreate(_StrictModel):
    title: str = Field(min_length=1, max_length=MAX_TASK_TITLE_LENGTH)
    details: str = Field(default="", max_length=MAX_TASK_DETAILS_LENGTH)
    enabled: bool = True
    assigned_profile: str | None = Field(default=None, pattern=PROFILE_PATTERN)
    trigger: TaskTrigger = Field(default_factory=ManualTrigger)

    _title = field_validator("title")(_nonblank)
    _details = field_validator("details")(lambda value: value.strip())
    _assigned_profile = field_validator("assigned_profile")(_optional_nonblank)


class TaskUpdate(_StrictModel):
    title: str | None = Field(
        default=None, min_length=1, max_length=MAX_TASK_TITLE_LENGTH
    )
    details: str | None = Field(default=None, max_length=MAX_TASK_DETAILS_LENGTH)
    state: TaskState | None = None
    enabled: bool | None = None
    assigned_profile: str | None = Field(default=None, pattern=PROFILE_PATTERN)
    trigger: TaskTrigger | None = None

    _title = field_validator("title")(_optional_nonblank)
    _details = field_validator("details")(_optional_details)
    _assigned_profile = field_validator("assigned_profile")(_optional_nonblank)

    @model_validator(mode="after")
    def require_change(self) -> TaskUpdate:
        if not self.model_fields_set:
            raise ValueError("at least one task field must be supplied")
        if "title" in self.model_fields_set and self.title is None:
            raise ValueError("title must not be null")
        if "details" in self.model_fields_set and self.details is None:
            raise ValueError("details must not be null")
        if "state" in self.model_fields_set and self.state is None:
            raise ValueError("state must not be null")
        if "enabled" in self.model_fields_set and self.enabled is None:
            raise ValueError("enabled must not be null")
        if "trigger" in self.model_fields_set and self.trigger is None:
            raise ValueError("trigger must not be null")
        return self


class TaskCenterDocument(_StrictModel):
    schema_version: Literal[1] = TASK_CENTER_SCHEMA_VERSION
    tasks: tuple[TaskDefinition, ...] = ()

    @model_validator(mode="after")
    def unique_task_ids(self) -> TaskCenterDocument:
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task ids must be unique")
        return self


class TaskEventKind(StrEnum):
    APP_START = auto()
    SESSION_START = auto()


class TaskSourceEvent(_StrictModel):
    event_id: str = Field(pattern=EVENT_ID_PATTERN)
    kind: TaskEventKind
    occurred_at: datetime

    _occurred_at = field_validator("occurred_at")(_aware_utc)


class TaskTriggeredEvent(BaseEvent):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=TASK_ID_PATTERN)
    run_id: str = Field(pattern=RUN_ID_PATTERN)
    trigger_instance_id: str = Field(min_length=1, max_length=200)
    trigger_kind: TaskTriggerKind
    title: str = Field(min_length=1, max_length=MAX_TASK_TITLE_LENGTH)
    details: str = Field(default="", max_length=MAX_TASK_DETAILS_LENGTH)
    assigned_profile: str | None = Field(default=None, pattern=PROFILE_PATTERN)
    scheduled_for: datetime | None = None
    triggered_at: datetime
    coalesced: bool = False

    _scheduled_for = field_validator("scheduled_for")(_optional_aware_utc)
    _triggered_at = field_validator("triggered_at")(_aware_utc)
