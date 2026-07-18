from __future__ import annotations

from enum import StrEnum, auto
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vibe.core.task_center.models import (
    MAX_TASK_DETAILS_LENGTH,
    MAX_TASK_ERROR_LENGTH,
    MAX_TASK_TITLE_LENGTH,
    PROFILE_PATTERN,
    RUN_ID_PATTERN,
    TASK_ID_PATTERN,
    TaskExecutionAuthorization,
    TaskTriggerKind,
)


class TaskExecutionDisposition(StrEnum):
    QUEUED_FOR_APPROVAL = auto()
    STARTED = auto()
    BLOCKED = auto()


class _StrictExecutionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskExecutionRequest(_StrictExecutionModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=TASK_ID_PATTERN)
    run_id: str = Field(pattern=RUN_ID_PATTERN)
    title: str = Field(min_length=1, max_length=MAX_TASK_TITLE_LENGTH)
    details: str = Field(default="", max_length=MAX_TASK_DETAILS_LENGTH)
    assigned_profile: str | None = Field(default=None, pattern=PROFILE_PATTERN)
    trigger_kind: TaskTriggerKind
    requested_authorization: TaskExecutionAuthorization = TaskExecutionAuthorization.ASK


class TaskExecutionResult(_StrictExecutionModel):
    disposition: TaskExecutionDisposition
    authorization: TaskExecutionAuthorization = TaskExecutionAuthorization.ASK
    managed_agent_id: str | None = Field(default=None, min_length=1, max_length=200)
    error: str | None = Field(
        default=None, min_length=1, max_length=MAX_TASK_ERROR_LENGTH
    )

    @field_validator("managed_agent_id", "error")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @model_validator(mode="after")
    def validate_disposition(self) -> TaskExecutionResult:
        if (
            self.disposition is TaskExecutionDisposition.STARTED
            and self.authorization is not TaskExecutionAuthorization.ALWAYS
        ):
            raise ValueError(
                "automatic task start requires explicit ALWAYS authorization"
            )
        if (
            self.disposition is TaskExecutionDisposition.QUEUED_FOR_APPROVAL
            and self.authorization is not TaskExecutionAuthorization.ASK
        ):
            raise ValueError("approval queue must use ASK authorization")
        if self.disposition is TaskExecutionDisposition.BLOCKED and self.error is None:
            raise ValueError("blocked task execution requires an error")
        if (
            self.managed_agent_id is not None
            and self.disposition is not TaskExecutionDisposition.STARTED
        ):
            raise ValueError("managed_agent_id is only valid for a started task")
        return self


class TaskExecutionPort(Protocol):
    def is_profile_available(self, profile: str) -> bool: ...

    async def handoff(self, request: TaskExecutionRequest) -> TaskExecutionResult: ...
