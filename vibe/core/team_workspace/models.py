from __future__ import annotations

from datetime import datetime
from enum import StrEnum, auto
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vibe.core.team_workspace.privacy import (
    MAX_SHARED_MESSAGE_LENGTH,
    sanitize_shared_message,
)

SCHEMA_VERSION = 1
MAX_LABEL_LENGTH = 80
MAX_RUNS_PER_SNAPSHOT = 100
MAX_HISTORY_PER_RUN = 200
_ID_PATTERN = r"^[a-z][a-z0-9_-]{7,79}$"


class PrivacyMode(StrEnum):
    STATUS = auto()
    SUMMARIES = auto()


class HistoryScope(StrEnum):
    STATUS = auto()
    MARKERS = auto()
    MESSAGES = auto()


class ConversationRole(StrEnum):
    USER = auto()
    ASSISTANT = auto()


class ConnectionState(StrEnum):
    DISABLED = auto()
    DISCONNECTED = auto()
    CONNECTED = auto()
    DEGRADED = auto()


class PresenceState(StrEnum):
    ONLINE = auto()
    OFFLINE = auto()


class ActivityState(StrEnum):
    IDLE = auto()
    REQUESTED = auto()
    RUNNING = auto()
    WORKING = auto()
    ATTENTION = auto()
    FAILED = auto()
    COMPLETED = auto()
    CANCELLED = auto()

    @property
    def is_terminal(self) -> bool:
        return self in {self.FAILED, self.COMPLETED, self.CANCELLED}


class ActivitySummary(StrEnum):
    STARTING = auto()
    THINKING = auto()
    USING_TOOL = auto()
    WAITING_FOR_APPROVAL = auto()
    WAITING_FOR_INPUT = auto()
    FINISHED = auto()
    FAILED = auto()
    CANCELLED = auto()


class SyncError(StrEnum):
    INVALID_ROOT = auto()
    READ_FAILED = auto()
    WRITE_FAILED = auto()
    MANIFEST_MISMATCH = auto()
    TRANSPORT_FAILED = auto()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    return value


def _validate_optional_utc(value: datetime | None) -> datetime | None:
    return None if value is None else _validate_utc(value)


def _validate_label(value: str) -> str:
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > MAX_LABEL_LENGTH:
        raise ValueError(f"label must contain 1-{MAX_LABEL_LENGTH} characters")
    if any(character in normalized for character in ("/", "\\", "\x00")):
        raise ValueError("label must not contain path separators")
    return normalized


class TeamWorkspaceIdentity(_StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    workspace_id: str = Field(pattern=_ID_PATTERN)
    project_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    display_name: str

    _display_name = field_validator("display_name")(_validate_label)


class TeamWorkspaceManifest(_StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    identity: TeamWorkspaceIdentity
    privacy_mode: PrivacyMode
    history_scope: HistoryScope = HistoryScope.STATUS
    history_limit: int = Field(default=50, ge=1, le=MAX_HISTORY_PER_RUN)
    created_at: datetime

    _created_at = field_validator("created_at")(_validate_utc)


class TeamMemberPresence(_StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    workspace_id: str = Field(pattern=_ID_PATTERN)
    member_id: str = Field(pattern=_ID_PATTERN)
    member_display_name: str
    client_id: str = Field(pattern=_ID_PATTERN)
    branch: str | None = Field(default=None, max_length=160)
    revision: int = Field(ge=1)
    last_seen_at: datetime

    _display_name = field_validator("member_display_name")(_validate_label)
    _last_seen_at = field_validator("last_seen_at")(_validate_utc)

    @field_validator("branch")
    @classmethod
    def validate_branch(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if (
            not normalized
            or normalized.startswith(("/", "."))
            or ".." in normalized
            or any(character in normalized for character in ("\\", "\x00", "\n"))
        ):
            raise ValueError("invalid branch name")
        return normalized


class TeamActivityEvent(_StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    workspace_id: str = Field(pattern=_ID_PATTERN)
    event_id: str = Field(pattern=_ID_PATTERN)
    member_id: str = Field(pattern=_ID_PATTERN)
    member_display_name: str
    client_id: str = Field(pattern=_ID_PATTERN)
    sequence: int = Field(ge=1)
    run_id: str = Field(pattern=_ID_PATTERN)
    agent_name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    agent_display_name: str
    state: ActivityState
    privacy_mode: PrivacyMode
    summary: ActivitySummary | None = None
    started_at: datetime | None = None
    occurred_at: datetime

    _member_display_name = field_validator("member_display_name")(_validate_label)
    _agent_display_name = field_validator("agent_display_name")(_validate_label)
    _started_at = field_validator("started_at")(_validate_optional_utc)
    _occurred_at = field_validator("occurred_at")(_validate_utc)

    @model_validator(mode="after")
    def status_mode_has_no_summary(self) -> TeamActivityEvent:
        if self.privacy_mode is PrivacyMode.STATUS and self.summary is not None:
            raise ValueError("status privacy mode cannot include activity summaries")
        return self


class TeamConversationEntry(_StrictModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    workspace_id: str = Field(pattern=_ID_PATTERN)
    entry_id: str = Field(pattern=_ID_PATTERN)
    member_id: str = Field(pattern=_ID_PATTERN)
    client_id: str = Field(pattern=_ID_PATTERN)
    sequence: int = Field(ge=1)
    run_id: str = Field(pattern=_ID_PATTERN)
    role: ConversationRole
    history_scope: HistoryScope
    text: str | None = Field(default=None, max_length=MAX_SHARED_MESSAGE_LENGTH)
    occurred_at: datetime

    _occurred_at = field_validator("occurred_at")(_validate_utc)

    @field_validator("text")
    @classmethod
    def sanitize_text(cls, value: str | None) -> str | None:
        return None if value is None else sanitize_shared_message(value)

    @model_validator(mode="after")
    def validate_history_scope(self) -> TeamConversationEntry:
        if self.history_scope is HistoryScope.STATUS:
            raise ValueError("status history scope cannot publish conversation entries")
        if self.history_scope is HistoryScope.MARKERS and self.text is not None:
            raise ValueError("marker history scope cannot include text")
        if self.history_scope is HistoryScope.MESSAGES and self.text is None:
            raise ValueError("message history scope requires sanitized text")
        return self


class TeamRunSnapshot(_StrictModel):
    run_id: str = Field(pattern=_ID_PATTERN)
    member_id: str = Field(pattern=_ID_PATTERN)
    member_display_name: str
    client_id: str = Field(pattern=_ID_PATTERN)
    agent_name: str
    agent_display_name: str
    state: ActivityState
    summary: ActivitySummary | None = None
    started_at: datetime
    updated_at: datetime
    sequence: int = Field(ge=1)
    history: tuple[TeamConversationEntry, ...] = Field(
        default=(), max_length=MAX_HISTORY_PER_RUN
    )

    _started_at = field_validator("started_at")(_validate_utc)
    _updated_at = field_validator("updated_at")(_validate_utc)


class TeamMemberSnapshot(_StrictModel):
    member_id: str = Field(pattern=_ID_PATTERN)
    display_name: str
    presence: PresenceState
    branch: str | None = None
    last_seen_at: datetime
    client_count: int = Field(ge=1)
    active_run_count: int = Field(ge=0)

    _last_seen_at = field_validator("last_seen_at")(_validate_utc)


class TeamWorkspaceSnapshot(_StrictModel):
    identity: TeamWorkspaceIdentity
    privacy_mode: PrivacyMode
    history_scope: HistoryScope = HistoryScope.STATUS
    connection_state: ConnectionState
    generated_at: datetime
    members: tuple[TeamMemberSnapshot, ...] = ()
    runs: tuple[TeamRunSnapshot, ...] = Field(
        default=(), max_length=MAX_RUNS_PER_SNAPSHOT
    )
    error: SyncError | None = None

    _generated_at = field_validator("generated_at")(_validate_utc)
