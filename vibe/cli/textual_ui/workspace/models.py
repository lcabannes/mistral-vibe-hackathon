from __future__ import annotations

from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict

from vibe.core.types import LLMUsage


class WorkspaceView(StrEnum):
    HOME = auto()
    CHAT = auto()
    OFFICE = auto()
    AGENTS = auto()
    MCP = auto()
    USAGE = auto()
    COWORKERS = auto()


class AgentRunState(StrEnum):
    IDLE = auto()
    REQUESTED = auto()
    RUNNING = auto()
    WORKING = auto()
    ATTENTION = auto()
    FAILED = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    STOPPED = auto()

    @property
    def is_terminal(self) -> bool:
        return self in {self.FAILED, self.COMPLETED, self.CANCELLED, self.STOPPED}


class AgentActivity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_call_id: str
    parent_session_id: str
    agent_name: str
    agent_display_name: str
    task: str
    state: AgentRunState
    started_at: float
    updated_at: float
    child_session_id: str | None = None
    current_activity: str | None = None
    turns_used: int | None = None
    usage: LLMUsage | None = None
    is_primary: bool = False
    owner_display_name: str | None = None
    branch: str | None = None
    managed_agent_id: str | None = None
    event_sequence: int | None = None
    queued_messages: int = 0
    last_response: str = ""
    error: str | None = None

    @property
    def is_managed(self) -> bool:
        return self.managed_agent_id is not None

    @property
    def activity_id(self) -> str:
        if self.is_primary:
            return f"primary:{self.parent_session_id}"
        if self.managed_agent_id is not None:
            return f"managed:{self.managed_agent_id}"
        return f"task:{self.tool_call_id}"


class AgentActivitySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    activities: tuple[AgentActivity, ...] = ()


class _TaskCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task: str
    agent: str


class _TaskOutcome(BaseModel):
    model_config = ConfigDict(extra="ignore")

    turns_used: int
    completed: bool
