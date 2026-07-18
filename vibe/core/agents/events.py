from __future__ import annotations

from contextvars import ContextVar, Token

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.agents.models import ManagedAgentState
from vibe.core.types import BaseEvent

MAX_MANAGED_AGENT_TASK_CHARS = 4_000
MAX_MANAGED_AGENT_ACTIVITY_CHARS = 1_000
MAX_MANAGED_AGENT_ERROR_CHARS = 2_000
MAX_MANAGED_AGENT_RESPONSE_CHARS = 12_000
MAX_MANAGED_AGENT_ID_CHARS = 128
MAX_MANAGED_AGENT_PROFILE_CHARS = 256
MAX_MANAGED_AGENT_SESSION_ID_CHARS = 256


class ManagedAgentLifecycleEvent(BaseEvent):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    agent_id: str = Field(min_length=1, max_length=MAX_MANAGED_AGENT_ID_CHARS)
    profile: str = Field(min_length=1, max_length=MAX_MANAGED_AGENT_PROFILE_CHARS)
    agent_display_name: str = Field(
        min_length=1, max_length=MAX_MANAGED_AGENT_PROFILE_CHARS
    )
    parent_session_id: str = Field(
        min_length=1, max_length=MAX_MANAGED_AGENT_SESSION_ID_CHARS
    )
    child_session_id: str = Field(
        min_length=1, max_length=MAX_MANAGED_AGENT_SESSION_ID_CHARS
    )
    state: ManagedAgentState
    current_activity: str | None = Field(
        default=None, max_length=MAX_MANAGED_AGENT_ACTIVITY_CHARS
    )
    queued_messages: int = Field(default=0, ge=0)


class ManagedAgentCallbackContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1, max_length=MAX_MANAGED_AGENT_ID_CHARS)
    profile: str = Field(min_length=1, max_length=MAX_MANAGED_AGENT_PROFILE_CHARS)


_MANAGED_AGENT_CALLBACK_CONTEXT: ContextVar[ManagedAgentCallbackContext | None] = (
    ContextVar("managed_agent_callback_context", default=None)
)


def get_managed_agent_callback_context() -> ManagedAgentCallbackContext | None:
    return _MANAGED_AGENT_CALLBACK_CONTEXT.get()


def _set_managed_agent_callback_context(
    context: ManagedAgentCallbackContext,
) -> Token[ManagedAgentCallbackContext | None]:
    return _MANAGED_AGENT_CALLBACK_CONTEXT.set(context)


def _reset_managed_agent_callback_context(
    token: Token[ManagedAgentCallbackContext | None],
) -> None:
    _MANAGED_AGENT_CALLBACK_CONTEXT.reset(token)
