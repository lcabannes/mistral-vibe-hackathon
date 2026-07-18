from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.agents.models import ManagedAgentSnapshot, ManagedAgentState
from vibe.core.team_workspace.models import TeamWorkspaceSnapshot


class AgentRoomConversationMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str = ""
    client_message_id: str | None = None
    role: str
    content: str = ""
    status: str = "succeeded"
    created_at: float = 0.0
    updated_at: float = 0.0
    error_code: str | None = None


class AgentRoomProfile(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    name: str
    display_name: str | None = None
    description: str = ""


class AgentRoomTeamWorkspace(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    snapshot: TeamWorkspaceSnapshot
    local_member_id: str
    local_agent_links: dict[str, str] = Field(default_factory=dict)
    published_at: float = 0.0


class AgentRoomRun(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    tool_call_id: str = Field(min_length=1)
    parent_session_id: str | None = None
    child_session_id: str | None = None
    agent_name: str
    agent_display_name: str
    task: str
    state: str
    started_at: float = 0.0
    updated_at: float = 0.0
    current_activity: str | None = None
    turns_used: int = 0
    usage: dict[str, int] = Field(default_factory=dict)
    context_tokens: int = 0
    context_limit: int | None = None
    estimated_cost_usd: float = 0.0
    model: str | None = None
    is_primary: bool = False
    is_orchestrator: bool = False
    group_id: str = "unassigned"
    runtime_live: bool = False
    resumable: bool = True
    conversation: tuple[AgentRoomConversationMessage, ...] = ()
    approvals: tuple[dict[str, Any], ...] = ()
    questions: tuple[dict[str, Any], ...] = ()
    error: str | None = None
    queued_messages: int = 0
    worktree_path: str | None = None
    branch: str | None = None
    worktree_dirty: bool = False
    merge_status: str | None = None

    @property
    def last_response(self) -> str:
        return next(
            (
                message.content
                for message in reversed(self.conversation)
                if message.role == "assistant"
            ),
            "",
        )

    def managed_snapshot(self) -> ManagedAgentSnapshot:
        state = {
            "requested": ManagedAgentState.STARTING,
            "running": ManagedAgentState.RUNNING,
            "working": ManagedAgentState.WORKING,
            "attention": ManagedAgentState.ATTENTION,
            "idle": ManagedAgentState.IDLE,
            "failed": ManagedAgentState.FAILED,
            "completed": ManagedAgentState.STOPPED,
            "cancelled": ManagedAgentState.STOPPED,
            "stopped": ManagedAgentState.STOPPED,
        }.get(self.state, ManagedAgentState.FAILED)
        return ManagedAgentSnapshot(
            agent_id=self.tool_call_id,
            child_session_id=self.child_session_id or self.tool_call_id,
            profile=self.agent_name,
            state=state,
            task=self.task,
            current_activity=self.current_activity,
            last_response=self.last_response,
            error=self.error,
            queued_messages=self.queued_messages,
            started_at=self.started_at,
            updated_at=self.updated_at,
            turns_used=self.turns_used,
            prompt_tokens=int(self.usage.get("prompt_tokens", 0)),
            completion_tokens=int(self.usage.get("completion_tokens", 0)),
            context_tokens=self.context_tokens,
            context_limit=self.context_limit,
            estimated_cost_usd=self.estimated_cost_usd,
            model=self.model,
        )


class AgentRoomSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    api_version: int = 1
    instance_id: str = ""
    revision: int = 0
    connected: bool = True
    workspace: dict[str, Any] = Field(default_factory=dict)
    activities: tuple[AgentRoomRun, ...] = ()
    profiles: tuple[AgentRoomProfile, ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    coordination: dict[str, Any] = Field(default_factory=dict)
    network: dict[str, Any] = Field(default_factory=dict)
    team_workspace: AgentRoomTeamWorkspace | None = None
