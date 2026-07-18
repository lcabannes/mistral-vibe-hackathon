from __future__ import annotations

from collections.abc import AsyncGenerator
from enum import StrEnum, auto
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vibe.core.agents.models import ManagedAgentSnapshot
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.permissions import PermissionContext
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolCallEvent, ToolResultEvent, ToolStreamEvent

if TYPE_CHECKING:
    from vibe.core.config import AnyVibeConfig


class ManageAgentsAction(StrEnum):
    START = auto()
    LIST = auto()
    MESSAGE = auto()
    OUTPUT = auto()
    STOP = auto()


class ManageAgentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ManageAgentsAction
    profile: str | None = Field(
        default=None, description="Agent profile to launch, such as default or explore"
    )
    task: str | None = Field(default=None, description="Initial task for a new agent")
    name: str | None = Field(
        default=None, description="Optional stable agent id prefix"
    )
    agent_id: str | None = Field(default=None, description="Managed agent id")
    message: str | None = Field(
        default=None, description="Follow-up message for an agent"
    )

    @field_validator("profile", "task", "name", "agent_id", "message")
    @classmethod
    def trim_optional_value(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @model_validator(mode="after")
    def validate_action_fields(self) -> ManageAgentsArgs:
        provided = {
            name
            for name in ("profile", "task", "name", "agent_id", "message")
            if getattr(self, name) is not None
        }
        required: dict[ManageAgentsAction, set[str]] = {
            ManageAgentsAction.START: {"profile", "task"},
            ManageAgentsAction.LIST: set(),
            ManageAgentsAction.MESSAGE: {"agent_id", "message"},
            ManageAgentsAction.OUTPUT: {"agent_id"},
            ManageAgentsAction.STOP: {"agent_id"},
        }
        allowed = required[self.action] | (
            {"name"} if self.action is ManageAgentsAction.START else set()
        )
        if missing := required[self.action] - provided:
            fields = " and ".join(sorted(missing))
            raise ValueError(f"{self.action.value} requires {fields}")
        if unexpected := provided - allowed:
            fields = ", ".join(sorted(unexpected))
            raise ValueError(f"{self.action.value} does not accept field(s): {fields}")
        return self


class ManageAgentsResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str = Field(min_length=1)
    agents: list[ManagedAgentSnapshot] = Field(default_factory=list)
    available_profiles: list[str] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def trim_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped

    @field_validator("available_profiles")
    @classmethod
    def validate_profiles(cls, values: list[str]) -> list[str]:
        profiles = [value.strip() for value in values]
        if any(not value for value in profiles):
            raise ValueError("available profiles must not contain blank values")
        if len(profiles) != len(set(profiles)):
            raise ValueError("available profiles must be unique")
        return profiles


class ManageAgentsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class ManageAgents(
    BaseTool[ManageAgentsArgs, ManageAgentsResult, ManageAgentsConfig, BaseToolState],
    ToolUIData[ManageAgentsArgs, ManageAgentsResult],
):
    @classmethod
    def is_available(cls, config: AnyVibeConfig | None = None) -> bool:
        return bool(
            config
            and config.enable_orchestrator_controls
            and config.enable_agent_management
        )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if isinstance(event.args, ManageAgentsArgs):
            return ToolCallDisplay(summary=f"Agent control: {event.args.action.value}")
        return ToolCallDisplay(summary="Agent control")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, ManageAgentsResult):
            return ToolResultDisplay(success=True, message=event.result.message)
        return ToolResultDisplay(success=True, message="Agent control updated")

    @classmethod
    def get_status_text(cls) -> str:
        return "Managing agents"

    def resolve_permission(self, args: ManageAgentsArgs) -> PermissionContext | None:
        if args.action is ManageAgentsAction.START:
            return None
        return PermissionContext(permission=ToolPermission.ALWAYS)

    async def run(
        self, args: ManageAgentsArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | ManageAgentsResult, None]:
        if ctx is None or ctx.agent_management is None:
            raise ToolError("Managed agents are not available on this surface")
        management = ctx.agent_management

        try:
            match args.action:
                case ManageAgentsAction.START:
                    agent = await management.start(
                        cast(str, args.profile), cast(str, args.task), name=args.name
                    )
                    result = ManageAgentsResult(
                        message=f"Started {agent.agent_id}", agents=[agent]
                    )
                case ManageAgentsAction.LIST:
                    agents = list(management.list())
                    result = ManageAgentsResult(
                        message=f"{len(agents)} managed agents",
                        agents=agents,
                        available_profiles=list(management.available_profiles()),
                    )
                case ManageAgentsAction.MESSAGE:
                    agent = await management.message(
                        cast(str, args.agent_id), cast(str, args.message)
                    )
                    result = ManageAgentsResult(
                        message=f"Queued message for {agent.agent_id}", agents=[agent]
                    )
                case ManageAgentsAction.OUTPUT:
                    agent = management.output(cast(str, args.agent_id))
                    result = ManageAgentsResult(
                        message=f"Output for {agent.agent_id}", agents=[agent]
                    )
                case ManageAgentsAction.STOP:
                    agent = await management.stop(cast(str, args.agent_id))
                    result = ManageAgentsResult(
                        message=f"Stopped {agent.agent_id}", agents=[agent]
                    )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

        yield result
