from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vibe.core.control_port import (
    CLICommandRequest,
    CLIControlAction,
    CLIControlRequest,
    CLIControlResult,
    CLINavigateWorkspaceRequest,
    CLISwitchAgentRequest,
    WorkspaceDestination,
)
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolCallEvent, ToolResultEvent, ToolStreamEvent

if TYPE_CHECKING:
    from vibe.core.config import AnyVibeConfig


class ControlCLIArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: CLIControlAction
    value: str = Field(
        min_length=1,
        description=("Slash command, agent profile name, or workspace destination"),
    )

    @field_validator("value")
    @classmethod
    def trim_value(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @model_validator(mode="after")
    def validate_action_value(self) -> ControlCLIArgs:
        if self.action is CLIControlAction.COMMAND and not self.value.startswith("/"):
            raise ValueError("command value must start with '/'")
        if self.action is CLIControlAction.NAVIGATE_WORKSPACE:
            try:
                WorkspaceDestination(self.value.lower())
            except ValueError as exc:
                destinations = ", ".join(item.value for item in WorkspaceDestination)
                raise ValueError(
                    f"workspace destination must be one of: {destinations}"
                ) from exc
        return self

    def to_request(self) -> CLIControlRequest:
        match self.action:
            case CLIControlAction.COMMAND:
                return CLICommandRequest(command=self.value)
            case CLIControlAction.SWITCH_AGENT:
                return CLISwitchAgentRequest(profile=self.value)
            case CLIControlAction.NAVIGATE_WORKSPACE:
                return CLINavigateWorkspaceRequest(
                    destination=WorkspaceDestination(self.value.lower())
                )


class ControlCLIConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class ControlCLI(
    BaseTool[ControlCLIArgs, CLIControlResult, ControlCLIConfig, BaseToolState],
    ToolUIData[ControlCLIArgs, CLIControlResult],
):
    @classmethod
    def get_name(cls) -> str:
        return "control_cli"

    @classmethod
    def is_available(cls, config: AnyVibeConfig | None = None) -> bool:
        return bool(
            config and config.enable_orchestrator_controls and config.enable_cli_control
        )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if isinstance(event.args, ControlCLIArgs):
            return ToolCallDisplay(summary=f"CLI control: {event.args.action.value}")
        return ToolCallDisplay(summary="CLI control")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, CLIControlResult):
            return ToolResultDisplay(success=True, message=event.result.message)
        return ToolResultDisplay(success=True, message="CLI action deferred")

    @classmethod
    def get_status_text(cls) -> str:
        return "Controlling CLI"

    async def run(
        self, args: ControlCLIArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | CLIControlResult, None]:
        if ctx is None or ctx.cli_control is None:
            raise ToolError("CLI controls are not available on this surface")
        if not ctx.cli_control.capabilities.supports(args.action):
            raise ToolError(
                f"CLI control action '{args.action.value}' is not supported"
            )

        try:
            result = await ctx.cli_control.defer(args.to_request())
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        yield result
