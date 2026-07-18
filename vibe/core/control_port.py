from __future__ import annotations

from enum import StrEnum, auto
from typing import Annotated, Literal, Protocol, final

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CLIControlAction(StrEnum):
    COMMAND = auto()
    SWITCH_AGENT = auto()
    NAVIGATE_WORKSPACE = auto()


class WorkspaceDestination(StrEnum):
    HOME = auto()
    CHAT = auto()
    OFFICE = auto()
    AGENTS = auto()
    MCP = auto()
    USAGE = auto()


class _StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _trim_nonblank(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("value must not be blank")
    return stripped


@final
class CLICommandRequest(_StrictRequestModel):
    action: Literal[CLIControlAction.COMMAND] = CLIControlAction.COMMAND
    command: str = Field(min_length=1)

    @field_validator("command")
    @classmethod
    def trim_command(cls, value: str) -> str:
        return _trim_nonblank(value)


@final
class CLISwitchAgentRequest(_StrictRequestModel):
    action: Literal[CLIControlAction.SWITCH_AGENT] = CLIControlAction.SWITCH_AGENT
    profile: str = Field(min_length=1)

    @field_validator("profile")
    @classmethod
    def trim_profile(cls, value: str) -> str:
        return _trim_nonblank(value)


@final
class CLINavigateWorkspaceRequest(_StrictRequestModel):
    action: Literal[CLIControlAction.NAVIGATE_WORKSPACE] = (
        CLIControlAction.NAVIGATE_WORKSPACE
    )
    destination: WorkspaceDestination

    @field_validator("destination", mode="before")
    @classmethod
    def trim_destination(cls, value: object) -> object:
        if isinstance(value, str):
            return _trim_nonblank(value).lower()
        return value


type CLIControlRequest = Annotated[
    CLICommandRequest | CLISwitchAgentRequest | CLINavigateWorkspaceRequest,
    Field(discriminator="action"),
]


class CLIControlCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    actions: frozenset[CLIControlAction] = Field(default_factory=frozenset)

    def supports(self, action: CLIControlAction) -> bool:
        return action in self.actions


class CLIControlDisposition(StrEnum):
    DEFERRED = auto()


class CLIControlResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition: CLIControlDisposition = CLIControlDisposition.DEFERRED
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def trim_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped


class CLIControlPort(Protocol):
    @property
    def capabilities(self) -> CLIControlCapabilities: ...

    async def defer(self, request: CLIControlRequest) -> CLIControlResult: ...
