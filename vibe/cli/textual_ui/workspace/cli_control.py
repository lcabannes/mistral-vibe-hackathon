from __future__ import annotations

from collections.abc import Callable

from vibe.cli.commands import CommandRegistry
from vibe.core.control_port import (
    CLICommandRequest,
    CLIControlAction,
    CLIControlCapabilities,
    CLIControlRequest,
    CLIControlResult,
    CLINavigateWorkspaceRequest,
    CLISwitchAgentRequest,
)


class TextualCLIControl:
    def __init__(
        self,
        *,
        command_registry: CommandRegistry,
        resolve_primary_profile: Callable[[str], str | None],
    ) -> None:
        self._command_registry = command_registry
        self._resolve_primary_profile = resolve_primary_profile
        self._pending: CLIControlRequest | None = None

    @property
    def capabilities(self) -> CLIControlCapabilities:
        return CLIControlCapabilities(
            actions=frozenset({
                CLIControlAction.COMMAND,
                CLIControlAction.SWITCH_AGENT,
                CLIControlAction.NAVIGATE_WORKSPACE,
            })
        )

    async def defer(self, request: CLIControlRequest) -> CLIControlResult:
        if self._pending is not None:
            raise ValueError("A CLI action is already queued for this turn")

        match request:
            case CLICommandRequest(command=command):
                if not command.startswith("/") or not self._command_registry.parse_command(
                    command
                ):
                    raise ValueError("Command must be an available slash command")
                message = f"Deferred command {command.split(maxsplit=1)[0]}"
            case CLISwitchAgentRequest(profile=profile):
                canonical_profile = self._resolve_primary_profile(profile)
                if canonical_profile is None:
                    raise ValueError("Agent must be an available primary profile")
                request = request.model_copy(update={"profile": canonical_profile})
                message = f"Deferred agent switch to {canonical_profile}"
            case CLINavigateWorkspaceRequest(destination=destination):
                message = f"Deferred workspace navigation to {destination.value}"

        self._pending = request
        return CLIControlResult(message=message)

    def pop_pending(self) -> CLIControlRequest | None:
        request = self._pending
        self._pending = None
        return request

    def discard_pending(self) -> None:
        self._pending = None


__all__ = ["TextualCLIControl"]
