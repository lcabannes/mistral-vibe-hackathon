from __future__ import annotations

import asyncio

import pytest

from vibe.cli.commands import CommandRegistry
from vibe.cli.textual_ui.workspace.cli_control import TextualCLIControl
from vibe.core.control_port import (
    CLICommandRequest,
    CLIControlAction,
    CLINavigateWorkspaceRequest,
    CLISwitchAgentRequest,
    WorkspaceDestination,
)


def _control() -> TextualCLIControl:
    return TextualCLIControl(
        command_registry=CommandRegistry(),
        resolve_primary_profile=lambda profile: (
            "plan" if profile.casefold() == "plan" else None
        ),
    )


def test_control_advertises_exact_textual_capabilities() -> None:
    assert _control().capabilities.actions == frozenset({
        CLIControlAction.COMMAND,
        CLIControlAction.SWITCH_AGENT,
        CLIControlAction.NAVIGATE_WORKSPACE,
    })


@pytest.mark.asyncio
async def test_control_defers_one_valid_action_and_clears_it_on_take() -> None:
    control = _control()
    request = CLICommandRequest(command="/status details")

    result = await control.defer(request)

    assert result.message == "Deferred command /status"
    assert control.pop_pending() == request
    assert control.pop_pending() is None


@pytest.mark.asyncio
async def test_control_rejects_unregistered_commands_subagents_and_second_action() -> None:
    control = _control()

    with pytest.raises(ValueError, match="available slash command"):
        await control.defer(CLICommandRequest(command="/missing"))
    with pytest.raises(ValueError, match="available slash command"):
        await control.defer(CLICommandRequest(command="clear"))
    with pytest.raises(ValueError, match="primary profile"):
        await control.defer(CLISwitchAgentRequest(profile="explore"))

    await control.defer(CLISwitchAgentRequest(profile="Plan"))
    with pytest.raises(ValueError, match="already queued"):
        await control.defer(CLICommandRequest(command="/clear"))

    assert control.pop_pending() == CLISwitchAgentRequest(profile="plan")
    control.discard_pending()
    assert control.pop_pending() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", list(WorkspaceDestination))
async def test_control_defers_every_workspace_destination(
    destination: WorkspaceDestination,
) -> None:
    control = _control()
    request = CLINavigateWorkspaceRequest(destination=destination)

    result = await control.defer(request)

    assert result.message == f"Deferred workspace navigation to {destination.value}"
    assert control.pop_pending() == request


@pytest.mark.asyncio
async def test_concurrent_defer_allows_exactly_one_request() -> None:
    control = _control()
    results = await asyncio.gather(
        control.defer(CLICommandRequest(command="/clear")),
        control.defer(CLISwitchAgentRequest(profile="plan")),
        return_exceptions=True,
    )

    assert sum(not isinstance(result, Exception) for result in results) == 1
    errors = [result for result in results if isinstance(result, Exception)]
    assert len(errors) == 1
    assert str(errors[0]) == "A CLI action is already queued for this turn"
