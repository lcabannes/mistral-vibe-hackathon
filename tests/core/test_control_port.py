from __future__ import annotations

from pydantic import TypeAdapter, ValidationError
import pytest

from vibe.core.control_port import (
    CLICommandRequest,
    CLIControlAction,
    CLIControlCapabilities,
    CLIControlRequest,
    CLINavigateWorkspaceRequest,
    WorkspaceDestination,
)


def test_control_request_union_is_strict_and_discriminated() -> None:
    request = TypeAdapter(CLIControlRequest).validate_python({
        "action": "navigate_workspace",
        "destination": "office",
    })

    assert isinstance(request, CLINavigateWorkspaceRequest)
    assert request.destination is WorkspaceDestination.OFFICE

    with pytest.raises(ValidationError, match="extra_forbidden"):
        TypeAdapter(CLIControlRequest).validate_python({
            "action": "command",
            "command": "/status",
            "profile": "default",
        })


def test_control_request_trims_and_rejects_blank_values() -> None:
    assert CLICommandRequest(command="  /status  ").command == "/status"

    with pytest.raises(ValidationError, match="must not be blank"):
        CLICommandRequest(command="   ")


def test_workspace_destinations_match_delivery_surface_values() -> None:
    assert {destination.value for destination in WorkspaceDestination} == {
        "home",
        "chat",
        "office",
        "agents",
        "mcp",
        "usage",
    }


def test_capabilities_are_explicit_per_action() -> None:
    capabilities = CLIControlCapabilities(actions=frozenset({CLIControlAction.COMMAND}))

    assert capabilities.supports(CLIControlAction.COMMAND)
    assert not capabilities.supports(CLIControlAction.SWITCH_AGENT)
