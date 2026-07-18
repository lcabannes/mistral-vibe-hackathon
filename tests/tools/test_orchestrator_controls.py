from __future__ import annotations

from pydantic import ValidationError
import pytest

from tests.conftest import build_test_vibe_config
from tests.mock.utils import collect_result
from vibe.core.agents.models import (
    ORCHESTRATOR,
    ManagedAgentSnapshot,
    ManagedAgentState,
)
from vibe.core.control_port import (
    CLIControlAction,
    CLIControlCapabilities,
    CLIControlDisposition,
    CLIControlRequest,
    CLIControlResult,
    WorkspaceDestination,
)
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError, ToolPermission
from vibe.core.tools.builtins.control_cli import (
    ControlCLI,
    ControlCLIArgs,
    ControlCLIConfig,
)
from vibe.core.tools.builtins.manage_agents import (
    ManageAgents,
    ManageAgentsAction,
    ManageAgentsArgs,
    ManageAgentsConfig,
)
from vibe.core.tools.manager import ToolManager


class FakeCLIControl:
    def __init__(self, actions: frozenset[CLIControlAction]) -> None:
        self.capabilities = CLIControlCapabilities(actions=actions)
        self.requests: list[CLIControlRequest] = []

    async def defer(self, request: CLIControlRequest) -> CLIControlResult:
        self.requests.append(request)
        return CLIControlResult(message="Queued for the end of the turn")


class FakeAgentManagement:
    def __init__(self, snapshot: ManagedAgentSnapshot) -> None:
        self.snapshot = snapshot
        self.started: tuple[str, str, str | None] | None = None

    async def start(
        self, profile: str, task: str, *, name: str | None = None
    ) -> ManagedAgentSnapshot:
        self.started = (profile, task, name)
        return self.snapshot

    def list(self) -> tuple[ManagedAgentSnapshot, ...]:
        return (self.snapshot,)

    def available_profiles(self) -> tuple[str, ...]:
        return ("default", "explore")

    async def message(self, agent_id: str, message: str) -> ManagedAgentSnapshot:
        return self.snapshot

    def output(self, agent_id: str) -> ManagedAgentSnapshot:
        return self.snapshot

    async def stop(self, agent_id: str) -> ManagedAgentSnapshot:
        return self.snapshot


def _snapshot() -> ManagedAgentSnapshot:
    return ManagedAgentSnapshot(
        agent_id="worker",
        child_session_id="child-session",
        profile="default",
        state=ManagedAgentState.STARTING,
        task="work",
    )


def test_control_tools_require_profile_and_runtime_capability_gates() -> None:
    regular_config = build_test_vibe_config()
    regular = ToolManager(lambda: regular_config)
    orchestrator_config = ORCHESTRATOR.apply_to_config(regular_config)
    without_adapters = ToolManager(lambda: orchestrator_config)
    with_adapters_config = regular_config.model_copy(
        update={"enable_cli_control": True, "enable_agent_management": True}
    )
    with_adapters_config = ORCHESTRATOR.apply_to_config(with_adapters_config)
    with_adapters = ToolManager(lambda: with_adapters_config)

    assert "manage_agents" not in regular.available_tools
    assert "control_cli" not in regular.available_tools
    assert "manage_agents" not in without_adapters.available_tools
    assert "control_cli" not in without_adapters.available_tools
    assert "manage_agents" in with_adapters.available_tools
    assert "control_cli" in with_adapters.available_tools


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"action": "start"}, "start requires profile and task"),
        (
            {"action": "message", "agent_id": "worker", "message": "   "},
            "must not be blank",
        ),
        ({"action": "list", "profile": "default"}, "list does not accept field"),
        ({"action": "list", "unexpected": True}, "extra_forbidden"),
    ],
)
def test_manage_agents_args_are_strict(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        ManageAgentsArgs.model_validate(payload)


def test_control_cli_args_are_strict() -> None:
    with pytest.raises(ValidationError, match="must start with"):
        ControlCLIArgs(action=CLIControlAction.COMMAND, value="status")
    with pytest.raises(ValidationError, match="workspace destination"):
        ControlCLIArgs(action=CLIControlAction.NAVIGATE_WORKSPACE, value="settings")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ControlCLIArgs.model_validate({
            "action": "switch_agent",
            "value": "default",
            "extra": True,
        })


def test_only_start_requires_manage_agents_approval() -> None:
    permission = ToolPermission.ASK
    config = ManageAgentsConfig(permission=permission)
    tool = ManageAgents(config_getter=lambda: config, state=BaseToolState())

    start = ManageAgentsArgs(
        action=ManageAgentsAction.START, profile="default", task="work"
    )
    message = ManageAgentsArgs(
        action=ManageAgentsAction.MESSAGE, agent_id="worker", message="continue"
    )

    assert tool.config.permission is permission
    assert tool.resolve_permission(start) is None
    resolved = tool.resolve_permission(message)
    assert resolved is not None
    assert resolved.permission is ToolPermission.ALWAYS


@pytest.mark.parametrize(
    "permission", [ToolPermission.NEVER, ToolPermission.ASK, ToolPermission.ALWAYS]
)
def test_control_cli_preserves_configured_permission(
    permission: ToolPermission,
) -> None:
    config = ControlCLIConfig(permission=permission)
    tool = ControlCLI(config_getter=lambda: config, state=BaseToolState())

    assert tool.config.permission is permission
    assert (
        tool.resolve_permission(
            ControlCLIArgs(action=CLIControlAction.COMMAND, value="/status")
        )
        is None
    )


@pytest.mark.asyncio
async def test_control_cli_enforces_capabilities_and_returns_deferred_result() -> None:
    control = FakeCLIControl(frozenset({CLIControlAction.NAVIGATE_WORKSPACE}))
    tool = ControlCLI(config_getter=ControlCLIConfig, state=BaseToolState())

    result = await collect_result(
        tool.run(
            ControlCLIArgs(
                action=CLIControlAction.NAVIGATE_WORKSPACE, value=" office "
            ),
            InvokeContext(tool_call_id="call", cli_control=control),
        )
    )

    assert result.disposition is CLIControlDisposition.DEFERRED
    request = control.requests[0]
    assert request.action is CLIControlAction.NAVIGATE_WORKSPACE
    assert request.destination is WorkspaceDestination.OFFICE

    with pytest.raises(ToolError, match="not supported"):
        await collect_result(
            tool.run(
                ControlCLIArgs(action=CLIControlAction.SWITCH_AGENT, value="default"),
                InvokeContext(tool_call_id="call", cli_control=control),
            )
        )


@pytest.mark.asyncio
async def test_manage_agents_uses_management_protocol() -> None:
    management = FakeAgentManagement(_snapshot())
    tool = ManageAgents(config_getter=ManageAgentsConfig, state=BaseToolState())

    result = await collect_result(
        tool.run(
            ManageAgentsArgs(
                action=ManageAgentsAction.START,
                profile=" default ",
                task=" work ",
                name=" worker ",
            ),
            InvokeContext(tool_call_id="call", agent_management=management),
        )
    )

    assert management.started == ("default", "work", "worker")
    assert result.agents == [_snapshot()]


def test_managed_agent_snapshot_is_strict() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ManagedAgentSnapshot.model_validate({
            **_snapshot().model_dump(),
            "private_state": "hidden",
        })
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        ManagedAgentSnapshot.model_validate({
            **_snapshot().model_dump(),
            "queued_messages": -1,
        })
