from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from tests.conftest import build_test_vibe_config
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry
from vibe.core.agent_loop import AgentLoop
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.control_port import (
    CLIControlCapabilities,
    CLIControlRequest,
    CLIControlResult,
)


class ReloadingConfigOrchestrator:
    def __init__(self) -> None:
        self.config = build_test_vibe_config()

    async def reload(self) -> None:
        self.config = build_test_vibe_config()

    async def set_field(
        self,
        path: str,
        value: object,
        reason: str = "No reason",
        *,
        target_layer: str | None = None,
    ) -> list[BaseException]:
        return []


class FakeCLIControlPort:
    @property
    def capabilities(self) -> CLIControlCapabilities:
        return CLIControlCapabilities()

    async def defer(self, request: CLIControlRequest) -> CLIControlResult:
        return CLIControlResult(message="deferred")


def make_agent_loop(
    config_orchestrator: ReloadingConfigOrchestrator | None = None,
    *,
    agent_name: str = BuiltinAgentName.ORCHESTRATOR,
) -> AgentLoop:
    return AgentLoop(
        config_orchestrator=config_orchestrator or ReloadingConfigOrchestrator(),
        agent_name=agent_name,
        backend=FakeBackend(),
        mcp_registry=FakeMCPRegistry(),
    )


@pytest.mark.asyncio
async def test_interactive_capabilities_survive_config_reload() -> None:
    config_orchestrator = ReloadingConfigOrchestrator()
    loop = make_agent_loop(config_orchestrator)

    assert loop.config.enable_agent_management is False
    assert loop.config.enable_cli_control is False
    assert loop.config.enable_orchestrator_controls is False

    loop.enable_interactive_surface_capabilities()
    assert loop.config.enable_agent_management is True
    assert loop.config.enable_cli_control is True
    assert loop.config.enable_orchestrator_controls is True

    await loop.refresh_config()

    assert config_orchestrator.config.enable_agent_management is False
    assert config_orchestrator.config.enable_cli_control is False
    assert loop.config.enable_agent_management is True
    assert loop.config.enable_cli_control is True
    assert loop.config.enable_orchestrator_controls is True
    await loop.aclose()


@pytest.mark.asyncio
async def test_session_reset_stops_lazily_created_supervisor(monkeypatch) -> None:
    loop = make_agent_loop()
    loop.enable_interactive_surface_capabilities()
    supervisor = loop._get_agent_supervisor()
    stop_for_session_change = AsyncMock(wraps=supervisor.stop_for_session_change)
    initialize_experiments = AsyncMock()
    monkeypatch.setattr(supervisor, "stop_for_session_change", stop_for_session_change)
    monkeypatch.setattr(loop, "initialize_experiments", initialize_experiments)
    old_session_id = loop.session_id

    await loop._reset_session()

    stop_for_session_change.assert_awaited_once_with()
    assert loop.session_id != old_session_id
    await loop.aclose()


@pytest.mark.asyncio
async def test_set_cli_control_port_accepts_port_and_none() -> None:
    loop = make_agent_loop()
    port = FakeCLIControlPort()

    loop.set_cli_control_port(port)
    assert loop.cli_control is port
    loop.set_cli_control_port(None)
    assert loop.cli_control is None
    await loop.aclose()


@pytest.mark.asyncio
async def test_managed_event_subscription_survives_profile_switches() -> None:
    loop = make_agent_loop(agent_name=BuiltinAgentName.DEFAULT)
    with pytest.raises(RuntimeError, match="Interactive agent management"):
        await anext(loop.managed_agent_events())

    loop.enable_interactive_surface_capabilities()
    supervisor = loop._get_agent_supervisor()
    events = loop.managed_agent_events()
    pending_event = asyncio.create_task(anext(events))
    await asyncio.sleep(0)

    assert pending_event.done() is False
    assert "manage_agents" not in loop.tool_manager.available_tools
    await loop.switch_agent(BuiltinAgentName.ORCHESTRATOR)
    assert loop._get_agent_supervisor() is supervisor
    assert pending_event.done() is False
    assert "manage_agents" in loop.tool_manager.available_tools
    await loop.switch_agent(BuiltinAgentName.DEFAULT)
    assert loop._get_agent_supervisor() is supervisor
    assert pending_event.done() is False
    assert "manage_agents" not in loop.tool_manager.available_tools

    await loop.aclose()
    with pytest.raises(StopAsyncIteration):
        await pending_event
