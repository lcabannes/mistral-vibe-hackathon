from __future__ import annotations

import vibe.core.agents as agents
from vibe.core.agents import events, management_port
from vibe.core.agents.management_port import AgentManagementPort


def test_management_port_contains_only_agent_operations() -> None:
    assert not hasattr(management_port, "ManagedAgentLifecycleListener")
    public_members = {
        name for name in vars(AgentManagementPort) if not name.startswith("_")
    }
    assert public_members == {
        "start",
        "list",
        "available_profiles",
        "message",
        "output",
        "stop",
    }


def test_package_reexports_only_the_lifecycle_event() -> None:
    assert agents.ManagedAgentLifecycleEvent is events.ManagedAgentLifecycleEvent
    assert not hasattr(agents, "ManagedAgentLifecycleListener")
