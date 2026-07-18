from __future__ import annotations

from typing import Protocol

from vibe.core.agents.models import ManagedAgentSnapshot


class AgentManagementPort(Protocol):
    async def start(
        self, profile: str, task: str, *, name: str | None = None
    ) -> ManagedAgentSnapshot: ...

    def list(self) -> tuple[ManagedAgentSnapshot, ...]: ...

    def available_profiles(self) -> tuple[str, ...]: ...

    async def message(self, agent_id: str, message: str) -> ManagedAgentSnapshot: ...

    def output(self, agent_id: str) -> ManagedAgentSnapshot: ...

    async def stop(self, agent_id: str) -> ManagedAgentSnapshot: ...
