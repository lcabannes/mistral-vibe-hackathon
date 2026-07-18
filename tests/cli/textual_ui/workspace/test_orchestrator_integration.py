from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

from pydantic import BaseModel
import pytest

from tests.conftest import build_test_vibe_app
import vibe.cli.textual_ui.app as app_module
from vibe.cli.textual_ui.widgets.approval_app import ApprovalApp
from vibe.cli.textual_ui.workspace.models import AgentRunState, WorkspaceView
from vibe.core.agents.events import ManagedAgentCallbackContext
from vibe.core.control_port import (
    CLICommandRequest,
    CLINavigateWorkspaceRequest,
    CLISwitchAgentRequest,
    WorkspaceDestination,
)
from vibe.core.types import BaseEvent


class _ApprovalArgs(BaseModel):
    command: str = "echo hello"


def _deferred_turn(
    app: app_module.VibeApp,
    request: CLICommandRequest
    | CLISwitchAgentRequest
    | CLINavigateWorkspaceRequest,
    error: BaseException | None = None,
):
    async def act(*_args: object, **_kwargs: object) -> AsyncGenerator[BaseEvent]:
        await app._cli_control.defer(request)
        if error is not None:
            raise error
        if False:
            yield BaseEvent()

    return act


@pytest.mark.asyncio
async def test_successful_turn_applies_each_deferred_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app()
    handle_command = AsyncMock(return_value=True)
    switch_agent = AsyncMock()
    monkeypatch.setattr(app, "_handle_command", handle_command)
    monkeypatch.setattr(app, "_switch_to_agent", switch_agent)
    monkeypatch.setattr(app, "_handle_agent_loop_init", AsyncMock())
    monkeypatch.setattr(app, "_ensure_loading_widget", AsyncMock())

    async with app.run_test():
        command = CLICommandRequest(command="/status")
        monkeypatch.setattr(app.agent_loop, "act", _deferred_turn(app, command))
        await app._handle_agent_loop_turn("apply command")
        handle_command.assert_awaited_once_with("/status")

        switch = CLISwitchAgentRequest(profile="plan")
        monkeypatch.setattr(app.agent_loop, "act", _deferred_turn(app, switch))
        await app._handle_agent_loop_turn("apply switch")
        switch_agent.assert_awaited_once_with("plan")

        navigation = CLINavigateWorkspaceRequest(
            destination=WorkspaceDestination.OFFICE
        )
        monkeypatch.setattr(app.agent_loop, "act", _deferred_turn(app, navigation))
        await app._handle_agent_loop_turn("apply navigation")
        assert app._workspace_view is WorkspaceView.OFFICE


@pytest.mark.asyncio
async def test_failed_and_cancelled_turns_discard_deferred_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app()
    monkeypatch.setattr(app, "_handle_agent_loop_init", AsyncMock())
    monkeypatch.setattr(app, "_ensure_loading_widget", AsyncMock())
    request = CLINavigateWorkspaceRequest(destination=WorkspaceDestination.OFFICE)

    async with app.run_test():
        monkeypatch.setattr(
            app.agent_loop,
            "act",
            _deferred_turn(app, request, RuntimeError("turn failed")),
        )
        await app._handle_agent_loop_turn("failing turn")
        assert app._workspace_view is WorkspaceView.CHAT
        assert app._cli_control.pop_pending() is None

        monkeypatch.setattr(
            app.agent_loop,
            "act",
            _deferred_turn(app, request, asyncio.CancelledError()),
        )
        with pytest.raises(asyncio.CancelledError):
            await app._handle_agent_loop_turn("cancelled turn")
        assert app._workspace_view is WorkspaceView.CHAT
        assert app._cli_control.pop_pending() is None


@pytest.mark.asyncio
async def test_managed_approval_uses_chat_lock_without_mutating_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBE_TYPING_GRACE_PERIOD_MS", "0")
    monkeypatch.setattr(
        app_module,
        "get_managed_agent_callback_context",
        lambda: ManagedAgentCallbackContext(agent_id="worker-1", profile="explore"),
    )
    app = build_test_vibe_app()

    async with app.run_test() as pilot:
        primary_before = app._activity_store.snapshot.activities[0]
        pending = asyncio.create_task(
            app._approval_callback("bash", _ApprovalArgs(), "call-1", None)
        )
        await pilot.pause()

        assert app.query_one(ApprovalApp).is_on_screen
        primary_during = app._activity_store.snapshot.activities[0]
        assert primary_before.state is AgentRunState.IDLE
        assert primary_during.state is AgentRunState.IDLE

        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending


@pytest.mark.asyncio
async def test_orchestrate_handler_switches_primary_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app()
    switch_agent = AsyncMock()
    monkeypatch.setattr(app, "_switch_to_agent", switch_agent)

    async with app.run_test():
        await app._show_orchestrator()

    switch_agent.assert_awaited_once_with("orchestrator")
