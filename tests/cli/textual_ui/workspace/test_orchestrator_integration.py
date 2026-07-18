from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

from pydantic import BaseModel
import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_app
import vibe.cli.textual_ui.app as app_module
from vibe.cli.textual_ui.widgets.approval_app import ApprovalApp
from vibe.cli.textual_ui.workspace.models import AgentRunState, WorkspaceView
from vibe.cli.textual_ui.workspace.pages import HomePage, OfficePage
from vibe.core.agents.events import (
    ManagedAgentCallbackContext,
    ManagedAgentLifecycleEvent,
)
from vibe.core.agents.models import BuiltinAgentName, ManagedAgentState
from vibe.core.control_port import (
    CLICommandRequest,
    CLINavigateWorkspaceRequest,
    CLISwitchAgentRequest,
    WorkspaceDestination,
)
from vibe.core.session.resume_sessions import ResumeSessionInfo
from vibe.core.types import BaseEvent, LLMUsage


class _ApprovalArgs(BaseModel):
    command: str = "echo hello"


def _deferred_turn(
    app: app_module.VibeApp,
    request: CLICommandRequest | CLISwitchAgentRequest | CLINavigateWorkspaceRequest,
    error: BaseException | None = None,
):
    async def act(*_args: object, **_kwargs: object) -> AsyncGenerator[BaseEvent]:
        await app._cli_control.defer(request)
        if error is not None:
            raise error
        if False:
            yield BaseEvent()

    return act


def _managed_event(session_id: str) -> ManagedAgentLifecycleEvent:
    event = ManagedAgentLifecycleEvent(
        sequence=3,
        agent_id="worker-1",
        profile="explore",
        agent_display_name="Explore",
        parent_session_id=session_id,
        child_session_id="child-worker-1",
        state=ManagedAgentState.IDLE,
        current_activity=None,
        queued_messages=0,
    )
    for name, value in {
        "task": "Inspect the repository",
        "last_response": "The relevant code is in app.py.",
        "error": None,
        "usage": LLMUsage(prompt_tokens=20, completion_tokens=5),
    }.items():
        object.__setattr__(event, name, value)
    return event


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


@pytest.mark.asyncio
async def test_managed_event_consumer_updates_home_office_and_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app()
    release = asyncio.Event()
    closed = asyncio.Event()
    event = _managed_event(app.agent_loop.session_id)

    async def managed_events() -> AsyncGenerator[ManagedAgentLifecycleEvent]:
        try:
            yield event
            await release.wait()
        finally:
            closed.set()

    monkeypatch.setattr(app.agent_loop, "managed_agent_events", managed_events)

    async with app.run_test() as pilot:
        await pilot.pause()
        activity = next(
            item
            for item in app._activity_store.snapshot.activities
            if item.managed_agent_id == "worker-1"
        )
        assert activity.last_response == "The relevant code is in app.py."
        assert any(
            item.managed_agent_id == "worker-1"
            for item in app.query_one(HomePage)._view.snapshot.activities
        )
        assert any(
            item.managed_agent_id == "worker-1"
            for item in app.query_one(OfficePage)._view.snapshot.activities
        )

        await app._stop_managed_agent_events()
        await asyncio.wait_for(closed.wait(), timeout=1)
        assert app._managed_agent_events_task is None


@pytest.mark.asyncio
async def test_default_mount_subscription_survives_orchestrator_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app()
    release = asyncio.Event()
    finish = asyncio.Event()
    event = _managed_event(app.agent_loop.session_id)

    async def managed_events() -> AsyncGenerator[ManagedAgentLifecycleEvent]:
        await release.wait()
        yield event
        await finish.wait()

    monkeypatch.setattr(app.agent_loop, "managed_agent_events", managed_events)

    async with app.run_test() as pilot:
        consumer = app._managed_agent_events_task
        assert consumer is not None and not consumer.done()
        assert app.agent_loop.agent_profile.name == BuiltinAgentName.DEFAULT

        await app._switch_to_agent(BuiltinAgentName.ORCHESTRATOR)
        assert app.agent_loop.agent_profile.name == BuiltinAgentName.ORCHESTRATOR
        assert app._managed_agent_events_task is consumer

        release.set()
        await pilot.pause()
        assert any(
            item.managed_agent_id == "worker-1"
            for item in app._activity_store.snapshot.activities
        )
        finish.set()


@pytest.mark.asyncio
async def test_interactive_capabilities_and_port_survive_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app(
        agent_loop=build_test_agent_loop(agent_name=BuiltinAgentName.ORCHESTRATOR)
    )
    monkeypatch.setattr(app.agent_loop, "reload_with_initial_messages", AsyncMock())
    monkeypatch.setattr(app, "_resolve_plan", AsyncMock())

    async with app.run_test():
        assert app.config.enable_cli_control
        assert app.config.enable_agent_management
        assert app.agent_loop.cli_control is app._cli_control

        await app._reload_config()
        assert app.config.enable_cli_control
        assert app.config.enable_agent_management
        assert app.agent_loop.cli_control is app._cli_control


@pytest.mark.asyncio
async def test_resume_stops_workers_before_replacing_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app()
    old_session_id = app.agent_loop.session_id
    stopped_sessions: list[str] = []

    async def stop_workers() -> None:
        stopped_sessions.append(app.agent_loop.session_id)

    monkeypatch.setattr(
        app_module.SessionLoader, "find_session_by_id", lambda *_args: app.history_file
    )
    monkeypatch.setattr(
        app_module.SessionLoader, "load_session", lambda _path: ([], {})
    )
    monkeypatch.setattr(
        app.agent_loop, "stop_managed_agents_for_session_change", stop_workers
    )
    monkeypatch.setattr(
        app.agent_loop.session_logger, "resume_existing_session", lambda *_args: None
    )
    monkeypatch.setattr(app.agent_loop, "hydrate_experiments_from_session", AsyncMock())
    monkeypatch.setattr(app, "_resume_history_from_messages", AsyncMock())
    monkeypatch.setattr(app._loop_runner, "restore_from_session", lambda: None)

    async with app.run_test():
        await app._resume_local_session(
            ResumeSessionInfo(
                session_id="resume-target", cwd="", title=None, end_time=None
            )
        )

        assert stopped_sessions == [old_session_id]
        assert app.agent_loop.session_id == "resume-target"
        assert app._activity_store.snapshot.session_id == "resume-target"
