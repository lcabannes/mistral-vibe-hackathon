from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import AsyncMock

import httpx
from pydantic import BaseModel
import pytest
from textual.binding import Binding
from textual.widgets import ContentSwitcher, Input, Static

from tests.conftest import build_test_vibe_app, build_test_vibe_config
from vibe.cli.textual_ui.app import BottomApp, ChatScroll, StartupOptions, VibeApp
from vibe.cli.textual_ui.handlers.event_handler import EventHandler
from vibe.cli.textual_ui.widgets.approval_app import ApprovalApp
from vibe.cli.textual_ui.widgets.chat_input import ChatInputContainer
from vibe.cli.textual_ui.widgets.chat_input.text_area import ChatTextArea
from vibe.cli.textual_ui.widgets.mcp_app import MCPApp
from vibe.cli.textual_ui.widgets.messages import UserMessage
from vibe.cli.textual_ui.widgets.question_app import QuestionApp
from vibe.cli.textual_ui.workspace.coworkers import CoworkersPage
from vibe.cli.textual_ui.workspace.models import (
    AgentActivitySnapshot,
    AgentRunState,
    WorkspaceView,
)
from vibe.cli.textual_ui.workspace.navigation import (
    VISIBLE_WORKSPACE_VIEWS,
    WorkspaceNavigation,
)
from vibe.cli.textual_ui.workspace.pages import (
    AgentStateCard,
    HomePage,
    OfficeViewModel,
)
from vibe.core.agent_room import AgentRoomClient
from vibe.core.config import MCPStdio
from vibe.core.control_port import CLINavigateWorkspaceRequest, WorkspaceDestination
from vibe.core.tools.builtins.ask_user_question import (
    AskUserQuestionArgs,
    Choice,
    Question,
)
from vibe.core.tools.builtins.task import Task, TaskArgs
from vibe.core.types import BaseEvent, CompactEndEvent, LLMMessage, Role, ToolCallEvent


class _ApprovalArgs(BaseModel):
    command: str = "echo hello"


def _task_event(tool_call_id: str = "task-1") -> ToolCallEvent:
    return ToolCallEvent(
        tool_call_id=tool_call_id,
        tool_name="task",
        tool_class=Task,
        args=TaskArgs(task="inspect workspace", agent="explore"),
    )


@pytest.mark.asyncio
async def test_agent_home_uses_the_discovered_room_as_management_backend() -> None:
    def room_api(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "api_version": 1,
                "instance_id": "shared-room",
                "revision": 7,
                "connected": True,
                "workspace": {"integration_branch": "codex/agent-home-staging"},
                "profiles": [{"name": "default", "display_name": "Default"}],
                "activities": [
                    {
                        "tool_call_id": "agent-shared",
                        "child_session_id": "session-shared",
                        "agent_name": "default",
                        "agent_display_name": "Shared builder",
                        "task": "Implement the shared backend",
                        "state": "idle",
                        "runtime_live": True,
                        "conversation": [
                            {
                                "id": "message-shared",
                                "role": "assistant",
                                "content": "Visible in both clients",
                                "status": "succeeded",
                            }
                        ],
                        "usage": {"prompt_tokens": 40, "completion_tokens": 10},
                        "context_tokens": 50,
                        "context_limit": 1000,
                        "estimated_cost_usd": 0.002,
                        "group_id": "implementation",
                        "branch": "room-shared-builder",
                        "worktree_path": "/tmp/room-shared-builder",
                    }
                ],
            },
        )

    client = AgentRoomClient(
        "http://127.0.0.1:4173",
        "primary-session",
        transport=httpx.MockTransport(room_api),
    )
    app = build_test_vibe_app(agent_room_client=client)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.agent_loop.agent_management is client
        home = app.query_one(HomePage)
        activity = home._view.snapshot.activities[0]
        assert activity.managed_agent_id == "agent-shared"
        assert activity.conversation[-1].content == "Visible in both clients"
        assert activity.context_tokens == 50
        assert activity.worktree_path == "/tmp/room-shared-builder"
        assert home._inspected_id == activity.activity_id
        assert "Visible in both clients" in str(
            home.query_one("#office-detail-content", Static).render()
        )


@pytest.mark.asyncio
async def test_five_views_switch_without_rebuilding_chat_state() -> None:
    app = build_test_vibe_app(startup=StartupOptions())

    async with app.run_test() as pilot:
        switcher = app.query_one(ContentSwitcher)
        navigation = app.query_one(WorkspaceNavigation)
        chat = app.query_one("#chat", ChatScroll)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.value = "keep this draft"

        assert switcher.current == "workspace-home"
        assert navigation.selected_view is WorkspaceView.HOME

        for view in VISIBLE_WORKSPACE_VIEWS:
            app.action_show_workspace(view.value)
            assert switcher.current == f"workspace-{view.value}"
            assert navigation.selected_view is view

        await pilot.pause()

        assert app.query_one("#chat", ChatScroll) is chat
        assert app.query_one(ChatInputContainer) is chat_input
        assert chat_input.value == "keep this draft"
        assert len(app.query(MCPApp)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("view", VISIBLE_WORKSPACE_VIEWS)
async def test_ctrl_navigation_bindings_switch_real_content(
    view: WorkspaceView,
) -> None:
    app = build_test_vibe_app(startup=StartupOptions())

    async with app.run_test(size=(120, 36)) as pilot:
        switcher = app.query_one(ContentSwitcher)
        index = VISIBLE_WORKSPACE_VIEWS.index(view) + 1
        await pilot.press(f"ctrl+{index}")
        assert switcher.current == f"workspace-{view.value}"
        assert app.query_one(f"#workspace-{view.value}").display
        assert app.focused is not None and app.focused.is_on_screen


@pytest.mark.asyncio
async def test_numeric_navigation_outside_chat_switches_content() -> None:
    app = build_test_vibe_app(startup=StartupOptions())

    async with app.run_test(size=(120, 36)) as pilot:
        switcher = app.query_one(ContentSwitcher)
        await pilot.press("3")
        assert switcher.current == "workspace-mcp"


@pytest.mark.asyncio
async def test_numeric_key_remains_normal_chat_input() -> None:
    app = build_test_vibe_app()

    async with app.run_test(size=(120, 36)) as pilot:
        chat_input = app.query_one(ChatInputContainer)
        await pilot.press("3")
        assert app._workspace_view is WorkspaceView.CHAT
        assert chat_input.value == "3"


@pytest.mark.asyncio
async def test_rail_arrow_enter_message_switches_content_and_focus() -> None:
    app = build_test_vibe_app(startup=StartupOptions())

    async with app.run_test(size=(120, 36)) as pilot:
        switcher = app.query_one(ContentSwitcher)
        navigation = app.query_one(WorkspaceNavigation)
        assert app.focused is navigation
        await pilot.press("down")
        assert navigation.highlighted == 1
        await pilot.press("enter")
        await pilot.pause()
        assert switcher.current == "workspace-chat"
        assert isinstance(app.focused, ChatTextArea)
        await pilot.press(*list("visible text"))
        assert app.focused.text == "visible text"
        assert app.focused.styles.color.hex == "#E0E0E0"


@pytest.mark.asyncio
async def test_home_keyboard_flow_cycles_agents_opens_composer_and_returns_to_rail(
) -> None:
    app = build_test_vibe_app(startup=StartupOptions())

    async with app.run_test(size=(120, 36)) as pilot:
        navigation = app.query_one(WorkspaceNavigation)
        home = app.query_one(HomePage)
        assert app.focused is navigation

        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.focused, AgentStateCard)

        first = home._view.snapshot.activities[0]
        second = first.model_copy(
            update={
                "tool_call_id": "agent-second",
                "agent_display_name": "Second agent",
                "is_primary": False,
                "managed_agent_id": "agent-second",
            }
        )
        home.update_view(
            OfficeViewModel(
                AgentActivitySnapshot(
                    session_id=home._view.snapshot.session_id,
                    activities=(first, second),
                )
            )
        )
        await pilot.pause()
        home.focus_agents()

        await pilot.press("down")
        assert isinstance(app.focused, AgentStateCard)
        assert app.focused.activity.activity_id == second.activity_id
        assert home._inspected_id == second.activity_id

        await pilot.press("up", "m")
        command = home.query_one("#office-agent-command", Input)
        assert app.focused is command
        await pilot.press(*list("essage agent 12345"))
        assert command.value == "message agent 12345"
        assert app._workspace_view is WorkspaceView.HOME
        assert command.styles.color.hex == "#E0E0E0"

        await pilot.press("escape")
        assert app.focused is navigation


@pytest.mark.asyncio
async def test_clicking_rail_destination_switches_content() -> None:
    app = build_test_vibe_app(startup=StartupOptions())

    async with app.run_test(size=(120, 36)) as pilot:
        switcher = app.query_one(ContentSwitcher)
        navigation = app.query_one(WorkspaceNavigation)
        assert await pilot.click(navigation, offset=(3, 3))
        await pilot.pause()
        assert switcher.current == "workspace-mcp"


def test_fresh_and_startup_flows_choose_expected_initial_view() -> None:
    fresh = build_test_vibe_app(startup=StartupOptions())
    prompted = build_test_vibe_app(initial_prompt="inspect this project")
    resuming = build_test_vibe_app()
    resuming._configure_startup_workspace(StartupOptions(show_resume_picker=True))
    teleporting = build_test_vibe_app(
        config=build_test_vibe_config(vibe_code_enabled=True)
    )
    teleporting._configure_startup_workspace(StartupOptions(teleport_on_start=True))

    assert fresh._workspace_view is WorkspaceView.HOME
    assert prompted._workspace_view is WorkspaceView.CHAT
    assert resuming._workspace_view is WorkspaceView.CHAT
    assert teleporting._workspace_view is WorkspaceView.CHAT


def test_navigation_bindings_are_labeled_and_chat_mode_key_is_input_scoped() -> None:
    bindings = {
        binding.key: binding
        for binding in VibeApp.BINDINGS
        if isinstance(binding, Binding)
    }

    assert set(bindings) >= {"ctrl+1", "ctrl+2", "ctrl+3", "ctrl+4", "ctrl+5"}
    assert all(bindings[f"ctrl+{index}"].show for index in range(1, 6))
    assert "shift+tab" not in bindings
    assert any(binding.key == "shift+tab" for binding in ChatTextArea.BINDINGS)


@pytest.mark.asyncio
async def test_workspace_shell_uses_wide_medium_and_narrow_modes() -> None:
    app = build_test_vibe_app(startup=StartupOptions())

    async with app.run_test(size=(120, 36)) as pilot:
        shell = app.query_one("#workspace-shell")
        navigation = app.query_one(WorkspaceNavigation)
        assert shell.has_class("wide")
        assert navigation.region.width == 20

        await pilot.resize_terminal(90, 30)
        assert shell.has_class("medium")
        assert navigation.region.width == 17

        await pilot.resize_terminal(70, 24)
        assert shell.has_class("narrow")
        assert navigation.region.width == 14


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(140, 40), (100, 32), (70, 24)])
async def test_workspace_default_pages_fit_initial_viewport(
    size: tuple[int, int],
) -> None:
    app = build_test_vibe_app(startup=StartupOptions())

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        for view in VISIBLE_WORKSPACE_VIEWS:
            app.action_show_workspace(view.value)
            await pilot.pause()
            page = app.query_one(f"#workspace-{view.value}")

            assert page.max_scroll_y == 0

            match view:
                case WorkspaceView.HOME:
                    summary = page.query_one("#office-summary", Static)
                    assert "0 active" in str(summary.render())
                    grid = page.query_one("#office-agent-grid")
                    assert grid.region.bottom <= page.region.bottom
                case WorkspaceView.CHAT:
                    assert page.query_one("#chat").region.height > 0
                    assert (
                        page.query_one("#input-container").region.bottom
                        <= page.region.bottom
                    )
                case WorkspaceView.MCP:
                    options = page.query_one("#mcp-options")
                    assert options.region.height >= page.region.height // 2
                case WorkspaceView.USAGE:
                    cost = page.query_one("#usage-cost")
                    assert cost.region.bottom <= page.region.bottom
                case WorkspaceView.COWORKERS:
                    roster = app.query_one(CoworkersPage).query_one("#coworkers-list")
                    assert roster.region.bottom <= page.region.bottom


@pytest.mark.asyncio
async def test_activity_event_precedes_handler_and_refreshes_home() -> None:
    app = build_test_vibe_app()
    observed_before_handler: list[bool] = []

    class _RecordingHandler:
        async def handle_event(
            self, event: BaseEvent, loading_widget: Static | None = None
        ) -> None:
            observed_before_handler.append(
                any(
                    activity.tool_call_id == "task-1"
                    for activity in app._activity_store.snapshot.activities
                )
            )

    event = _task_event()

    async with app.run_test() as pilot:
        app.event_handler = cast(EventHandler, _RecordingHandler())
        await app._handle_injected_context_event(event)
        await pilot.pause()

        home = app.query_one(HomePage)
        assert any(
            activity.task == "inspect workspace"
            for activity in home._view.snapshot.activities
        )
        task_card = next(
            card
            for card in home.query(AgentStateCard)
            if card.activity.task == "inspect workspace"
        )
        assert "inspect workspace" in str(
            task_card.query_one(".agent-card-task").render()
        )

    assert observed_before_handler == [True]


@pytest.mark.asyncio
async def test_legacy_office_and_agents_destinations_open_home() -> None:
    app = build_test_vibe_app()

    async with app.run_test():
        app._show_workspace(WorkspaceView.OFFICE)
        assert app._workspace_view is WorkspaceView.HOME
        assert app.query_one(ContentSwitcher).current == "workspace-home"

        app._show_workspace(WorkspaceView.AGENTS)
        assert app._workspace_view is WorkspaceView.HOME
        assert app.query_one(ContentSwitcher).current == "workspace-home"


@pytest.mark.asyncio
async def test_mcp_command_routes_to_single_persistent_manager() -> None:
    app: VibeApp = build_test_vibe_app()

    async with app.run_test() as pilot:
        mcp_app = app.query_one(MCPApp)
        await app._show_mcp()
        await pilot.pause()

        assert app._workspace_view is WorkspaceView.MCP
        assert app.query_one(MCPApp) is mcp_app
        assert len(app.query("#mcp-app")) == 1


@pytest.mark.asyncio
async def test_named_mcp_open_and_sync_are_safe_before_mount() -> None:
    server = MCPStdio(name="pre-mounted", transport="stdio", command="cmd")
    app = build_test_vibe_app(
        config=build_test_vibe_config(mcp_servers=[server]), startup=StartupOptions()
    )

    await app._show_mcp(cmd_args="pre-mounted")
    app._sync_mcp_page_sources()

    assert app._workspace_view is WorkspaceView.MCP
    assert app._pending_mcp_source == "pre-mounted"
    assert app.screen_stack == []

    async with app.run_test():
        assert app.query_one(MCPApp)._viewing_server == "pre-mounted"


@pytest.mark.asyncio
async def test_shift_tab_does_not_switch_agent_outside_visible_chat() -> None:
    app = build_test_vibe_app(startup=StartupOptions())
    original_agent = app.agent_loop.agent_profile.name

    async with app.run_test() as pilot:
        assert app._workspace_view is WorkspaceView.HOME
        await pilot.press("shift+tab")
        await pilot.pause()

        assert app.agent_loop.agent_profile.name == original_agent
        assert not isinstance(app.focused, ChatTextArea)

        app.action_show_workspace(WorkspaceView.CHAT.value)
        await pilot.pause()
        await pilot.press("shift+tab")
        await app.workers.wait_for_complete()

        assert app.agent_loop.agent_profile.name != original_agent


@pytest.mark.asyncio
async def test_approval_keeps_chat_visible_during_workspace_navigation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app()
    typing_pause_started = asyncio.Event()
    release_typing_pause = asyncio.Event()

    async def wait_for_typing_pause() -> None:
        typing_pause_started.set()
        await release_typing_pause.wait()

    monkeypatch.setattr(app, "_wait_for_typing_pause", wait_for_typing_pause)

    async with app.run_test(size=(120, 36)) as pilot:
        pending = asyncio.create_task(
            app._approval_callback("bash", _ApprovalArgs(), "call-1", None)
        )
        await typing_pause_started.wait()
        assert app._pending_approval is not None
        assert not app._pending_approval.done()

        await pilot.press("ctrl+3")
        assert app._workspace_view is WorkspaceView.CHAT
        assert app.query_one(ContentSwitcher).current == "workspace-chat"

        navigation = app.query_one(WorkspaceNavigation)
        assert await pilot.click(navigation, offset=(3, 3))
        assert app._workspace_view is WorkspaceView.CHAT
        assert app.query_one(ContentSwitcher).current == "workspace-chat"

        await app._cli_control.defer(
            CLINavigateWorkspaceRequest(destination=WorkspaceDestination.USAGE)
        )
        await app._apply_deferred_cli_control()

        assert app._workspace_view is WorkspaceView.CHAT
        assert app.query_one(ContentSwitcher).current == "workspace-chat"

        release_typing_pause.set()
        await pilot.pause()

        assert app.query_one(ApprovalApp).is_on_screen
        assert app._current_bottom_app is BottomApp.Approval

        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending


@pytest.mark.asyncio
async def test_question_keeps_chat_visible_during_binding_and_rail_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app()
    typing_pause_started = asyncio.Event()
    release_typing_pause = asyncio.Event()

    async def wait_for_typing_pause() -> None:
        typing_pause_started.set()
        await release_typing_pause.wait()

    monkeypatch.setattr(app, "_wait_for_typing_pause", wait_for_typing_pause)
    args = AskUserQuestionArgs(
        questions=[
            Question(
                question="Which database?",
                header="Database",
                options=[Choice(label="PostgreSQL"), Choice(label="SQLite")],
            )
        ]
    )

    async with app.run_test(size=(120, 36)) as pilot:
        pending = asyncio.create_task(app._user_input_callback(args))
        await typing_pause_started.wait()
        assert app._pending_question is not None
        assert not app._pending_question.done()

        await pilot.press("ctrl+5")
        assert app._workspace_view is WorkspaceView.CHAT
        assert app.query_one(ContentSwitcher).current == "workspace-chat"

        navigation = app.query_one(WorkspaceNavigation)
        assert await pilot.click(navigation, offset=(3, 3))
        assert app._workspace_view is WorkspaceView.CHAT
        assert app.query_one(ContentSwitcher).current == "workspace-chat"

        await app._cli_control.defer(
            CLINavigateWorkspaceRequest(destination=WorkspaceDestination.USAGE)
        )
        await app._apply_deferred_cli_control()

        assert app._workspace_view is WorkspaceView.CHAT
        assert app.query_one(ContentSwitcher).current == "workspace-chat"

        release_typing_pause.set()
        await pilot.pause()

        assert app.query_one(QuestionApp).is_on_screen
        assert app._current_bottom_app is BottomApp.Question

        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending


@pytest.mark.asyncio
async def test_clear_rebinds_activity_store_to_new_core_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app()

    async with app.run_test():
        app._observe_public_event(_task_event("stale-clear-task"))
        previous_store = app._activity_store

        async def clear_history() -> None:
            app.agent_loop.session_id = "clear-session"

        monkeypatch.setattr(app.agent_loop, "clear_history", clear_history)
        await app._clear_history()

        snapshot = app._activity_store.snapshot
        assert app._activity_store is not previous_store
        assert snapshot.session_id == "clear-session"
        assert [item.tool_call_id for item in snapshot.activities] == [
            "primary:clear-session"
        ]


@pytest.mark.asyncio
async def test_manual_compaction_reports_running_then_idle_and_rebinds_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app()
    app.agent_loop.messages.append(LLMMessage(role=Role.user, content="compact me"))
    release = asyncio.Event()

    async with app.run_test() as pilot:
        app._observe_public_event(_task_event("stale-compact-task"))

        async def compact(*, extra_instructions: str = "") -> str:
            await release.wait()
            app.agent_loop.session_id = "compact-session"
            return extra_instructions

        monkeypatch.setattr(app.agent_loop, "compact", compact)
        await app._compact_history(cmd_args="focus")
        await pilot.pause()

        running = app._activity_store.snapshot.activities[0]
        assert app._agent_running is True
        assert running.state is AgentRunState.RUNNING
        assert (
            app.query_one(HomePage)._view.snapshot.activities[0].state
            is AgentRunState.RUNNING
        )

        release.set()
        assert app._agent_task is not None
        await app._agent_task
        await pilot.pause()

        snapshot = app._activity_store.snapshot
        assert app._agent_running is False
        assert snapshot.session_id == "compact-session"
        assert snapshot.activities[0].state is AgentRunState.IDLE
        assert [item.tool_call_id for item in snapshot.activities] == [
            "primary:compact-session"
        ]
        home_snapshot = app.query_one(HomePage)._view.snapshot
        assert home_snapshot.session_id == "compact-session"
        assert home_snapshot.activities[0].state is AgentRunState.IDLE


@pytest.mark.asyncio
async def test_observed_auto_compaction_session_transition_resets_activity() -> None:
    app = build_test_vibe_app()

    async with app.run_test():
        app._observe_public_event(_task_event("stale-auto-compact-task"))
        app.agent_loop.session_id = "auto-compact-session"

        app._observe_public_event(
            CompactEndEvent(
                tool_call_id="compact-1",
                summary_length=10,
                old_session_id="old-session",
                new_session_id="auto-compact-session",
            )
        )

        snapshot = app._activity_store.snapshot
        assert snapshot.session_id == "auto-compact-session"
        assert all(
            item.tool_call_id != "stale-auto-compact-task"
            for item in snapshot.activities
        )


@pytest.mark.asyncio
async def test_non_inplace_rewind_rebinds_activity_to_forked_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app()
    app.agent_loop.messages.append(LLMMessage(role=Role.user, content="rewind me"))

    async with app.run_test():
        app._observe_public_event(_task_event("stale-rewind-task"))
        target = UserMessage("rewind me", message_index=1)
        await app._messages_area.mount(target)
        app._rewind_mode = True
        app._rewind_highlighted_widget = target

        async def rewind_to_message(
            message_index: int, *, restore_files: bool, inplace: bool = False
        ) -> tuple[str, list[str], list[str]]:
            assert message_index == 1
            assert restore_files is False
            assert inplace is False
            app.agent_loop.session_id = "rewind-session"
            return "rewind me", [], []

        monkeypatch.setattr(
            app.agent_loop.rewind_manager, "rewind_to_message", rewind_to_message
        )
        await app._execute_rewind(restore_files=False)

        snapshot = app._activity_store.snapshot
        assert snapshot.session_id == "rewind-session"
        assert [item.tool_call_id for item in snapshot.activities] == [
            "primary:rewind-session"
        ]


@pytest.mark.asyncio
async def test_hidden_mcp_page_does_not_refresh_and_visible_page_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app(startup=StartupOptions())
    refresh = AsyncMock(return_value="Refreshed.")
    monkeypatch.setattr(app, "_refresh_mcp_browser", refresh)

    async with app.run_test() as pilot:
        mcp_app = app.query_one(MCPApp)
        await pilot.pause()
        refresh.assert_not_awaited()
        assert mcp_app._refresh_timer is None

        app.action_show_workspace(WorkspaceView.MCP.value)
        await pilot.pause()
        refresh.assert_awaited_once()
        assert mcp_app._refresh_timer is not None

        app.action_show_workspace(WorkspaceView.HOME.value)
        await pilot.pause()
        assert mcp_app._refresh_timer is None

        app.action_show_workspace(WorkspaceView.MCP.value)
        await pilot.pause()
        assert refresh.await_count == 2


@pytest.mark.asyncio
async def test_reload_syncs_persistent_mcp_sources_before_named_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_vibe_app()
    refresh = AsyncMock(return_value="Refreshed.")
    monkeypatch.setattr(app, "_refresh_mcp_browser", refresh)
    server = MCPStdio(name="fresh", transport="stdio", command="cmd")

    async with app.run_test() as pilot:

        async def reload_with_sources() -> None:
            app.agent_loop.config.mcp_servers = [server]

        monkeypatch.setattr(app.agent_loop.config_orchestrator, "reload", AsyncMock())
        monkeypatch.setattr(
            app.agent_loop, "reload_with_initial_messages", reload_with_sources
        )
        monkeypatch.setattr(app, "_resolve_plan", AsyncMock())

        await app._reload_config()
        mcp_app = app.query_one(MCPApp)
        assert mcp_app._mcp_servers == (server,)
        assert mcp_app._tool_manager is app.agent_loop.tool_manager

        await app._show_mcp(cmd_args="fresh")
        await pilot.pause()

        assert app._workspace_view is WorkspaceView.MCP
        assert mcp_app._viewing_server == "fresh"
