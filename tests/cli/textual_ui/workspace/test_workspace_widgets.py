from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Input, Link, OptionList, Static

from vibe.cli.textual_ui.widgets.mcp_app import MCPApp
from vibe.cli.textual_ui.widgets.navigable_option_list import NavigableOptionList
from vibe.cli.textual_ui.workspace.models import (
    AgentActivity,
    AgentActivitySnapshot,
    AgentRunState,
    WorkspaceView,
)
from vibe.cli.textual_ui.workspace.navigation import (
    VISIBLE_WORKSPACE_VIEWS,
    WorkspaceNavigation,
)
from vibe.cli.textual_ui.workspace.pages import (
    ActivityOverviewPage,
    AgentProfileViewModel,
    AgentsPage,
    AgentStateCard,
    AgentStateRow,
    AgentsViewModel,
    AnimatedStateBorder,
    HomePage,
    HomeViewModel,
    MCPPage,
    OfficeViewModel,
    UsagePage,
    UsageViewModel,
)
from vibe.core.agent_room.models import AgentRoomConversationMessage
from vibe.core.agents.models import DEFAULT
from vibe.core.config import MCPStdio


def _activity(
    state: AgentRunState,
    *,
    tool_call_id: str = "call-1",
    current_activity: str | None = "Reading files",
) -> AgentActivity:
    return AgentActivity(
        tool_call_id=tool_call_id,
        parent_session_id="parent",
        agent_name="explore",
        agent_display_name="Explore",
        task="Inspect the repository",
        state=state,
        started_at=1.0,
        updated_at=2.0,
        current_activity=current_activity,
    )


class _NavigationApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.selected: list[WorkspaceView] = []

    def compose(self) -> ComposeResult:
        yield WorkspaceNavigation()

    @on(WorkspaceNavigation.ViewSelected)
    def record_selection(self, message: WorkspaceNavigation.ViewSelected) -> None:
        self.selected.append(message.view)


@pytest.mark.asyncio
async def test_navigation_posts_typed_selection_for_every_view() -> None:
    app = _NavigationApp()

    async with app.run_test() as pilot:
        navigation = app.query_one(WorkspaceNavigation)
        assert isinstance(navigation, NavigableOptionList)

        for index, view in enumerate(VISIBLE_WORKSPACE_VIEWS):
            navigation.highlighted = index
            await pilot.press("enter")
            assert navigation.selected_view is view

    assert app.selected == list(VISIBLE_WORKSPACE_VIEWS)


@pytest.mark.parametrize(
    ("state", "label"),
    [
        (AgentRunState.IDLE, "○ idle"),
        (AgentRunState.REQUESTED, "◌ queued"),
        (AgentRunState.RUNNING, "◐ running"),
        (AgentRunState.WORKING, "● working"),
        (AgentRunState.ATTENTION, "! attention"),
        (AgentRunState.FAILED, "× failed"),
        (AgentRunState.COMPLETED, "✓ finished"),
        (AgentRunState.CANCELLED, "○ cancelled"),
        (AgentRunState.STOPPED, "○ stopped"),
    ],
)
def test_agent_state_row_combines_glyph_word_and_state_class(
    state: AgentRunState, label: str
) -> None:
    row = AgentStateRow(state)

    assert str(row.render()) == label
    assert row.has_class(
        {
            AgentRunState.IDLE: "state-idle",
            AgentRunState.REQUESTED: "state-warning",
            AgentRunState.RUNNING: "state-working",
            AgentRunState.WORKING: "state-working",
            AgentRunState.ATTENTION: "state-attention",
            AgentRunState.FAILED: "state-failed",
            AgentRunState.COMPLETED: "state-finished",
            AgentRunState.CANCELLED: "state-idle",
            AgentRunState.STOPPED: "state-idle",
        }[state]
    )


class _BorderApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.border = AnimatedStateBorder(AgentRunState.ATTENTION)

    def compose(self) -> ComposeResult:
        yield self.border


@pytest.mark.asyncio
async def test_animated_state_border_stops_when_hidden_finished_and_unmounted() -> None:
    app = _BorderApp()

    async with app.run_test(size=(78, 12)) as pilot:
        border = app.border
        assert border.is_animating
        first_track = str(border.render())
        await pilot.pause(0.18)
        assert str(border.render()) != first_track

        border.display = False
        await pilot.pause()
        assert not border.is_animating

        border.display = True
        await pilot.pause()
        assert border.is_animating

        border.update_state(AgentRunState.WORKING)
        working_spans = border.render().spans
        await pilot.pause(0.18)
        assert border.render().spans != working_spans

        border.update_state(AgentRunState.COMPLETED)
        assert not border.is_animating
        assert set(str(border.render())) == {"─"}
        completed_track = str(border.render())
        await pilot.pause(0.18)
        assert str(border.render()) == completed_track

        border.update_state(AgentRunState.IDLE)
        idle_track = str(border.render())
        await pilot.pause(0.18)
        assert str(border.render()) == idle_track

    assert not app.border.is_animating


def test_idle_primary_is_not_counted_as_active() -> None:
    activity = _activity(AgentRunState.IDLE)
    snapshot = AgentActivitySnapshot(session_id="parent", activities=(activity,))
    home = ActivityOverviewPage(HomeViewModel(snapshot))
    office = HomePage(OfficeViewModel(snapshot))

    assert "0 active" in str(home._overview_text())
    assert str(home._action_options()[0].prompt) == "✓ Clear"
    assert "✓ Ready" in str(home._system_text())
    assert office._summary_text() == "1 agent  ·  0 active"


def test_terminal_failure_is_history_not_live_action() -> None:
    activity = _activity(AgentRunState.FAILED)
    snapshot = AgentActivitySnapshot(session_id="parent", activities=(activity,))
    home = ActivityOverviewPage(HomeViewModel(snapshot))

    assert "0 attention" in str(home._overview_text())
    assert "1 recent fail" in str(home._overview_text())
    assert str(home._action_options()[0].prompt) == "✓ Clear"
    assert "× failed" in str(home._activity_text())


class _PagesApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        empty_snapshot = AgentActivitySnapshot(session_id="parent")
        self.home = ActivityOverviewPage(HomeViewModel(empty_snapshot))
        self.office = HomePage(OfficeViewModel(empty_snapshot))
        self.agents = AgentsPage(AgentsViewModel())
        self.usage = UsagePage(
            UsageViewModel(
                steps=0,
                prompt_tokens=0,
                completion_tokens=0,
                context_tokens=0,
                tool_calls_succeeded=0,
                tool_calls_failed=0,
                tool_calls_rejected=0,
                session_cost=0.0,
                last_turn_duration=0.0,
                tokens_per_second=0.0,
            )
        )
        self.selected_agents: list[AgentProfileViewModel] = []
        self.selected_attention: list[AgentActivity] = []
        self.agent_messages: list[tuple[str, str]] = []
        self.agent_tasks: list[str] = []
        self.stopped_agents: list[str] = []
        self.cancelled_agents: list[str] = []
        self.agent_approvals: list[tuple[str, str, str]] = []
        self.agent_answers: list[tuple[str, str, list[dict[str, object]]]] = []

    def compose(self) -> ComposeResult:
        yield self.home
        yield self.office
        yield self.agents
        yield self.usage

    @on(AgentsPage.AgentSelected)
    def record_agent_selection(self, message: AgentsPage.AgentSelected) -> None:
        self.selected_agents.append(message.profile)

    @on(ActivityOverviewPage.AttentionSelected)
    def record_attention_selection(
        self, message: ActivityOverviewPage.AttentionSelected
    ) -> None:
        self.selected_attention.append(message.activity)

    @on(HomePage.AgentMessageSubmitted)
    def record_agent_message(self, message: HomePage.AgentMessageSubmitted) -> None:
        self.agent_messages.append((message.agent_id, message.content))

    @on(HomePage.AgentCreateRequested)
    def record_agent_create(self, message: HomePage.AgentCreateRequested) -> None:
        self.agent_tasks.append(message.task)

    @on(HomePage.AgentStopRequested)
    def record_agent_stop(self, message: HomePage.AgentStopRequested) -> None:
        self.stopped_agents.append(message.agent_id)

    @on(HomePage.AgentCancelRequested)
    def record_agent_cancel(self, message: HomePage.AgentCancelRequested) -> None:
        self.cancelled_agents.append(message.agent_id)

    @on(HomePage.AgentApprovalResolved)
    def record_agent_approval(self, message: HomePage.AgentApprovalResolved) -> None:
        self.agent_approvals.append((
            message.agent_id,
            message.approval_id,
            message.decision,
        ))

    @on(HomePage.AgentQuestionAnswered)
    def record_agent_answer(self, message: HomePage.AgentQuestionAnswered) -> None:
        self.agent_answers.append((
            message.agent_id,
            message.question_id,
            message.answers,
        ))


@pytest.mark.asyncio
async def test_pages_refresh_from_immutable_view_models_at_narrow_width() -> None:
    app = _PagesApp()
    activity = _activity(AgentRunState.ATTENTION)
    snapshot = AgentActivitySnapshot(session_id="parent", activities=(activity,))

    async with app.run_test(size=(78, 36)) as pilot:
        assert app.home.has_class("narrow")
        assert app.office.has_class("narrow")
        assert app.agents.has_class("narrow")
        assert app.usage.has_class("narrow")

        app.home.update_view(HomeViewModel(snapshot, "MCP connected"))
        app.office.update_view(OfficeViewModel(snapshot))
        app.agents.update_view(
            AgentsViewModel((AgentProfileViewModel.from_profile(DEFAULT),))
        )
        app.usage.update_view(
            UsageViewModel(
                steps=3,
                prompt_tokens=1200,
                completion_tokens=300,
                context_tokens=800,
                tool_calls_succeeded=2,
                tool_calls_failed=1,
                tool_calls_rejected=0,
                session_cost=0.0123,
                last_turn_duration=1.5,
                tokens_per_second=42.0,
            )
        )
        await pilot.pause()

        assert "1 attention" in str(
            app.home.query_one("#home-overview", Static).render()
        )
        assert "MCP connected" in str(
            app.home.query_one("#home-system", Static).render()
        )
        assert len(app.office.query(AgentStateCard)) == 1
        assert "Reading files" in str(
            app.office.query_one(".agent-card-task", Static).render()
        )
        assert "Requires approval" in str(
            app.agents.query_one("#agent-detail", Static).render()
        )
        assert "1.5K total" in str(
            app.usage.query_one("#usage-tokens", Static).render()
        )
        assert "$0.0123" in str(app.usage.query_one("#usage-cost", Static).render())

        agents_list = app.agents.query_one("#agents-list", NavigableOptionList)
        agents_list.focus()
        await pilot.press("enter")
        assert app.selected_agents == [AgentProfileViewModel.from_profile(DEFAULT)]


@pytest.mark.asyncio
async def test_office_card_ids_encode_opaque_activity_ids_without_collisions() -> None:
    app = _PagesApp()
    tool_call_ids = ("primary:session / α", "a:b", "a/b")
    snapshot = AgentActivitySnapshot(
        session_id="parent",
        activities=tuple(
            _activity(AgentRunState.WORKING, tool_call_id=tool_call_id)
            for tool_call_id in tool_call_ids
        ),
    )

    async with app.run_test() as pilot:
        app.office.update_view(OfficeViewModel(snapshot))
        await pilot.pause()

        card_ids = {card.id for card in app.office.query(AgentStateCard)}
        assert card_ids == {
            f"activity-{activity.activity_id.encode().hex()}"
            for activity in snapshot.activities
        }


@pytest.mark.asyncio
async def test_office_mount_and_update_keep_namespaced_activity_ids_distinct() -> None:
    app = _PagesApp()
    task_activity = _activity(AgentRunState.REQUESTED, tool_call_id="managed:worker-1")
    managed_activity = _activity(
        AgentRunState.RUNNING, tool_call_id="managed:worker-1"
    ).model_copy(
        update={"managed_agent_id": "worker-1", "agent_display_name": "Worker"}
    )
    snapshot = AgentActivitySnapshot(
        session_id="parent", activities=(task_activity, managed_activity)
    )

    assert task_activity.activity_id == "task:managed:worker-1"
    assert managed_activity.activity_id == "managed:worker-1"

    async with app.run_test() as pilot:
        app.office.update_view(OfficeViewModel(snapshot))
        await pilot.pause()

        cards = tuple(app.office.query(AgentStateCard))
        assert len(cards) == 2
        assert len({card.id for card in cards}) == 2

        updated = snapshot.model_copy(
            update={
                "activities": (
                    task_activity.model_copy(update={"state": AgentRunState.COMPLETED}),
                    managed_activity.model_copy(
                        update={"state": AgentRunState.WORKING}
                    ),
                )
            }
        )
        app.office.update_view(OfficeViewModel(updated))
        await pilot.pause()

        cards = tuple(app.office.query(AgentStateCard))
        assert len(cards) == 2
        assert {card.activity.state for card in cards} == {
            AgentRunState.COMPLETED,
            AgentRunState.WORKING,
        }


@pytest.mark.asyncio
async def test_home_attention_and_office_details_use_typed_activity() -> None:
    app = _PagesApp()
    activity = _activity(AgentRunState.ATTENTION).model_copy(
        update={
            "managed_agent_id": "researcher-1",
            "queued_messages": 2,
            "last_response": "Found the relevant implementation.",
        }
    )
    snapshot = AgentActivitySnapshot(session_id="parent", activities=(activity,))

    async with app.run_test(size=(78, 36)) as pilot:
        app.home.update_view(HomeViewModel(snapshot))
        app.office.update_view(OfficeViewModel(snapshot))
        await pilot.pause()

        actions = app.home.query_one("#home-action-needed", NavigableOptionList)
        actions.focus()
        await pilot.press("enter")
        assert app.selected_attention == [activity]

        card = app.office.query_one(AgentStateCard)
        card.focus()
        await pilot.press("enter")
        detail = app.office.query_one("#office-detail")
        assert detail.display
        assert "researcher-1" in str(
            app.office.query_one("#office-detail-content", Static).render()
        )
        assert "Found the relevant implementation" in str(
            app.office.query_one("#office-detail-content", Static).render()
        )

        await pilot.press("escape")
        assert detail.display


@pytest.mark.asyncio
async def test_home_auto_selects_agent_and_renders_complete_conversation() -> None:
    app = _PagesApp()
    conversation = tuple(
        AgentRoomConversationMessage(
            id=f"message-{index}",
            role="user" if index % 2 == 0 else "assistant",
            content=f"retained message {index}",
        )
        for index in range(20)
    )
    activity = _activity(AgentRunState.IDLE).model_copy(
        update={"managed_agent_id": "agent-1", "conversation": conversation}
    )

    async with app.run_test(size=(100, 26)) as pilot:
        app.office.update_view(
            OfficeViewModel(
                AgentActivitySnapshot(session_id="room", activities=(activity,))
            )
        )
        await pilot.pause()

        detail = app.office.query_one("#office-detail")
        rendered = str(app.office.query_one("#office-detail-content", Static).render())
        assert app.office._inspected_id == activity.activity_id
        assert detail.display
        assert "retained message 0" in rendered
        assert "retained message 19" in rendered


@pytest.mark.asyncio
async def test_home_displays_clickable_agent_room_link() -> None:
    app = _PagesApp()
    room_url = "http://127.0.0.1:4183/web/agent-room/"

    async with app.run_test(size=(100, 26)) as pilot:
        app.office.update_view(
            OfficeViewModel(
                AgentActivitySnapshot(session_id="room"), server_url=room_url
            )
        )
        await pilot.pause()

        link = app.office.query_one("#office-room-link", Link)
        assert link.display
        assert link.url == room_url
        assert str(link.render()) == "Open Vibe Room in Browser"


@pytest.mark.asyncio
async def test_office_composer_creates_messages_and_stops_room_agents() -> None:
    app = _PagesApp()

    async with app.run_test(size=(100, 38)) as pilot:
        command = app.office.query_one("#office-agent-command", Input)
        command.value = "Implement the API adapter"
        command.focus()
        await pilot.press("enter")
        assert app.agent_tasks == ["Implement the API adapter"]

        activity = _activity(AgentRunState.WORKING).model_copy(
            update={
                "managed_agent_id": "agent-1",
                "runtime_live": True,
                "agent_display_name": "Builder",
            }
        )
        app.office.update_view(
            OfficeViewModel(
                AgentActivitySnapshot(session_id="room", activities=(activity,))
            )
        )
        app.office.inspect(activity)
        command.value = "Run the focused tests"
        command.focus()
        await pilot.press("enter")
        assert app.agent_messages == [("agent-1", "Run the focused tests")]

        await pilot.click("#office-agent-stop")
        assert app.stopped_agents == ["agent-1"]
        await pilot.click("#office-agent-cancel")
        assert app.cancelled_agents == ["agent-1"]


@pytest.mark.asyncio
async def test_office_resolves_room_approvals_and_questions() -> None:
    app = _PagesApp()
    approval_activity = _activity(AgentRunState.ATTENTION).model_copy(
        update={
            "managed_agent_id": "agent-1",
            "runtime_live": True,
            "approvals": (
                {"id": "approval-1", "status": "pending", "tool_name": "bash"},
            ),
        }
    )

    async with app.run_test(size=(100, 38)) as pilot:
        app.office.update_view(
            OfficeViewModel(
                AgentActivitySnapshot(
                    session_id="room", activities=(approval_activity,)
                )
            )
        )
        app.office.inspect(approval_activity)
        await pilot.pause()
        await pilot.click("#office-agent-approve")
        assert app.agent_approvals == [("agent-1", "approval-1", "approve_once")]

        question_activity = approval_activity.model_copy(
            update={
                "approvals": (),
                "questions": (
                    {
                        "id": "question-1",
                        "status": "pending",
                        "questions": [{"question": "Which database?"}],
                    },
                ),
            }
        )
        app.office.update_view(
            OfficeViewModel(
                AgentActivitySnapshot(
                    session_id="room", activities=(question_activity,)
                )
            )
        )
        command = app.office.query_one("#office-agent-command", Input)
        command.value = "PostgreSQL"
        command.focus()
        await pilot.press("enter")
        assert app.agent_answers == [
            (
                "agent-1",
                "question-1",
                [
                    {
                        "question": "Which database?",
                        "answer": "PostgreSQL",
                        "is_other": True,
                    }
                ],
            )
        ]


class _MCPPageApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        tool_manager = MagicMock()
        tool_manager.registered_tools = {}
        tool_manager.available_tools = {}
        self.mcp_app = MCPApp(
            [MCPStdio(name="local", transport="stdio", command="vibe-mcp")],
            tool_manager,
        )
        self.page = MCPPage(self.mcp_app)

    def compose(self) -> ComposeResult:
        yield self.page


@pytest.mark.asyncio
async def test_mcp_page_hosts_one_app_and_routes_existing_sources() -> None:
    app = _MCPPageApp()

    async with app.run_test() as pilot:
        assert len(app.page.query(MCPApp)) == 1
        assert not app.page.show_source("missing")

        app.mcp_app.update_sources(
            [
                MCPStdio(name="local", transport="stdio", command="vibe-mcp"),
                MCPStdio(name="added", transport="stdio", command="new-mcp"),
            ],
            connector_registry=None,
            mcp_registry=None,
        )
        options = app.mcp_app.query_one("#mcp-options", OptionList)
        highlighted = options.highlighted
        assert highlighted is not None
        assert options.get_option_at_index(highlighted).id == "server:local"
        assert app.page.show_source("added")
        await pilot.pause()
        assert "added" in str(app.page.query_one("#mcp-title", Static).render())

        assert app.page.show_source("local")
        await pilot.pause()
        assert "local" in str(app.page.query_one("#mcp-title", Static).render())

        app.page.show_index()
        assert "MCP Servers" in str(app.page.query_one("#mcp-title", Static).render())
