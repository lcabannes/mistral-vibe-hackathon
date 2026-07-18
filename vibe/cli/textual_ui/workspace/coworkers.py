from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from vibe.cli.textual_ui.widgets.navigable_option_list import NavigableOptionList
from vibe.cli.textual_ui.workspace.models import AgentRunState
from vibe.cli.textual_ui.workspace.pages import ResponsiveWorkspacePage


@dataclass(frozen=True, slots=True)
class CoworkerConversationEntryViewModel:
    entry_id: str
    role: str
    text: str | None
    updated_label: str = ""


@dataclass(frozen=True, slots=True)
class CoworkerAgentViewModel:
    run_id: str
    display_name: str
    state: AgentRunState
    summary: str
    updated_label: str = ""
    history: tuple[CoworkerConversationEntryViewModel, ...] = ()


@dataclass(frozen=True, slots=True)
class CoworkerViewModel:
    member_id: str
    display_name: str
    presence: str
    branch: str | None = None
    summary: str = ""
    updated_label: str = ""
    active_run_count: int = 0
    agents: tuple[CoworkerAgentViewModel, ...] = ()


@dataclass(frozen=True, slots=True)
class CoworkersViewModel:
    workspace_name: str = "Team workspace"
    connection_state: str = "disabled"
    privacy_label: str = "status only"
    members: tuple[CoworkerViewModel, ...] = ()
    error: str | None = None
    join_hint: str | None = None


def _state_presentation(state: AgentRunState) -> tuple[str, str, str]:
    match state:
        case AgentRunState.IDLE:
            return "○", "idle", "state-idle"
        case AgentRunState.REQUESTED:
            return "◌", "queued", "state-warning"
        case AgentRunState.RUNNING:
            return "◐", "running", "state-working"
        case AgentRunState.WORKING:
            return "●", "working", "state-working"
        case AgentRunState.ATTENTION:
            return "!", "attention", "state-attention"
        case AgentRunState.FAILED:
            return "×", "failed", "state-failed"
        case AgentRunState.COMPLETED:
            return "✓", "finished", "state-finished"
        case AgentRunState.CANCELLED:
            return "○", "cancelled", "state-idle"


def _member_presentation(member: CoworkerViewModel) -> tuple[str, str, str]:
    if any(agent.state is AgentRunState.ATTENTION for agent in member.agents):
        return "!", "attention", "state-attention"
    if member.presence == "offline":
        return "○", "offline", "state-idle"
    if member.presence == "error":
        return "×", "error", "state-failed"
    if member.presence == "stale":
        return "!", "stale", "state-warning"
    if member.active_run_count:
        return "●", "active", "state-working"
    return "○", "idle", "state-idle"


def _connection_text(view: CoworkersViewModel) -> Text:
    member_count = len(view.members)
    member_label = "member" if member_count == 1 else "members"
    text = Text(f"{view.workspace_name}  ·  {member_count} {member_label}", style="bold")
    text.append("  ·  ")
    match view.connection_state:
        case "connected":
            text.append("✓ live", style="bold")
        case "degraded":
            text.append("! stale", style="bold")
        case "disconnected":
            text.append("○ offline", style="dim")
        case "connecting":
            text.append("◌ connecting", style="bold")
        case "error":
            text.append("× sync error", style="bold")
        case _:
            text.append("○ local only", style="dim")
    text.append(f"  ·  {view.privacy_label}", style="dim")
    return text


class CoworkersPage(ResponsiveWorkspacePage):
    can_focus = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("backspace", "back", "Back", show=False),
    ]

    DEFAULT_CSS = """
    CoworkersPage {
        layout: grid;
        grid-size: 1;
        grid-columns: 1fr;
        grid-rows: 2 2 1fr;
    }

    CoworkersPage #coworkers-summary {
        height: 2;
        color: $text-muted;

        &:ansi {
            text-style: dim;
        }
    }

    CoworkersPage #coworkers-body {
        width: 1fr;
        height: 1fr;
        min-height: 0;
    }

    CoworkersPage #coworkers-list {
        width: 34;
        height: 1fr;
        margin-right: 2;
        background: transparent;
        border: none;
    }

    CoworkersPage #coworker-detail {
        width: 1fr;
        height: 1fr;
        min-height: 0;
    }

    CoworkersPage #coworker-main-detail {
        width: 1fr;
        height: 1fr;
        min-height: 0;
        layout: grid;
        grid-size: 1;
        grid-columns: 1fr;
        grid-rows: 2 3 1 8 1 1fr;
    }

    CoworkersPage #coworker-heading {
        height: 2;
        text-style: bold;
    }

    CoworkersPage #coworker-work {
        height: 3;
        color: $text-muted;

        &:ansi {
            text-style: dim;
        }
    }

    CoworkersPage #coworker-agents {
        width: 1fr;
        height: 8;
        min-height: 0;
        background: transparent;
        border: none;
    }

    CoworkersPage #coworker-activity {
        width: 1fr;
        height: 1fr;
        min-height: 0;
        overflow-y: auto;
        color: $text-muted;

        &:ansi {
            text-style: dim;
        }
    }

    CoworkersPage #coworker-run-detail {
        display: none;
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
    }

    CoworkersPage.run-detail #coworker-main-detail {
        display: none;
    }

    CoworkersPage.run-detail #coworker-run-detail {
        display: block;
    }

    CoworkersPage.narrow #coworkers-body {
        layout: vertical;
    }

    CoworkersPage.narrow #coworkers-list {
        width: 1fr;
        height: 7;
        margin-right: 0;
        margin-bottom: 1;
    }

    CoworkersPage.narrow #coworker-detail {
        width: 1fr;
        height: 1fr;
    }

    CoworkersPage.narrow #coworker-main-detail {
        grid-rows: 1 2 1 1fr;
    }

    CoworkersPage.narrow #coworker-heading {
        height: 1;
    }

    CoworkersPage.narrow #coworker-work {
        height: 2;
    }

    CoworkersPage.narrow #coworker-agents {
        height: 1fr;
    }

    CoworkersPage.narrow #coworker-recent-title,
    CoworkersPage.narrow #coworker-activity {
        display: none;
    }
    """

    def __init__(self, view: CoworkersViewModel, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._view = view
        self._selected_member_id = view.members[0].member_id if view.members else None
        self._selected_run_id: str | None = None

    @property
    def selected_member_id(self) -> str | None:
        return self._selected_member_id

    @property
    def selected_run_id(self) -> str | None:
        return self._selected_run_id

    def compose(self) -> ComposeResult:
        yield Static("Coworkers", classes="workspace-title")
        yield Static(_connection_text(self._view), id="coworkers-summary")
        with Horizontal(id="coworkers-body"):
            yield NavigableOptionList(*self._member_options(), id="coworkers-list")
            with Vertical(id="coworker-detail"):
                with Vertical(id="coworker-main-detail"):
                    yield Static(self._heading_text(), id="coworker-heading")
                    yield Static(self._work_text(), id="coworker-work")
                    yield Static("AGENTS", classes="workspace-section-title")
                    yield NavigableOptionList(
                        *self._agent_options(), id="coworker-agents"
                    )
                    yield Static(
                        "RECENT ACTIVITY",
                        id="coworker-recent-title",
                        classes="workspace-section-title",
                    )
                    yield Static(self._activity_text(), id="coworker-activity")
                yield Static(self._run_detail_text(), id="coworker-run-detail")

    def update_view(self, view: CoworkersViewModel) -> None:
        self._view = view
        member_ids = {member.member_id for member in view.members}
        if self._selected_member_id not in member_ids:
            self._selected_member_id = view.members[0].member_id if view.members else None
            self._selected_run_id = None
        if not self.is_mounted:
            return

        self.query_one("#coworkers-summary", Static).update(_connection_text(view))
        members = self.query_one("#coworkers-list", NavigableOptionList)
        members.clear_options()
        members.add_options(self._member_options())
        if self._selected_member_id is not None:
            members.highlighted = self._member_index(self._selected_member_id)
        self._refresh_detail()

    def focus_roster(self) -> None:
        self.remove_class("run-detail")
        self.query_one("#coworkers-list", NavigableOptionList).focus()

    def action_back(self) -> None:
        if self.has_class("run-detail"):
            self.remove_class("run-detail")
            self.query_one("#coworker-agents", NavigableOptionList).focus()
            return
        self.focus_roster()

    def on_key(self, event: events.Key) -> None:
        if event.key != "backspace":
            return
        event.stop()
        event.prevent_default()
        self.action_back()

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option.id in {None, "empty"}:
            return
        if event.option_list.id == "coworkers-list":
            event.stop()
            if self._selected_member_id == event.option.id:
                return
            self._selected_member_id = event.option.id
            self._selected_run_id = None
            self.remove_class("run-detail")
            self._refresh_detail()
            return
        if event.option_list.id == "coworker-agents":
            event.stop()
            self._selected_run_id = event.option.id

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id in {None, "empty"}:
            event.stop()
            return
        if event.option_list.id == "coworkers-list":
            event.stop()
            self._selected_member_id = event.option.id
            self._selected_run_id = None
            self._refresh_detail()
            agents = self.query_one("#coworker-agents", NavigableOptionList)
            if agents.option_count:
                agents.focus()
            return
        if event.option_list.id != "coworker-agents" or event.option.id is None:
            return
        event.stop()
        self._selected_run_id = event.option.id
        self.add_class("run-detail")
        self.query_one("#coworker-run-detail", Static).update(
            self._run_detail_text()
        )
        self.focus()

    def _refresh_detail(self) -> None:
        self.query_one("#coworker-heading", Static).update(self._heading_text())
        self.query_one("#coworker-work", Static).update(self._work_text())
        agents = self.query_one("#coworker-agents", NavigableOptionList)
        agents.clear_options()
        agents.add_options(self._agent_options())
        run_ids = {agent.run_id for agent in self._selected_agents()}
        if self._selected_run_id not in run_ids:
            self._selected_run_id = next(iter(run_ids), None)
        if self._selected_run_id is not None:
            agents.highlighted = self._agent_index(self._selected_run_id)
        self.query_one("#coworker-activity", Static).update(self._activity_text())
        self.query_one("#coworker-run-detail", Static).update(
            self._run_detail_text()
        )

    def _member_options(self) -> tuple[Option, ...]:
        if not self._view.members:
            message = self._view.error
            if message is None and self._view.join_hint:
                message = f"Not joined · {self._view.join_hint}"
            if message is None and self._view.connection_state == "disconnected":
                message = "Waiting for team workspace"
            message = message or "No coworkers are sharing this workspace"
            return (Option(Text(f"○ {message}", style="dim"), id="empty"),)

        options: list[Option] = []
        for member in self._view.members:
            glyph, state, style = _member_presentation(member)
            text = Text(no_wrap=True, overflow="ellipsis")
            text.append(f"{glyph} {member.display_name}", style="bold")
            text.append(f"  {state}", style="dim" if style == "state-idle" else "bold")
            if member.active_run_count:
                text.append(f"  ·  {member.active_run_count} active", style="dim")
            options.append(Option(text, id=member.member_id))
        return tuple(options)

    def _agent_options(self) -> tuple[Option, ...]:
        agents = self._selected_agents()
        if not agents:
            return (Option(Text("○ No active agent runs", style="dim"), id="empty"),)
        options: list[Option] = []
        for agent in agents:
            glyph, state, _style = _state_presentation(agent.state)
            text = Text(no_wrap=True, overflow="ellipsis")
            text.append(f"{glyph} {state:<9}", style="bold")
            text.append(agent.display_name, style="bold")
            if agent.summary:
                text.append(f"  {agent.summary}", style="dim")
            if agent.updated_label:
                text.append(f"  {agent.updated_label}", style="dim")
            options.append(Option(text, id=agent.run_id))
        return tuple(options)

    def _selected_member(self) -> CoworkerViewModel | None:
        return next(
            (
                member
                for member in self._view.members
                if member.member_id == self._selected_member_id
            ),
            None,
        )

    def _selected_agents(self) -> tuple[CoworkerAgentViewModel, ...]:
        member = self._selected_member()
        return member.agents if member else ()

    def _selected_agent(self) -> CoworkerAgentViewModel | None:
        return next(
            (
                agent
                for agent in self._selected_agents()
                if agent.run_id == self._selected_run_id
            ),
            None,
        )

    def _member_index(self, member_id: str) -> int:
        return next(
            index
            for index, member in enumerate(self._view.members)
            if member.member_id == member_id
        )

    def _agent_index(self, run_id: str) -> int:
        return next(
            index
            for index, agent in enumerate(self._selected_agents())
            if agent.run_id == run_id
        )

    def _heading_text(self) -> Text:
        member = self._selected_member()
        if member is None:
            return Text(
                self._view.error
                or self._view.join_hint
                or "No coworker selected",
                style="dim",
            )
        glyph, state, _style = _member_presentation(member)
        text = Text(member.display_name, style="bold")
        text.append(f"  ·  {glyph} {state}", style="bold")
        if member.updated_label:
            text.append(f"  ·  {member.updated_label}", style="dim")
        return text

    def _work_text(self) -> Text:
        member = self._selected_member()
        if member is None:
            return Text("Select a coworker to inspect shared work", style="dim")
        text = Text("WORK", style="bold")
        text.append(f"\n{member.branch or 'No branch shared'}")
        if member.summary:
            text.append(f"\n{member.summary}", style="dim")
        elif self._view.privacy_label == "status only":
            text.append("\nActivity summaries are private", style="dim")
        return text

    def _activity_text(self) -> Text:
        agents = self._selected_agents()
        if not agents:
            return Text("○ No recent shared activity", style="dim")
        text = Text()
        for index, agent in enumerate(agents[:8]):
            if index:
                text.append("\n")
            glyph, state, _style = _state_presentation(agent.state)
            text.append(f"{glyph} {agent.display_name}", style="bold")
            text.append(f"  {agent.summary or state}", style="dim")
            if agent.updated_label:
                text.append(f"  ·  {agent.updated_label}", style="dim")
        return text

    def _run_detail_text(self) -> Text:
        member = self._selected_member()
        agent = self._selected_agent()
        if member is None or agent is None:
            return Text("No agent run selected", style="dim")
        glyph, state, _style = _state_presentation(agent.state)
        text = Text(agent.display_name, style="bold")
        text.append(f"\n{glyph} {state}", style="bold")
        text.append(f"\n\nOwner   {member.display_name}")
        text.append(f"\nBranch  {member.branch or 'Not shared'}")
        text.append(f"\nRun     {agent.run_id}", style="dim")
        if agent.updated_label:
            text.append(f"\nUpdated {agent.updated_label}", style="dim")
        if agent.summary:
            text.append(f"\n\n{agent.summary}")
        elif self._view.privacy_label == "status only":
            text.append("\n\nActivity summary is private", style="dim")
        text.append("\n\nCONVERSATION HISTORY", style="bold")
        if not agent.history:
            text.append("\n○ No shared conversation history", style="dim")
            return text
        for entry in agent.history:
            role = "USER" if entry.role == "user" else "ASSISTANT"
            text.append(f"\n{role}", style="bold")
            if entry.updated_label:
                text.append(f"  {entry.updated_label}", style="dim")
            text.append(f"\n{entry.text or 'Message shared without content'}")
        return text


__all__ = [
    "CoworkerAgentViewModel",
    "CoworkerConversationEntryViewModel",
    "CoworkerViewModel",
    "CoworkersPage",
    "CoworkersViewModel",
]
