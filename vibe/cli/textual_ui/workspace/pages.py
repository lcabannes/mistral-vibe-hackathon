from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, Input, Link, OptionList, Static
from textual.widgets.option_list import Option, OptionDoesNotExist

from vibe.cli.textual_ui.widgets.navigable_option_list import NavigableOptionList
from vibe.cli.textual_ui.widgets.vscode_compat import (
    VscodeCompatInput,
    patch_vscode_space,
)
from vibe.cli.textual_ui.workspace.models import (
    AgentActivity,
    AgentActivitySnapshot,
    AgentRunState,
)
from vibe.core.agents.models import AgentProfile
from vibe.core.types import AgentStats

if TYPE_CHECKING:
    from vibe.cli.textual_ui.widgets.mcp_app import MCPApp

_WIDE_BREAKPOINT = 110
_MEDIUM_BREAKPOINT = 82
_ANIMATION_INTERVAL_SECONDS = 0.14
_COMPACT_BASE = 1000
_COMPACT_DECIMAL_THRESHOLD = 10
_ACTIVE_STATES = frozenset({
    AgentRunState.REQUESTED,
    AgentRunState.RUNNING,
    AgentRunState.WORKING,
    AgentRunState.ATTENTION,
})
_STATE_RICH_STYLES = {
    "state-idle": "dim",
    "state-warning": "bold",
    "state-working": "bold",
    "state-attention": "bold",
    "state-failed": "bold",
    "state-finished": "bold",
}


def _state_presentation(state: AgentRunState) -> tuple[str, str, str]:
    match state:
        case AgentRunState.IDLE:
            presentation = "○", "idle", "state-idle"
        case AgentRunState.REQUESTED:
            presentation = "◌", "queued", "state-warning"
        case AgentRunState.RUNNING:
            presentation = "◐", "running", "state-working"
        case AgentRunState.WORKING:
            presentation = "●", "working", "state-working"
        case AgentRunState.ATTENTION:
            presentation = "!", "attention", "state-attention"
        case AgentRunState.FAILED:
            presentation = "×", "failed", "state-failed"
        case AgentRunState.COMPLETED:
            presentation = "✓", "finished", "state-finished"
        case AgentRunState.CANCELLED:
            presentation = "○", "cancelled", "state-idle"
        case AgentRunState.STOPPED:
            presentation = "○", "stopped", "state-idle"
    return presentation


def _activity_line(activity: AgentActivity) -> Text:
    glyph, state_label, style_class = _state_presentation(activity.state)
    text = Text(no_wrap=True, overflow="ellipsis")
    text.append(f"{glyph} {state_label:<9}", style=_STATE_RICH_STYLES[style_class])
    if activity.owner_display_name:
        text.append(activity.owner_display_name, style="bold")
        text.append(" · ", style="dim")
    text.append(activity.agent_display_name, style="bold")
    text.append(f"  {activity.current_activity or activity.task}", style="dim")
    return text


def _activity_count_label(count: int, noun: str = "agent") -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {noun}{suffix}"


def _compact_count(value: int) -> str:
    units = ("", "K", "M", "B", "T")
    scaled = float(value)
    unit = units[0]
    for unit in units:
        if scaled < _COMPACT_BASE or unit == units[-1]:
            break
        scaled /= _COMPACT_BASE
    if not unit:
        return str(value)
    precision = 1 if scaled < _COMPACT_DECIMAL_THRESHOLD else 0
    return f"{scaled:.{precision}f}{unit}"


def _activity_widget_id(tool_call_id: str) -> str:
    return f"activity-{tool_call_id.encode().hex()}"


@dataclass(frozen=True, slots=True)
class HomeViewModel:
    snapshot: AgentActivitySnapshot
    system_summary: str = "System ready"
    sync_summary: str | None = None


@dataclass(frozen=True, slots=True)
class OfficeViewModel:
    snapshot: AgentActivitySnapshot
    scope_label: str | None = None
    server_url: str | None = None


@dataclass(frozen=True, slots=True)
class AgentProfileViewModel:
    name: str
    display_name: str
    description: str
    safety: str
    agent_type: str
    install_required: bool

    @classmethod
    def from_profile(cls, profile: AgentProfile) -> AgentProfileViewModel:
        return cls(
            name=profile.name,
            display_name=profile.display_name,
            description=profile.description,
            safety=profile.safety.value,
            agent_type=profile.agent_type.value,
            install_required=profile.install_required,
        )


@dataclass(frozen=True, slots=True)
class AgentsViewModel:
    profiles: tuple[AgentProfileViewModel, ...] = ()

    @classmethod
    def from_profiles(cls, profiles: tuple[AgentProfile, ...]) -> AgentsViewModel:
        return cls(tuple(AgentProfileViewModel.from_profile(item) for item in profiles))


@dataclass(frozen=True, slots=True)
class UsageViewModel:
    steps: int
    prompt_tokens: int
    completion_tokens: int
    context_tokens: int
    tool_calls_succeeded: int
    tool_calls_failed: int
    tool_calls_rejected: int
    session_cost: float
    last_turn_duration: float
    tokens_per_second: float

    @classmethod
    def from_stats(cls, stats: AgentStats) -> UsageViewModel:
        return cls(
            steps=stats.steps,
            prompt_tokens=stats.session_prompt_tokens,
            completion_tokens=stats.session_completion_tokens,
            context_tokens=stats.context_tokens,
            tool_calls_succeeded=stats.tool_calls_succeeded,
            tool_calls_failed=stats.tool_calls_failed,
            tool_calls_rejected=(
                stats.tool_calls_rejected + stats.tool_calls_hook_denied
            ),
            session_cost=stats.session_cost,
            last_turn_duration=stats.last_turn_duration,
            tokens_per_second=stats.tokens_per_second,
        )


class ResponsiveWorkspacePage(Container):
    DEFAULT_CSS = """
    ResponsiveWorkspacePage {
        width: 1fr;
        height: 1fr;
        padding: 1 2;
        background: transparent;
        overflow: hidden;
    }

    ResponsiveWorkspacePage.narrow {
        padding: 1;
    }

    .workspace-title {
        height: 2;
        color: $foreground;
        text-style: bold;
    }

    .workspace-section-title {
        height: 1;
        margin-bottom: 1;
        color: $primary;
        text-style: bold;
    }

    .workspace-muted {
        color: $text-muted;

        &:ansi {
            text-style: dim;
        }
    }
    """

    def on_mount(self) -> None:
        self._set_layout_class(self.size.width)

    def on_resize(self, event: events.Resize) -> None:
        self._set_layout_class(event.size.width)

    def _set_layout_class(self, width: int) -> None:
        self.remove_class("wide", "medium", "narrow")
        if width >= _WIDE_BREAKPOINT:
            self.add_class("wide")
        elif width >= _MEDIUM_BREAKPOINT:
            self.add_class("medium")
        else:
            self.add_class("narrow")


class AnimatedStateBorder(Static):
    DEFAULT_CSS = """
    AnimatedStateBorder {
        width: 1fr;
        height: 1;
        color: $text-muted;
        background: transparent;

        &:ansi {
            text-style: dim;
        }
    }

    AnimatedStateBorder.state-working {
        color: $primary;
    }

    AnimatedStateBorder.state-warning {
        color: $warning;
    }

    AnimatedStateBorder.state-attention,
    AnimatedStateBorder.state-failed {
        color: $error;
    }

    AnimatedStateBorder.state-finished {
        color: $success;
    }
    """

    _ANIMATED_STATES: ClassVar[frozenset[AgentRunState]] = frozenset({
        AgentRunState.RUNNING,
        AgentRunState.WORKING,
        AgentRunState.ATTENTION,
    })

    def __init__(self, state: AgentRunState, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self.state = state
        self._phase = 0
        self._timer: Timer | None = None
        self._apply_state_class()

    @property
    def is_animating(self) -> bool:
        return self._timer is not None

    def update_state(self, state: AgentRunState) -> None:
        if state is self.state:
            return
        self.state = state
        self._phase = 0
        self._apply_state_class()
        self._sync_timer()
        self.refresh()

    def on_mount(self) -> None:
        self._sync_timer()

    def on_show(self) -> None:
        self._sync_timer()

    def on_hide(self) -> None:
        self._stop_timer()

    def on_unmount(self) -> None:
        self._stop_timer()

    def render(self) -> Text:
        width = max(1, self.size.width)
        match self.state:
            case AgentRunState.RUNNING | AgentRunState.WORKING:
                track = Text("─" * width, style="dim")
                start = self._phase % width
                track.stylize("bold", start, min(width, start + 3))
                return track
            case AgentRunState.ATTENTION:
                pattern = "──  "
                offset = self._phase % len(pattern)
                return Text(
                    "".join(
                        pattern[(index + offset) % len(pattern)]
                        for index in range(width)
                    )
                )
            case AgentRunState.COMPLETED:
                return Text("─" * width)
            case AgentRunState.FAILED:
                return Text("━" * width)
            case (
                AgentRunState.IDLE
                | AgentRunState.REQUESTED
                | AgentRunState.CANCELLED
                | AgentRunState.STOPPED
            ):
                return Text("·" * width)

    def _tick(self) -> None:
        self._phase += 1
        self.refresh()

    def _sync_timer(self) -> None:
        should_animate = self.is_mounted and self.state in self._ANIMATED_STATES
        if should_animate and self._timer is None:
            self._timer = self.set_interval(_ANIMATION_INTERVAL_SECONDS, self._tick)
        elif not should_animate:
            self._stop_timer()

    def _stop_timer(self) -> None:
        if self._timer is None:
            return
        self._timer.stop()
        self._timer = None

    def _apply_state_class(self) -> None:
        self.remove_class(
            "state-idle",
            "state-warning",
            "state-working",
            "state-attention",
            "state-failed",
            "state-finished",
        )
        self.add_class(_state_presentation(self.state)[2])


class AgentStateRow(Static):
    DEFAULT_CSS = """
    AgentStateRow {
        width: 1fr;
        height: 1;
        background: transparent;
        color: $text-muted;

        &:ansi {
            text-style: dim;
        }
    }

    AgentStateRow.state-working {
        color: $primary;
    }

    AgentStateRow.state-warning {
        color: $warning;
    }

    AgentStateRow.state-attention,
    AgentStateRow.state-failed {
        color: $error;
    }

    AgentStateRow.state-finished {
        color: $success;
    }
    """

    def __init__(self, state: AgentRunState, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self.state = state
        self.update_state(state)

    def update_state(self, state: AgentRunState) -> None:
        self.state = state
        glyph, word, state_class = _state_presentation(state)
        self.remove_class(
            "state-idle",
            "state-warning",
            "state-working",
            "state-attention",
            "state-failed",
            "state-finished",
        )
        self.add_class(state_class)
        self.update(f"{glyph} {word}")


class AgentStateCard(Vertical):
    can_focus = True

    DEFAULT_CSS = """
    AgentStateCard {
        width: 1fr;
        height: 5;
        padding: 0 1;
        background: $surface 45%;
    }

    AgentStateCard .agent-card-heading {
        width: 1fr;
        height: 1;
        text-style: bold;
    }

    AgentStateCard .agent-card-task {
        width: 1fr;
        height: 1;
        color: $text-muted;

        &:ansi {
            text-style: dim;
        }
    }

    AgentStateCard:focus {
        background: $primary 12%;
    }

    AgentStateCard.selected {
        background: $primary 18%;
    }
    """

    class InspectRequested(Message):
        def __init__(
            self, activity: AgentActivity, *, focus_composer: bool = False
        ) -> None:
            super().__init__()
            self.activity = activity
            self.focus_composer = focus_composer

    def __init__(self, activity: AgentActivity, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.activity = activity

    def compose(self) -> ComposeResult:
        yield AnimatedStateBorder(self.activity.state)
        yield Static(self._heading_text(), classes="agent-card-heading")
        yield AgentStateRow(self.activity.state)
        yield Static(self._task_text(), classes="agent-card-task")

    def update_activity(self, activity: AgentActivity) -> None:
        self.activity = activity
        if not self.is_mounted:
            return
        self.query_one(AnimatedStateBorder).update_state(activity.state)
        self.query_one(AgentStateRow).update_state(activity.state)
        self.query_one(".agent-card-heading", Static).update(self._heading_text())
        self.query_one(".agent-card-task", Static).update(self._task_text())

    def _task_text(self) -> str:
        return self.activity.current_activity or self.activity.task

    def _heading_text(self) -> str:
        if self.activity.owner_display_name:
            return (
                f"{self.activity.owner_display_name} · "
                f"{self.activity.agent_display_name}"
            )
        return self.activity.agent_display_name

    def on_key(self, event: events.Key) -> None:
        if event.key != "enter":
            return
        event.stop()
        self.post_message(self.InspectRequested(self.activity, focus_composer=True))

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.focus()
        self.post_message(self.InspectRequested(self.activity))


class ActivityOverviewPage(ResponsiveWorkspacePage):
    DEFAULT_CSS = """
    ActivityOverviewPage {
        layout: grid;
        grid-size: 1;
        grid-columns: 1fr;
        grid-rows: 2 6 1fr;
    }

    ActivityOverviewPage .home-summary-row {
        width: 1fr;
        height: 6;
        layout: horizontal;
        margin-bottom: 1;
    }

    ActivityOverviewPage .home-summary {
        width: 1fr;
        height: 5;
        padding: 0 1;
        border-left: solid $foreground-muted;
    }

    ActivityOverviewPage .home-summary .workspace-section-title {
        margin-bottom: 0;
    }

    ActivityOverviewPage.narrow .home-summary {
        padding: 0 0 0 1;
    }

    ActivityOverviewPage #home-activity {
        width: 1fr;
        height: 1fr;
        min-height: 0;
    }

    ActivityOverviewPage #home-activity-list {
        width: 1fr;
        height: 1fr;
        overflow-y: auto;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    ActivityOverviewPage #home-action-needed {
        width: 1fr;
        height: 1fr;
        min-height: 0;
        padding: 0;
        color: $foreground;
        background: transparent;
        border: none;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    ActivityOverviewPage #home-action-needed.action-clear {
        color: $foreground;
    }

    ActivityOverviewPage #home-action-needed > .option-list--option-highlighted {
        color: $foreground;
        background: $error 18%;
        text-style: bold;
    }

    ActivityOverviewPage #home-action-needed.action-clear
        > .option-list--option-highlighted {
        background: $success 18%;
    }

    ActivityOverviewPage #home-system {
        color: $foreground;
    }
    """

    class AttentionSelected(Message):
        def __init__(self, activity: AgentActivity) -> None:
            super().__init__()
            self.activity = activity

    def __init__(self, view: HomeViewModel, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._view = view

    def compose(self) -> ComposeResult:
        yield Static("Home", classes="workspace-title")
        with Horizontal(classes="home-summary-row"):
            with Vertical(classes="home-summary"):
                yield Static("OVERVIEW", classes="workspace-section-title")
                yield Static(self._overview_text(), id="home-overview")
            with Vertical(classes="home-summary"):
                yield Static("ACTION NEEDED", classes="workspace-section-title")
                yield NavigableOptionList(
                    *self._action_options(),
                    id="home-action-needed",
                    classes=self._action_state_class(),
                )
            with Vertical(classes="home-summary"):
                yield Static(
                    "SYNC" if self._view.sync_summary else "SYSTEM",
                    id="home-system-title",
                    classes="workspace-section-title",
                )
                yield Static(self._system_text(), id="home-system")
        with Vertical(id="home-activity"):
            yield Static("RECENT ACTIVITY", classes="workspace-section-title")
            yield Static(self._activity_text(), id="home-activity-list")

    def update_view(self, view: HomeViewModel) -> None:
        self._view = view
        if not self.is_mounted:
            return
        self.query_one("#home-overview", Static).update(self._overview_text())
        action = self.query_one("#home-action-needed", NavigableOptionList)
        highlighted = action.highlighted or 0
        action.clear_options()
        action.add_options(self._action_options())
        action.highlighted = min(highlighted, action.option_count - 1)
        action.remove_class("action-attention", "action-clear")
        action.add_class(self._action_state_class())
        self.query_one("#home-system", Static).update(self._system_text())
        self.query_one("#home-system-title", Static).update(
            "SYNC" if view.sync_summary else "SYSTEM"
        )
        self.query_one("#home-activity-list", Static).update(self._activity_text())

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "home-action-needed" or event.option.id is None:
            return
        event.stop()
        activity = next(
            (
                item
                for item in self._view.snapshot.activities
                if event.option.id == f"action-{item.activity_id.encode().hex()}"
            ),
            None,
        )
        if activity is not None:
            self.post_message(self.AttentionSelected(activity))

    def _overview_text(self) -> Text:
        activities = self._view.snapshot.activities
        active = sum(item.state in _ACTIVE_STATES for item in activities)
        attention = sum(item.state is AgentRunState.ATTENTION for item in activities)
        failures = sum(item.state is AgentRunState.FAILED for item in activities)
        text = Text()
        text.append(_activity_count_label(len(activities)), style="bold")
        text.append(f"\n● {active} active", style="bold" if active else "dim")
        text.append(f"\n! {attention} attention", style="bold" if attention else "dim")
        text.append(f"\n× {failures} recent fail", style="bold" if failures else "dim")
        return text

    def _action_options(self) -> tuple[Option, ...]:
        needs_action = [
            item
            for item in self._view.snapshot.activities
            if item.state is AgentRunState.ATTENTION
        ]
        if not needs_action:
            return (Option(Text("✓ Clear", style="bold"), id="action-clear"),)
        options: list[Option] = []
        for index, activity in enumerate(needs_action, start=1):
            text = Text(no_wrap=True, overflow="ellipsis")
            text.append(f"! {index} Attention ", style="bold")
            if activity.owner_display_name:
                text.append(f"{activity.owner_display_name} · ", style="bold")
            text.append(activity.agent_display_name, style="bold")
            text.append(f": {activity.current_activity or activity.task}", style="dim")
            options.append(
                Option(text, id=f"action-{activity.activity_id.encode().hex()}")
            )
        return tuple(options)

    def _action_state_class(self) -> str:
        if any(
            item.state is AgentRunState.ATTENTION
            for item in self._view.snapshot.activities
        ):
            return "action-attention"
        return "action-clear"

    def _system_text(self) -> Text:
        if summary := self._view.sync_summary:
            match summary[0]:
                case "✓":
                    style = "bold"
                case "!" | "×":
                    style = "bold"
                case _:
                    style = "dim"
            return Text(summary, style=style)
        summary = self._view.system_summary
        suffix = summary.removeprefix("System ready").lstrip(" ·")
        text = Text("✓ Ready", style="bold")
        if suffix:
            text.append(f"\n{suffix}", style="dim")
        return text

    def _activity_text(self) -> Text:
        activities = self._view.snapshot.activities
        if not activities:
            return Text("○ No recent agent runs", style="dim")
        text = Text()
        for index, activity in enumerate(reversed(activities[-8:])):
            if index:
                text.append("\n")
            text.append_text(_activity_line(activity))
        return text


class ChatPage(ResponsiveWorkspacePage):
    def __init__(self, content: Widget | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._content = content

    def compose(self) -> ComposeResult:
        if self._content is None:
            yield Static("Chat", classes="workspace-title")
            yield Static("Conversation", classes="workspace-muted")
            return
        yield self._content


class HomePage(ResponsiveWorkspacePage):
    class NavigationRequested(Message):
        pass

    class AgentMessageSubmitted(Message):
        def __init__(self, agent_id: str, content: str) -> None:
            super().__init__()
            self.agent_id = agent_id
            self.content = content

    class AgentCreateRequested(Message):
        def __init__(self, task: str) -> None:
            super().__init__()
            self.task = task

    class AgentStopRequested(Message):
        def __init__(self, agent_id: str) -> None:
            super().__init__()
            self.agent_id = agent_id

    class AgentCancelRequested(Message):
        def __init__(self, agent_id: str) -> None:
            super().__init__()
            self.agent_id = agent_id

    class AgentApprovalResolved(Message):
        def __init__(self, agent_id: str, approval_id: str, decision: str) -> None:
            super().__init__()
            self.agent_id = agent_id
            self.approval_id = approval_id
            self.decision = decision

    class AgentQuestionAnswered(Message):
        def __init__(
            self, agent_id: str, question_id: str, answers: list[dict[str, object]]
        ) -> None:
            super().__init__()
            self.agent_id = agent_id
            self.question_id = question_id
            self.answers = answers

    DEFAULT_CSS = """
    HomePage {
        layout: grid;
        grid-size: 1;
        grid-columns: 1fr;
        grid-rows: 2 2 1fr 3;
    }

    HomePage #office-body {
        width: 1fr;
        height: 1fr;
        min-height: 0;
        layout: horizontal;
    }

    HomePage #office-agent-grid {
        width: 38;
        height: 1fr;
        min-height: 0;
        layout: grid;
        grid-size: 1;
        grid-columns: 1fr;
        grid-rows: 5;
        grid-gutter: 0;
        padding-right: 1;
        overflow-y: auto;
    }

    HomePage #office-empty {
        width: 1fr;
        height: 2;
        padding: 0 1;
        color: $text-muted;

        &:ansi {
            text-style: dim;
        }
    }

    HomePage.medium #office-agent-grid {
        width: 34;
    }

    HomePage.narrow #office-body {
        layout: vertical;
    }

    HomePage.narrow #office-agent-grid {
        width: 1fr;
        height: 10;
        padding-right: 0;
        margin-bottom: 1;
    }

    HomePage #office-summary {
        width: 1fr;
        height: 2;
        color: $text-muted;

        &:ansi {
            text-style: dim;
        }
    }

    HomePage #office-summary-row {
        width: 1fr;
        height: 2;
        layout: horizontal;
    }

    HomePage #office-room-link {
        width: auto;
        height: 1;
    }

    HomePage #office-detail {
        width: 1fr;
        height: 1fr;
        min-height: 0;
        padding: 0 0 0 1;
        border-left: solid $foreground-muted;
        overflow-y: auto;
    }

    HomePage #office-detail-content {
        width: 1fr;
        height: auto;
    }

    HomePage.narrow #office-detail {
        width: 1fr;
        height: 1fr;
        min-height: 0;
        padding: 1 0 0 0;
        border-left: none;
        border-top: solid $foreground-muted;
    }

    HomePage #office-agent-actions {
        width: 1fr;
        height: 3;
        layout: horizontal;
        align-vertical: middle;
    }

    HomePage #office-agent-command {
        width: 1fr;
        height: 3;
        color: $foreground;

        &:ansi:dark {
            color: #e0e0e0;

            & > .input--cursor {
                color: #202020;
                background: #e0e0e0;
            }
        }

        &:ansi:light {
            color: #202020;

            & > .input--cursor {
                color: #e0e0e0;
                background: #202020;
            }
        }
    }

    HomePage #office-agent-actions Button {
        min-width: 8;
        height: 2;
        margin-left: 1;
    }
    """

    def __init__(self, view: OfficeViewModel, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._view = view
        self._inspected_id = (
            view.snapshot.activities[0].activity_id
            if view.snapshot.activities
            else None
        )

    def compose(self) -> ComposeResult:
        yield Static("Home", classes="workspace-title")
        with Horizontal(id="office-summary-row"):
            yield Static(self._summary_text(), id="office-summary")
            room_url = self._view.server_url or ""
            link = Link(
                "Open Vibe Room in Browser",
                url=room_url,
                tooltip=room_url or None,
                id="office-room-link",
            )
            link.display = bool(room_url)
            yield link
        with Horizontal(id="office-body"):
            with Container(id="office-agent-grid"):
                yield from self._cards()
                if not self._view.snapshot.activities:
                    yield Static("○ No agent runs in this session", id="office-empty")
            detail = VerticalScroll(
                Static(id="office-detail-content"), id="office-detail"
            )
            detail.display = self._inspected_id is not None
            yield detail
        with Horizontal(id="office-agent-actions"):
            yield VscodeCompatInput(
                placeholder="Task for a new agent",
                id="office-agent-command",
                select_on_focus=False,
            )
            yield Button("New agent", id="office-agent-create", variant="primary")
            stop = Button("Stop", id="office-agent-stop", variant="error")
            stop.display = False
            yield stop
            cancel = Button("Cancel", id="office-agent-cancel")
            cancel.display = False
            yield cancel
            approve = Button("Approve", id="office-agent-approve", variant="success")
            approve.display = False
            yield approve
            deny = Button("Deny", id="office-agent-deny", variant="error")
            deny.display = False
            yield deny
            send = Button("Send", id="office-agent-send", variant="primary")
            send.display = False
            yield send

    def update_view(self, view: OfficeViewModel) -> None:
        self._view = view
        activity_ids = {
            activity.activity_id for activity in self._view.snapshot.activities
        }
        if self._inspected_id not in activity_ids:
            self._inspected_id = (
                self._view.snapshot.activities[0].activity_id
                if self._view.snapshot.activities
                else None
            )
        if not self.is_mounted:
            return
        self.query_one("#office-summary", Static).update(self._summary_text())
        room_link = self.query_one("#office-room-link", Link)
        room_link.url = self._view.server_url or ""
        room_link.tooltip = self._view.server_url
        room_link.display = self._view.server_url is not None
        self._refresh_cards()
        self._refresh_detail()

    def on_agent_state_card_inspect_requested(
        self, message: AgentStateCard.InspectRequested
    ) -> None:
        message.stop()
        self.inspect(message.activity)
        if message.focus_composer:
            self.query_one("#office-agent-command", Input).focus()

    def on_key(self, event: events.Key) -> None:
        focused = self.screen.focused
        if event.key == "escape":
            event.stop()
            self.post_message(self.NavigationRequested())
            return
        if not isinstance(focused, AgentStateCard):
            return
        patch_vscode_space(event)
        if event.character is not None and event.character.isprintable():
            event.stop()
            event.prevent_default()
            command = self.query_one("#office-agent-command", Input)
            command.focus()
            command.insert_text_at_cursor(event.character)
            return
        direction = {"up": -1, "left": -1, "down": 1, "right": 1}.get(event.key)
        if direction is None:
            return
        event.stop()
        self._focus_relative_agent(focused, direction)

    def focus_agents(self) -> None:
        cards = tuple(self.query(AgentStateCard))
        if not cards:
            self.query_one("#office-agent-command", Input).focus()
            return
        selected = next(
            (card for card in cards if card.activity.activity_id == self._inspected_id),
            cards[0],
        )
        selected.focus()
        self.inspect(selected.activity)

    def _focus_relative_agent(self, focused: AgentStateCard, direction: int) -> None:
        cards = tuple(self.query(AgentStateCard))
        if not cards:
            return
        try:
            index = cards.index(focused)
        except ValueError:
            index = 0
        selected = cards[(index + direction) % len(cards)]
        selected.focus()
        selected.scroll_visible(animate=False)
        self.inspect(selected.activity)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "office-agent-command":
            return
        event.stop()
        self._submit_command()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "office-agent-send":
            self._submit_command()
        elif event.button.id == "office-agent-create":
            self._submit_create()
        elif event.button.id == "office-agent-stop":
            activity = self._inspected_activity()
            if activity is not None and activity.managed_agent_id is not None:
                self.post_message(self.AgentStopRequested(activity.managed_agent_id))
        elif event.button.id == "office-agent-cancel":
            activity = self._inspected_activity()
            if activity is not None and activity.managed_agent_id is not None:
                self.post_message(self.AgentCancelRequested(activity.managed_agent_id))
        elif event.button.id in {"office-agent-approve", "office-agent-deny"}:
            activity = self._inspected_activity()
            approval = self._pending_interaction(activity, "approvals")
            if (
                activity is not None
                and activity.managed_agent_id is not None
                and approval is not None
            ):
                decision = (
                    "approve_once"
                    if event.button.id == "office-agent-approve"
                    else "deny"
                )
                self.post_message(
                    self.AgentApprovalResolved(
                        activity.managed_agent_id, str(approval["id"]), decision
                    )
                )

    def inspect(self, activity: AgentActivity) -> None:
        self._inspected_id = activity.activity_id
        self._refresh_detail()
        for card in self.query(AgentStateCard):
            card.set_class(card.activity.activity_id == self._inspected_id, "selected")
        self.query_one("#office-detail", VerticalScroll).scroll_home(animate=False)

    def _cards(self) -> tuple[AgentStateCard, ...]:
        cards: list[AgentStateCard] = []
        for activity in self._view.snapshot.activities:
            card = AgentStateCard(
                activity, id=_activity_widget_id(activity.activity_id)
            )
            card.set_class(activity.activity_id == self._inspected_id, "selected")
            cards.append(card)
        return tuple(cards)

    def _refresh_cards(self) -> None:
        grid = self.query_one("#office-agent-grid", Container)
        existing = {
            card.activity.activity_id: card for card in grid.query(AgentStateCard)
        }
        desired_ids = {
            activity.activity_id for activity in self._view.snapshot.activities
        }
        for empty in grid.query("#office-empty"):
            if desired_ids:
                empty.remove()
        for activity_id, card in existing.items():
            if activity_id not in desired_ids:
                card.remove()
        for activity in self._view.snapshot.activities:
            if card := existing.get(activity.activity_id):
                card.update_activity(activity)
                card.set_class(activity.activity_id == self._inspected_id, "selected")
            else:
                card = AgentStateCard(
                    activity, id=_activity_widget_id(activity.activity_id)
                )
                card.set_class(activity.activity_id == self._inspected_id, "selected")
                grid.mount(card)
        if not desired_ids and not grid.query("#office-empty"):
            grid.mount(Static("○ No agent runs in this session", id="office-empty"))

    def _refresh_detail(self) -> None:
        if not self.is_mounted:
            return
        detail = self.query_one("#office-detail", VerticalScroll)
        activity = next(
            (
                item
                for item in self._view.snapshot.activities
                if item.activity_id == self._inspected_id
            ),
            None,
        )
        if activity is None:
            self._inspected_id = None
            detail.display = False
            self._refresh_actions(None)
            return
        detail.display = True
        self.query_one("#office-detail-content", Static).update(
            self._detail_text(activity)
        )
        self._refresh_actions(activity)

    def _inspected_activity(self) -> AgentActivity | None:
        return next(
            (
                item
                for item in self._view.snapshot.activities
                if item.activity_id == self._inspected_id
            ),
            None,
        )

    def _refresh_actions(self, activity: AgentActivity | None) -> None:
        command = self.query_one("#office-agent-command", Input)
        create = self.query_one("#office-agent-create", Button)
        send = self.query_one("#office-agent-send", Button)
        stop = self.query_one("#office-agent-stop", Button)
        cancel = self.query_one("#office-agent-cancel", Button)
        approve = self.query_one("#office-agent-approve", Button)
        deny = self.query_one("#office-agent-deny", Button)
        selected = activity is not None and activity.managed_agent_id is not None
        pending_approval = self._pending_interaction(activity, "approvals")
        pending_question = self._pending_interaction(activity, "questions")
        command.placeholder = (
            "Answer the agent's question"
            if pending_question is not None
            else f"Message {activity.agent_display_name}"
            if selected and activity is not None
            else "Task for a new agent"
        )
        create.display = not selected
        send.display = selected and pending_approval is None
        send.label = "Answer" if pending_question is not None else "Send"
        stop.display = selected and activity is not None and activity.runtime_live
        cancel.display = (
            selected
            and activity is not None
            and activity.state
            in {AgentRunState.RUNNING, AgentRunState.WORKING, AgentRunState.ATTENTION}
        )
        approve.display = pending_approval is not None
        deny.display = pending_approval is not None

    def _submit_command(self) -> None:
        activity = self._inspected_activity()
        if activity is None or activity.managed_agent_id is None:
            self._submit_create()
            return
        command = self.query_one("#office-agent-command", Input)
        content = command.value.strip()
        if not content:
            return
        command.value = ""
        question = self._pending_interaction(activity, "questions")
        if question is not None:
            raw_questions = question.get("questions")
            prompts = raw_questions if isinstance(raw_questions, list) else []
            answers = [
                {
                    "question": str(prompt.get("question") or "Question"),
                    "answer": content,
                    "is_other": True,
                }
                for prompt in prompts
                if isinstance(prompt, dict)
            ]
            if not answers:
                answers = [
                    {"question": "Question", "answer": content, "is_other": True}
                ]
            self.post_message(
                self.AgentQuestionAnswered(
                    activity.managed_agent_id, str(question["id"]), answers
                )
            )
            return
        self.post_message(
            self.AgentMessageSubmitted(activity.managed_agent_id, content)
        )

    def _submit_create(self) -> None:
        command = self.query_one("#office-agent-command", Input)
        task = command.value.strip()
        if not task:
            return
        command.value = ""
        self.post_message(self.AgentCreateRequested(task))

    @staticmethod
    def _pending_interaction(
        activity: AgentActivity | None, field: str
    ) -> dict[str, object] | None:
        if activity is None:
            return None
        interactions = (
            activity.approvals if field == "approvals" else activity.questions
        )
        return next(
            (
                interaction
                for interaction in reversed(interactions)
                if interaction.get("status") == "pending"
                and interaction.get("id") is not None
            ),
            None,
        )

    @staticmethod
    def _detail_text(activity: AgentActivity) -> Text:
        text = Text()
        text.append(activity.agent_display_name, style="bold")
        text.append(f"\n{activity.agent_name}", style="dim")
        if activity.managed_agent_id is not None:
            text.append(f"\nWorker  {activity.managed_agent_id}")
            text.append(f"\nQueued  {activity.queued_messages}")
        if activity.group_id:
            text.append(f"\nGroup   {activity.group_id}")
        text.append(f"\nState   {_state_presentation(activity.state)[1]}")
        HomePage._append_pending_interactions(text, activity)
        if activity.error:
            text.append(f"\n\nError\n{activity.error}", style="bold")
        HomePage._append_conversation(text, activity)
        text.append("\n\nRun details", style="bold")
        HomePage._append_run_metrics(text, activity)
        return text

    @staticmethod
    def _append_run_metrics(text: Text, activity: AgentActivity) -> None:
        if activity.model:
            text.append(f"\nModel   {activity.model}")
        if activity.usage is not None:
            total_tokens = (
                activity.usage.prompt_tokens + activity.usage.completion_tokens
            )
            text.append(f"\nTokens  {total_tokens:,}")
        if activity.context_limit:
            text.append(
                f"\nMemory  {activity.context_tokens:,} / {activity.context_limit:,}"
            )
        if activity.estimated_cost_usd:
            text.append(f"\nCost    ${activity.estimated_cost_usd:.4f}")
        if activity.branch:
            text.append(f"\nBranch  {activity.branch}")
        if activity.worktree_path:
            text.append(f"\nTree    {activity.worktree_path}")

    @staticmethod
    def _append_pending_interactions(text: Text, activity: AgentActivity) -> None:
        approval = HomePage._pending_interaction(activity, "approvals")
        if approval is not None:
            text.append("\n\nApproval required", style="bold")
            text.append(f"\n{approval.get('tool_name') or 'Tool call'}")
            arguments = approval.get("arguments")
            if arguments:
                text.append(f"\n{arguments}", style="dim")
        question = HomePage._pending_interaction(activity, "questions")
        if question is not None:
            text.append("\n\nAgent question", style="bold")
            prompts = question.get("questions")
            if isinstance(prompts, list):
                for prompt in prompts:
                    if isinstance(prompt, dict):
                        text.append(f"\n{prompt.get('question') or 'Question'}")

    @staticmethod
    def _append_conversation(text: Text, activity: AgentActivity) -> None:
        text.append("\n\nConversation", style="bold")
        if activity.conversation:
            labels = {
                "user": "You",
                "assistant": "Agent",
                "system": "System",
                "tool": "Tool",
            }
            for message in activity.conversation:
                label = labels.get(message.role, message.role.title())
                text.append(f"\n\n{label}  ", style="bold")
                text.append(message.content)
                if message.status not in {"succeeded", "completed"}:
                    text.append(f"  [{message.status}]", style="dim")
        elif activity.last_response:
            text.append(f"\n\nAgent  {activity.last_response}")
        elif activity.is_managed:
            text.append("\n\nNo messages yet", style="dim")

    def _summary_text(self) -> str:
        activities = self._view.snapshot.activities
        active = sum(item.state in _ACTIVE_STATES for item in activities)
        summary = f"{_activity_count_label(len(activities))}  ·  {active} active"
        context_tokens = sum(item.context_tokens for item in activities)
        estimated_cost = sum(item.estimated_cost_usd for item in activities)
        if context_tokens:
            summary += f"  ·  {_compact_count(context_tokens)} context"
        if estimated_cost:
            summary += f"  ·  ${estimated_cost:.4f}"
        if self._view.scope_label:
            return f"{self._view.scope_label}  ·  {summary}"
        return summary


class AgentsPage(ResponsiveWorkspacePage):
    DEFAULT_CSS = """
    AgentsPage {
        layout: grid;
        grid-size: 1;
        grid-columns: 1fr;
        grid-rows: 2 1fr;
    }

    AgentsPage .agents-body {
        width: 1fr;
        height: 1fr;
        layout: horizontal;
    }

    AgentsPage #agents-list {
        width: 34;
        height: 1fr;
        margin-right: 2;
        background: transparent;
        border: none;
    }

    AgentsPage #agent-detail {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
    }

    AgentsPage.narrow .agents-body {
        layout: vertical;
    }

    AgentsPage.narrow #agents-list {
        width: 1fr;
        height: 7;
        margin-right: 0;
        margin-bottom: 1;
    }

    AgentsPage.narrow #agent-detail {
        width: 1fr;
        height: 1fr;
        padding: 0;
    }
    """

    class AgentSelected(Message):
        def __init__(self, profile: AgentProfileViewModel) -> None:
            super().__init__()
            self.profile = profile

    def __init__(self, view: AgentsViewModel, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._view = view
        self._selected_name = view.profiles[0].name if view.profiles else None

    def compose(self) -> ComposeResult:
        yield Static("Agents", classes="workspace-title")
        with Horizontal(classes="agents-body"):
            yield NavigableOptionList(*self._options(), id="agents-list")
            yield Static(self._detail_text(), id="agent-detail")

    def update_view(self, view: AgentsViewModel) -> None:
        self._view = view
        names = {profile.name for profile in view.profiles}
        if self._selected_name not in names:
            self._selected_name = view.profiles[0].name if view.profiles else None
        if not self.is_mounted:
            return
        options = self.query_one("#agents-list", NavigableOptionList)
        options.clear_options()
        options.add_options(self._options())
        if self._selected_name is not None:
            options.highlighted = self._profile_index(self._selected_name)
        self.query_one("#agent-detail", Static).update(self._detail_text())

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "agents-list" or event.option.id is None:
            return
        event.stop()
        self._selected_name = event.option.id
        self.query_one("#agent-detail", Static).update(self._detail_text())
        if profile := self._selected_profile():
            self.post_message(self.AgentSelected(profile))

    def _options(self) -> tuple[Option, ...]:
        options: list[Option] = []
        for profile in self._view.profiles:
            text = Text(no_wrap=True, overflow="ellipsis")
            text.append(profile.display_name, style="bold")
            text.append("  ·  ", style="dim")
            text.append(
                profile.agent_type,
                style="bold" if profile.agent_type == "subagent" else "dim",
            )
            options.append(Option(text, id=profile.name))
        return tuple(options)

    def _selected_profile(self) -> AgentProfileViewModel | None:
        return next(
            (
                profile
                for profile in self._view.profiles
                if profile.name == self._selected_name
            ),
            None,
        )

    def _profile_index(self, name: str) -> int:
        return next(
            index
            for index, profile in enumerate(self._view.profiles)
            if profile.name == name
        )

    def _detail_text(self) -> Text:
        profile = self._selected_profile()
        if profile is None:
            return Text("No agents available", style="dim")
        text = Text()
        text.append(profile.display_name, style="bold")
        text.append(f"\n{profile.name}", style="dim")
        text.append(f"\n\n{profile.description or 'No description'}")
        text.append("\n\nSafety  ", style="dim")
        text.append(profile.safety)
        text.append("\nType    ", style="dim")
        text.append(
            profile.agent_type,
            style="bold" if profile.agent_type == "subagent" else None,
        )
        if profile.install_required:
            text.append("\n\n! Installation required", style="bold")
        return text


class MCPPage(ResponsiveWorkspacePage):
    DEFAULT_CSS = """
    MCPPage {
        padding: 0;
    }

    MCPPage #mcp-app {
        width: 1fr;
        height: 1fr;
        padding: 1 2;
        border: none;
    }

    MCPPage.narrow #mcp-app {
        padding: 1;
    }

    MCPPage #mcp-content {
        width: 1fr;
        height: 1fr;
    }

    MCPPage #mcp-options {
        width: 1fr;
        height: 1fr;
        max-height: 100%;
    }

    MCPPage #mcp-help {
        width: 1fr;
        height: 1;
        color: $text-muted;

        &:ansi {
            text-style: dim;
        }
    }
    """

    def __init__(self, mcp_app: MCPApp, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._mcp_app = mcp_app

    def compose(self) -> ComposeResult:
        yield self._mcp_app

    def show_index(self) -> None:
        self._mcp_app.action_back()

    def show_source(self, name: str) -> bool:
        self.show_index()
        options = self._mcp_app.query_one("#mcp-options", NavigableOptionList)
        for prefix in ("server", "connector"):
            try:
                index = options.get_option_index(f"{prefix}:{name}")
            except OptionDoesNotExist:
                continue
            options.highlighted = index
            options.action_select()
            return True
        return False


class UsagePage(ResponsiveWorkspacePage):
    DEFAULT_CSS = """
    UsagePage {
        layout: grid;
        grid-size: 1;
        grid-columns: 1fr;
        grid-rows: 2 2 1fr;
    }

    UsagePage .usage-grid {
        width: 1fr;
        height: auto;
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-gutter: 1 2;
    }

    UsagePage.medium .usage-grid {
        grid-gutter: 1 2;
    }

    UsagePage.narrow .usage-grid {
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-gutter: 1 2;
    }

    UsagePage .usage-value {
        height: 3;
        padding: 0 1;
        border-left: solid $foreground-muted;
    }
    """

    def __init__(self, view: UsageViewModel, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._view = view

    def compose(self) -> ComposeResult:
        yield Static("Usage", classes="workspace-title")
        yield Static(
            self._section_text(), id="usage-section", classes="workspace-section-title"
        )
        with Container(classes="usage-grid"):
            yield Static(self._tokens_text(), id="usage-tokens", classes="usage-value")
            yield Static(self._tools_text(), id="usage-tools", classes="usage-value")
            yield Static(self._pace_text(), id="usage-pace", classes="usage-value")
            yield Static(self._cost_text(), id="usage-cost", classes="usage-value")

    def update_view(self, view: UsageViewModel) -> None:
        self._view = view
        if not self.is_mounted:
            return
        self.query_one("#usage-section", Static).update(self._section_text())
        self.query_one("#usage-tokens", Static).update(self._tokens_text())
        self.query_one("#usage-tools", Static).update(self._tools_text())
        self.query_one("#usage-pace", Static).update(self._pace_text())
        self.query_one("#usage-cost", Static).update(self._cost_text())

    def _section_text(self) -> Text:
        text = Text("CURRENT SESSION", style="bold")
        if not any((
            self._view.prompt_tokens,
            self._view.completion_tokens,
            self._view.context_tokens,
            self._view.tool_calls_succeeded,
            self._view.tool_calls_failed,
            self._view.tool_calls_rejected,
            self._view.steps,
        )):
            text.append("  ·  No model usage yet", style="dim")
        return text

    def _tokens_text(self) -> Text:
        total = self._view.prompt_tokens + self._view.completion_tokens
        text = Text("TOKENS", style="bold")
        text.append(
            f"\n{_compact_count(total)} total  ·  "
            f"{_compact_count(self._view.context_tokens)} ctx",
            style="bold",
        )
        text.append(
            f"\n{_compact_count(self._view.prompt_tokens)} in  ·  "
            f"{_compact_count(self._view.completion_tokens)} out",
            style="dim",
        )
        return text

    def _tools_text(self) -> Text:
        total = (
            self._view.tool_calls_succeeded
            + self._view.tool_calls_failed
            + self._view.tool_calls_rejected
        )
        text = Text("TOOLS", style="bold")
        text.append(
            f"\n{_compact_count(total)} calls  ·  "
            f"{_compact_count(self._view.tool_calls_succeeded)} ok",
            style="bold",
        )
        text.append(
            f"\n{_compact_count(self._view.tool_calls_failed)} failed", style="dim"
        )
        text.append("  ·  ", style="dim")
        text.append(
            f"{_compact_count(self._view.tool_calls_rejected)} rejected", style="dim"
        )
        return text

    def _pace_text(self) -> Text:
        text = Text("PACE", style="bold")
        text.append(
            f"\n{_compact_count(self._view.steps)} steps  ·  "
            f"{self._view.last_turn_duration:.1f}s last",
            style="bold",
        )
        text.append(f"\n{self._view.tokens_per_second:.1f} tok/s", style="dim")
        return text

    def _cost_text(self) -> Text:
        text = Text("EST. COST", style="bold")
        text.append(f"\n${self._view.session_cost:.4f}", style="bold")
        return text


__all__ = [
    "ActivityOverviewPage",
    "AgentProfileViewModel",
    "AgentStateCard",
    "AgentStateRow",
    "AgentsPage",
    "AgentsViewModel",
    "AnimatedStateBorder",
    "ChatPage",
    "HomePage",
    "HomeViewModel",
    "MCPPage",
    "OfficeViewModel",
    "ResponsiveWorkspacePage",
    "UsagePage",
    "UsageViewModel",
]
