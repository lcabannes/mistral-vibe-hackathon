from __future__ import annotations

from rich.text import Text
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from vibe.cli.textual_ui.widgets.navigable_option_list import NavigableOptionList
from vibe.cli.textual_ui.workspace.models import WorkspaceView

_VIEW_LABELS: tuple[tuple[WorkspaceView, str, str], ...] = (
    (WorkspaceView.HOME, "⌂", "Home"),
    (WorkspaceView.CHAT, "◆", "Chat"),
    (WorkspaceView.MCP, "◇", "MCP"),
    (WorkspaceView.USAGE, "▥", "Usage"),
    (WorkspaceView.COWORKERS, "♢", "Coworkers"),
)
VISIBLE_WORKSPACE_VIEWS = tuple(view for view, _glyph, _label in _VIEW_LABELS)


def _view_option(view: WorkspaceView, glyph: str, label: str) -> Option:
    text = Text(no_wrap=True)
    text.append(f"{glyph} ", style="bold")
    text.append(label)
    return Option(text, id=view.value)


class WorkspaceNavigation(NavigableOptionList):
    DEFAULT_CSS = """
    WorkspaceNavigation {
        width: 22;
        height: 1fr;
        padding: 1 0;
        background: transparent;
        border: none;
        scrollbar-size: 0 0;

        & > .option-list--option-highlighted {
            color: $foreground;
            background: $primary 18%;
            text-style: bold;
        }
    }
    """

    can_focus = True

    class ViewSelected(Message):
        def __init__(self, view: WorkspaceView) -> None:
            super().__init__()
            self.view = view

    def __init__(self, selected: WorkspaceView = WorkspaceView.HOME) -> None:
        options = [
            _view_option(view, glyph, label) for view, glyph, label in _VIEW_LABELS
        ]
        super().__init__(*options, id="workspace-navigation")
        self._selected = selected
        self.highlighted = self._index_for_view(selected)

    @property
    def selected_view(self) -> WorkspaceView:
        return self._selected

    def select_view(self, view: WorkspaceView) -> None:
        self._selected = view
        self.highlighted = self._index_for_view(view)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is None:
            return
        event.stop()
        view = WorkspaceView(event.option.id)
        self.select_view(view)
        self.post_message(self.ViewSelected(view))

    @staticmethod
    def _index_for_view(view: WorkspaceView) -> int:
        return next(
            index
            for index, (candidate, _glyph, _label) in enumerate(_VIEW_LABELS)
            if candidate is view
        )


__all__ = ["VISIBLE_WORKSPACE_VIEWS", "WorkspaceNavigation"]
