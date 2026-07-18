from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from vibe.cli.textual_ui.widgets.navigable_option_list import NavigableOptionList
from vibe.cli.textual_ui.workspace.coworkers import (
    CoworkerAgentViewModel,
    CoworkerConversationEntryViewModel,
    CoworkersPage,
    CoworkersViewModel,
    CoworkerViewModel,
)
from vibe.cli.textual_ui.workspace.models import AgentRunState


def _view() -> CoworkersViewModel:
    return CoworkersViewModel(
        workspace_name="mistral-vibe-hackathon-agent-home",
        connection_state="connected",
        privacy_label="summaries shared",
        members=(
            CoworkerViewModel(
                member_id="member_alice",
                display_name="Alice Martin With A Very Long Display Name",
                presence="online",
                branch="codex/team-workspace-with-a-long-branch-name",
                summary="Coordinating the shared workspace integration",
                updated_label="now",
                active_run_count=1,
                agents=(
                    CoworkerAgentViewModel(
                        run_id="run_alice",
                        display_name="Explore",
                        state=AgentRunState.WORKING,
                        summary="Using tool",
                        updated_label="3s ago",
                    ),
                ),
            ),
            CoworkerViewModel(
                member_id="member_bob",
                display_name="Bob Chen",
                presence="stale",
                branch="codex/permissions",
                summary="Waiting on permission integration",
                updated_label="2m ago",
                active_run_count=1,
                agents=(
                    CoworkerAgentViewModel(
                        run_id="run_bob",
                        display_name="Orchestrator",
                        state=AgentRunState.ATTENTION,
                        summary="Waiting for input",
                        updated_label="2m ago",
                        history=(
                            CoworkerConversationEntryViewModel(
                                entry_id="entry_user",
                                role="user",
                                text="Implement the permission boundary",
                                updated_label="4m ago",
                            ),
                            CoworkerConversationEntryViewModel(
                                entry_id="entry_assistant",
                                role="assistant",
                                text=None,
                                updated_label="2m ago",
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


class _CoworkersApp(App[None]):
    def __init__(self, view: CoworkersViewModel | None = None) -> None:
        super().__init__()
        self.page = CoworkersPage(view or _view())

    def compose(self) -> ComposeResult:
        yield self.page


@pytest.mark.asyncio
async def test_roster_preview_run_drill_in_and_back_are_keyboard_accessible() -> None:
    app = _CoworkersApp()

    async with app.run_test(size=(140, 40)) as pilot:
        roster = app.page.query_one("#coworkers-list", NavigableOptionList)
        roster.focus()
        await pilot.press("down")

        assert app.page.selected_member_id == "member_bob"
        assert "Bob Chen" in str(
            app.page.query_one("#coworker-heading", Static).render()
        )

        await pilot.press("enter")
        agents = app.page.query_one("#coworker-agents", NavigableOptionList)
        assert app.focused is agents
        await pilot.press("enter")

        assert app.page.has_class("run-detail")
        detail = str(app.page.query_one("#coworker-run-detail", Static).render())
        assert "CONVERSATION HISTORY" in detail
        assert "Implement the permission boundary" in detail
        assert "Content not shared" in detail

        await pilot.press("backspace")
        assert not app.page.has_class("run-detail")
        assert app.focused is agents


@pytest.mark.asyncio
async def test_live_update_preserves_member_and_run_selection() -> None:
    app = _CoworkersApp()

    async with app.run_test(size=(100, 32)) as pilot:
        roster = app.page.query_one("#coworkers-list", NavigableOptionList)
        roster.highlighted = 1
        await pilot.pause()
        agents = app.page.query_one("#coworker-agents", NavigableOptionList)
        agents.highlighted = 0
        await pilot.pause()

        updated = _view()
        updated_bob = updated.members[1]
        app.page.update_view(
            CoworkersViewModel(
                workspace_name=updated.workspace_name,
                connection_state="degraded",
                privacy_label=updated.privacy_label,
                members=(updated.members[0], updated_bob),
            )
        )
        await pilot.pause()

        assert app.page.selected_member_id == "member_bob"
        assert app.page.selected_run_id == "run_bob"
        assert roster.highlighted == 1
        assert agents.highlighted == 0


@pytest.mark.asyncio
async def test_narrow_coworkers_stacks_local_scroll_regions_without_page_overflow() -> (
    None
):
    app = _CoworkersApp()

    async with app.run_test(size=(70, 24)) as pilot:
        await pilot.pause()
        page = app.page
        roster = page.query_one("#coworkers-list", NavigableOptionList)
        detail = page.query_one("#coworker-detail")

        assert page.has_class("narrow")
        assert page.max_scroll_y == 0
        assert roster.region.width == page.content_region.width
        assert roster.region.bottom <= detail.region.y
        assert detail.region.bottom <= page.region.bottom
        assert page.query_one("#coworker-agents").region.bottom <= detail.region.bottom


@pytest.mark.asyncio
async def test_unconfigured_state_shows_one_join_command_without_fake_members() -> None:
    app = _CoworkersApp(CoworkersViewModel(join_hint="vibe team join <team-repo-url>"))

    async with app.run_test(size=(70, 24)):
        roster = app.page.query_one("#coworkers-list", NavigableOptionList)
        assert roster.option_count == 1
        assert "vibe team join <team-repo-url>" in str(
            roster.get_option_at_index(0).prompt
        )
        assert app.page.selected_member_id is None
