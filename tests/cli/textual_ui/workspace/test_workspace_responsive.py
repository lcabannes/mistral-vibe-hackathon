from __future__ import annotations

import pytest
from textual.widgets import Static

from tests.conftest import build_test_vibe_app
from vibe.cli.textual_ui.app import StartupOptions
from vibe.cli.textual_ui.widgets.navigable_option_list import NavigableOptionList
from vibe.cli.textual_ui.workspace.models import (
    AgentActivity,
    AgentActivitySnapshot,
    AgentRunState,
    WorkspaceView,
)
from vibe.cli.textual_ui.workspace.pages import (
    HomePage,
    HomeViewModel,
    UsagePage,
    UsageViewModel,
)


def _activity(index: int, state: AgentRunState) -> AgentActivity:
    return AgentActivity(
        tool_call_id=f"responsive-{index}",
        parent_session_id="parent",
        agent_name="explore",
        agent_display_name=f"Explore {index}",
        task=f"Inspect long responsive workspace state {index}",
        state=state,
        started_at=float(index),
        updated_at=float(index + 1),
        current_activity=f"Waiting for approval on item {index}",
    )


@pytest.mark.asyncio
async def test_narrow_home_exposes_every_live_attention_in_local_scroll() -> None:
    app = build_test_vibe_app(startup=StartupOptions())
    attention = tuple(_activity(index, AgentRunState.ATTENTION) for index in range(7))
    failures = tuple(_activity(index, AgentRunState.FAILED) for index in range(7, 9))
    snapshot = AgentActivitySnapshot(
        session_id="parent", activities=attention + failures
    )

    async with app.run_test(size=(70, 24)) as pilot:
        home = app.query_one(HomePage)
        home.update_view(HomeViewModel(snapshot))
        await pilot.pause()

        action = home.query_one("#home-action-needed", NavigableOptionList)
        assert home.max_scroll_y == 0
        assert action.option_count == len(attention)
        assert {
            action.get_option_at_index(index).id for index in range(action.option_count)
        } == {f"action-{activity.activity_id.encode().hex()}" for activity in attention}
        assert action.virtual_size.height > action.size.height
        assert action.region.bottom <= home.region.bottom
        assert "2 recent fail" in str(home.query_one("#home-overview", Static).render())

        action.focus()
        await pilot.press(*("down" for _ in range(action.option_count - 1)))
        assert app.focused is action
        highlighted = action.highlighted
        assert highlighted is not None
        assert highlighted == action.option_count - 1
        assert action.scroll_offset.y > 0
        assert f"! {action.option_count} Attention" in str(
            action.get_option_at_index(highlighted).prompt
        )


@pytest.mark.asyncio
async def test_narrow_populated_usage_is_two_by_two_without_wrapping() -> None:
    app = build_test_vibe_app(startup=StartupOptions())
    populated = UsageViewModel(
        steps=12,
        prompt_tokens=12_345,
        completion_tokens=6_789,
        context_tokens=98_765,
        tool_calls_succeeded=42,
        tool_calls_failed=2,
        tool_calls_rejected=1,
        session_cost=1.2345,
        last_turn_duration=12.3,
        tokens_per_second=456.7,
    )

    async with app.run_test(size=(70, 24)) as pilot:
        app.action_show_workspace(WorkspaceView.USAGE.value)
        usage = app.query_one(UsagePage)
        usage.update_view(populated)
        await pilot.pause()

        metrics = tuple(usage.query(".usage-value"))
        assert usage.max_scroll_y == 0
        assert len(metrics) == 4
        assert len({metric.region.x for metric in metrics}) == 2
        assert len({metric.region.y for metric in metrics}) == 2
        assert all(metric.region.bottom <= usage.region.bottom for metric in metrics)
        assert all(metric.region.height == 3 for metric in metrics)
        for metric in metrics:
            lines = str(metric.render()).splitlines()
            assert len(lines) <= metric.content_region.height
            assert all(len(line) <= metric.content_region.width for line in lines)

        tokens = str(usage.query_one("#usage-tokens", Static).render())
        assert "No model usage yet" not in str(
            usage.query_one("#usage-section", Static).render()
        )
        assert "19K total" in tokens
        assert "99K ctx" in tokens
        assert "12K in" in tokens
        assert "6.8K out" in tokens
