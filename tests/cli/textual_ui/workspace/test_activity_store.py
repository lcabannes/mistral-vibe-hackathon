from __future__ import annotations

from collections.abc import Iterator
import subprocess
import sys

from pydantic import ValidationError
import pytest

from vibe.cli.textual_ui.workspace.activity_store import AgentActivityStore
from vibe.cli.textual_ui.workspace.models import AgentRunState
from vibe.core.agents.events import ManagedAgentLifecycleEvent
from vibe.core.agents.models import ManagedAgentState
from vibe.core.tools.builtins.task import Task, TaskArgs, TaskResult
from vibe.core.types import (
    LLMUsage,
    SubagentLifecycleEvent,
    SubagentLifecycleState,
    ToolCallEvent,
    ToolResultEvent,
)


def test_importing_workspace_does_not_import_agent_loop() -> None:
    code = """
import sys
import vibe.cli.textual_ui.workspace

if "vibe.core.agent_loop" in sys.modules:
    raise SystemExit("workspace eagerly imported agent loop")
"""

    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr or result.stdout


@pytest.fixture
def clock() -> Iterator[float]:
    return iter(float(value) for value in range(100))


def _call(tool_call_id: str, args: TaskArgs | None = None) -> ToolCallEvent:
    return ToolCallEvent(
        tool_call_id=tool_call_id, tool_name="task", tool_class=Task, args=args
    )


def _lifecycle(
    tool_call_id: str,
    state: SubagentLifecycleState,
    *,
    activity: str | None = None,
    usage: LLMUsage | None = None,
) -> SubagentLifecycleEvent:
    return SubagentLifecycleEvent(
        tool_call_id=tool_call_id,
        tool_name="task",
        message=activity or state.value,
        agent_name="explore",
        agent_display_name="Explore",
        task=f"task {tool_call_id}",
        child_session_id=f"child-{tool_call_id}",
        state=state,
        current_activity=activity,
        terminal_usage=usage,
    )


def _result(
    tool_call_id: str,
    *,
    result: TaskResult | None = None,
    error: str | None = None,
    skipped: bool = False,
    cancelled: bool = False,
) -> ToolResultEvent:
    return ToolResultEvent(
        tool_call_id=tool_call_id,
        tool_name="task",
        tool_class=Task,
        result=result,
        error=error,
        skipped=skipped,
        cancelled=cancelled,
    )


def _managed(
    state: ManagedAgentState,
    *,
    sequence: int = 1,
    agent_id: str = "worker-1",
    parent_session_id: str = "parent",
    activity: str | None = None,
    task: str = "Inspect the repository",
    last_response: str = "",
    error: str | None = None,
    usage: LLMUsage | None = None,
) -> ManagedAgentLifecycleEvent:
    event = ManagedAgentLifecycleEvent(
        sequence=sequence,
        agent_id=agent_id,
        profile="explore",
        agent_display_name="Explore",
        parent_session_id=parent_session_id,
        child_session_id=f"child-{agent_id}",
        state=state,
        current_activity=activity,
        queued_messages=2,
    )
    for name, value in {
        "task": task,
        "last_response": last_response,
        "error": error,
        "usage": usage,
    }.items():
        object.__setattr__(event, name, value)
    return event


def test_duplicate_streaming_calls_upsert_one_activity(clock: Iterator[float]) -> None:
    store = AgentActivityStore("parent", clock=lambda: next(clock))

    assert store.apply(_call("call-1"))
    assert store.apply(
        _call("call-1", TaskArgs(task="inspect repository", agent="explore"))
    )
    assert not store.apply(
        _call("call-1", TaskArgs(task="inspect repository", agent="explore"))
    )

    assert len(store.snapshot.activities) == 1
    activity = store.snapshot.activities[0]
    assert activity.tool_call_id == "call-1"
    assert activity.task == "inspect repository"
    assert activity.agent_name == "explore"
    assert activity.state is AgentRunState.REQUESTED


def test_interleaved_agents_remain_correlated_by_tool_call_id(
    clock: Iterator[float],
) -> None:
    store = AgentActivityStore("parent", clock=lambda: next(clock))
    store.apply(_call("one", TaskArgs(task="first")))
    store.apply(_call("two", TaskArgs(task="second")))
    store.apply(_lifecycle("two", SubagentLifecycleState.RUNNING))
    store.apply(
        _lifecycle("one", SubagentLifecycleState.WORKING, activity="Reading README")
    )

    activities = {
        activity.tool_call_id: activity for activity in store.snapshot.activities
    }
    assert activities["one"].state is AgentRunState.WORKING
    assert activities["one"].current_activity == "Reading README"
    assert activities["one"].child_session_id == "child-one"
    assert activities["two"].state is AgentRunState.RUNNING
    assert activities["two"].task == "task two"


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            _result(
                "completed",
                result=TaskResult(response="done", turns_used=2, completed=True),
            ),
            AgentRunState.COMPLETED,
        ),
        (_result("failed", error="boom"), AgentRunState.FAILED),
        (_result("cancelled", cancelled=True), AgentRunState.CANCELLED),
        (_result("skipped", skipped=True), AgentRunState.CANCELLED),
        (
            _result(
                "interrupted",
                result=TaskResult(response="partial", turns_used=1, completed=False),
            ),
            AgentRunState.CANCELLED,
        ),
    ],
)
def test_tool_results_set_all_terminal_states(
    event: ToolResultEvent, expected: AgentRunState, clock: Iterator[float]
) -> None:
    store = AgentActivityStore("parent", clock=lambda: next(clock))
    store.apply(_call(event.tool_call_id, TaskArgs(task="work")))

    assert store.apply(event)
    assert store.snapshot.activities[0].state is expected


def test_failed_lifecycle_is_not_overwritten_by_incomplete_result(
    clock: Iterator[float],
) -> None:
    store = AgentActivityStore("parent", clock=lambda: next(clock))
    store.apply(_call("failed", TaskArgs(task="work")))
    store.apply(_lifecycle("failed", SubagentLifecycleState.FAILED))
    store.apply(
        _result(
            "failed", result=TaskResult(response="error", turns_used=3, completed=False)
        )
    )

    activity = store.snapshot.activities[0]
    assert activity.state is AgentRunState.FAILED
    assert activity.turns_used == 3


def test_lifecycle_terminal_usage_is_preserved(clock: Iterator[float]) -> None:
    store = AgentActivityStore("parent", clock=lambda: next(clock))
    usage = LLMUsage(prompt_tokens=20, completion_tokens=5)

    store.apply(_lifecycle("completed", SubagentLifecycleState.COMPLETED, usage=usage))
    store.apply(
        _result(
            "completed",
            result=TaskResult(response="done", turns_used=4, completed=True),
        )
    )

    activity = store.snapshot.activities[0]
    assert activity.usage == usage
    assert activity.turns_used == 4


def test_snapshot_and_listener_payloads_are_immutable(clock: Iterator[float]) -> None:
    store = AgentActivityStore("parent", clock=lambda: next(clock))
    snapshots = []
    snapshots_after_removal = []
    store.add_listener(snapshots.append)
    store.add_listener(snapshots_after_removal.append)

    store.apply(_call("one", TaskArgs(task="first")))
    store.remove_listener(snapshots_after_removal.append)
    store.apply(_lifecycle("one", SubagentLifecycleState.RUNNING))

    assert len(snapshots) == 2
    assert len(snapshots_after_removal) == 1
    with pytest.raises(ValidationError):
        snapshots[-1].activities[0].task = "mutated"


def test_store_remains_bounded_and_prefers_evicting_terminal_rows(
    clock: Iterator[float],
) -> None:
    store = AgentActivityStore("parent", max_activities=2, clock=lambda: next(clock))
    store.apply(_call("active", TaskArgs(task="active")))
    store.apply(_call("done", TaskArgs(task="done")))
    store.apply(
        _result(
            "done", result=TaskResult(response="done", turns_used=1, completed=True)
        )
    )
    store.apply(_call("new", TaskArgs(task="new")))

    assert [activity.tool_call_id for activity in store.snapshot.activities] == [
        "active",
        "new",
    ]


def test_primary_agent_lifecycle_preserves_active_start_and_restarts_after_idle(
    clock: Iterator[float],
) -> None:
    store = AgentActivityStore("parent", clock=lambda: next(clock))
    snapshots = []
    store.add_listener(snapshots.append)

    assert store.update_primary("default", "Default", AgentRunState.RUNNING)
    assert store.update_primary(
        "default", "Default", AgentRunState.WORKING, "Calling model"
    )
    assert store.update_primary(
        "default", "Default", AgentRunState.ATTENTION, "Approval required"
    )

    active = store.snapshot.activities[0]
    assert active.tool_call_id == "primary:parent"
    assert active.task == "Current conversation"
    assert active.is_primary
    assert active.started_at == 0.0
    assert active.updated_at == 2.0

    assert store.update_primary("default", "Default", AgentRunState.IDLE)
    assert store.update_primary("default", "Default", AgentRunState.RUNNING)
    restarted = store.snapshot.activities[0]
    assert restarted.started_at == 4.0
    assert restarted.updated_at == 4.0
    assert len(snapshots) == 5


def test_duplicate_primary_update_is_a_noop(clock: Iterator[float]) -> None:
    store = AgentActivityStore("parent", clock=lambda: next(clock))
    snapshots = []
    store.add_listener(snapshots.append)

    assert store.update_primary("default", "Default", AgentRunState.IDLE)
    assert not store.update_primary("default", "Default", AgentRunState.IDLE)

    assert len(snapshots) == 1
    assert store.snapshot.activities[0].updated_at == 0.0


def test_primary_agent_is_exempt_from_subagent_bound(clock: Iterator[float]) -> None:
    store = AgentActivityStore("parent", max_activities=1, clock=lambda: next(clock))
    store.update_primary("default", "Default", AgentRunState.IDLE)
    store.apply(_call("one", TaskArgs(task="first")))
    store.apply(_call("two", TaskArgs(task="second")))

    activities = store.snapshot.activities
    assert [activity.tool_call_id for activity in activities] == [
        "primary:parent",
        "two",
    ]
    assert activities[0].is_primary
    assert not activities[1].is_primary


def test_primary_agent_rejects_subagent_only_states(clock: Iterator[float]) -> None:
    store = AgentActivityStore("parent", clock=lambda: next(clock))

    with pytest.raises(ValueError, match="Unsupported primary agent state"):
        store.update_primary("default", "Default", AgentRunState.COMPLETED)


@pytest.mark.parametrize(
    ("managed_state", "activity_state"),
    [
        (ManagedAgentState.STARTING, AgentRunState.REQUESTED),
        (ManagedAgentState.RUNNING, AgentRunState.RUNNING),
        (ManagedAgentState.WORKING, AgentRunState.WORKING),
        (ManagedAgentState.ATTENTION, AgentRunState.ATTENTION),
        (ManagedAgentState.IDLE, AgentRunState.IDLE),
        (ManagedAgentState.FAILED, AgentRunState.FAILED),
        (ManagedAgentState.STOPPED, AgentRunState.STOPPED),
    ],
)
def test_managed_lifecycle_maps_every_state(
    managed_state: ManagedAgentState,
    activity_state: AgentRunState,
    clock: Iterator[float],
) -> None:
    store = AgentActivityStore("parent", clock=lambda: next(clock))

    assert store.apply(_managed(managed_state))

    activity = store.snapshot.activities[0]
    assert activity.state is activity_state
    assert activity.managed_agent_id == "worker-1"
    assert activity.tool_call_id == "managed:worker-1"


def test_managed_lifecycle_projects_absolute_snapshot(clock: Iterator[float]) -> None:
    store = AgentActivityStore("parent", clock=lambda: next(clock))
    usage = LLMUsage(prompt_tokens=25, completion_tokens=7)

    store.apply(
        _managed(
            ManagedAgentState.FAILED,
            sequence=4,
            activity="Running tests",
            task="Verify the feature",
            last_response="Focused tests failed.",
            error="Assertion failed",
            usage=usage,
        )
    )

    activity = store.snapshot.activities[0]
    assert activity.task == "Verify the feature"
    assert activity.current_activity == "Running tests"
    assert activity.queued_messages == 2
    assert activity.last_response == "Focused tests failed."
    assert activity.error == "Assertion failed"
    assert activity.usage == usage
    assert activity.event_sequence == 4
    assert activity.child_session_id == "child-worker-1"


def test_managed_lifecycle_rejects_old_session_and_sequence(
    clock: Iterator[float],
) -> None:
    store = AgentActivityStore("parent", clock=lambda: next(clock))
    assert not store.apply(
        _managed(ManagedAgentState.RUNNING, parent_session_id="old-session")
    )
    assert store.apply(_managed(ManagedAgentState.RUNNING, sequence=3))
    assert not store.apply(_managed(ManagedAgentState.IDLE, sequence=3))
    assert not store.apply(_managed(ManagedAgentState.IDLE, sequence=2))
    assert store.snapshot.activities[0].state is AgentRunState.RUNNING


def test_managed_worker_can_restart_until_stopped(clock: Iterator[float]) -> None:
    store = AgentActivityStore("parent", clock=lambda: next(clock))

    store.apply(_managed(ManagedAgentState.FAILED, sequence=1))
    assert store.apply(_managed(ManagedAgentState.RUNNING, sequence=2))
    store.apply(_managed(ManagedAgentState.IDLE, sequence=3))
    assert store.apply(_managed(ManagedAgentState.RUNNING, sequence=4))
    store.apply(_managed(ManagedAgentState.STOPPED, sequence=5))
    assert not store.apply(_managed(ManagedAgentState.RUNNING, sequence=6))
    assert store.snapshot.activities[0].state is AgentRunState.STOPPED


def test_managed_and_task_internal_ids_do_not_overwrite(clock: Iterator[float]) -> None:
    store = AgentActivityStore("parent", clock=lambda: next(clock))
    store.apply(_call("managed:worker-1", TaskArgs(task="one-shot")))
    store.apply(_managed(ManagedAgentState.RUNNING))

    assert len(store.snapshot.activities) == 2
    assert {item.is_managed for item in store.snapshot.activities} == {False, True}
