from __future__ import annotations

from datetime import UTC, datetime, time

from pydantic import ValidationError
import pytest

from vibe.core.task_center import (
    DailyTrigger,
    IntervalTrigger,
    TaskCreate,
    TaskExecutionAuthorization,
    TaskExecutionDisposition,
    TaskExecutionResult,
    TaskUpdate,
    Weekday,
    WeeklyTrigger,
)


def test_trigger_union_rejects_unknown_fields_and_kinds() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TaskCreate.model_validate({
            "title": "Review failures",
            "trigger": {"kind": "manual", "cron": "* * * * *"},
        })

    with pytest.raises(ValidationError, match="union_tag_invalid"):
        TaskCreate.model_validate({
            "title": "Review failures",
            "trigger": {"kind": "cron", "expression": "* * * * *"},
        })


def test_task_text_and_assignment_are_normalized_and_validated() -> None:
    task = TaskCreate(
        title="  Review failures  ",
        details="  Inspect the latest failures.  ",
        assigned_profile="explore-agent",
    )

    assert task.title == "Review failures"
    assert task.details == "Inspect the latest failures."
    assert task.assigned_profile == "explore-agent"

    with pytest.raises(ValidationError, match="assigned_profile"):
        TaskCreate(title="Task", assigned_profile="bad profile")

    with pytest.raises(ValidationError, match="value must not be blank"):
        TaskCreate(title="   ")


def test_wall_clock_triggers_require_real_timezones_and_unique_weekdays() -> None:
    trigger = DailyTrigger(at=time(9, 30), timezone="Europe/Paris")
    assert trigger.timezone == "Europe/Paris"

    with pytest.raises(ValidationError, match="Unknown timezone"):
        DailyTrigger(at=time(9), timezone="Mars/Olympus")

    with pytest.raises(ValidationError, match="duplicates"):
        WeeklyTrigger(
            at=time(9), timezone="UTC", weekdays=(Weekday.MONDAY, Weekday.MONDAY)
        )

    with pytest.raises(ValidationError, match="must not include a UTC offset"):
        DailyTrigger(at=time(9, tzinfo=UTC), timezone="UTC")


def test_interval_anchor_must_be_aware() -> None:
    with pytest.raises(ValidationError, match="UTC offset"):
        IntervalTrigger(interval_seconds=60, anchor_at=datetime(2026, 1, 1, 12, 0))


def test_task_update_distinguishes_clear_assignment_from_missing_fields() -> None:
    clear_assignment = TaskUpdate(assigned_profile=None)
    assert clear_assignment.model_fields_set == {"assigned_profile"}

    with pytest.raises(ValidationError, match="at least one"):
        TaskUpdate()

    with pytest.raises(ValidationError, match="title must not be null"):
        TaskUpdate(title=None)


def test_automatic_execution_requires_explicit_always_proof() -> None:
    with pytest.raises(ValidationError, match="requires explicit ALWAYS"):
        TaskExecutionResult(
            disposition=TaskExecutionDisposition.STARTED,
            authorization=TaskExecutionAuthorization.ASK,
        )

    started = TaskExecutionResult(
        disposition=TaskExecutionDisposition.STARTED,
        authorization=TaskExecutionAuthorization.ALWAYS,
        managed_agent_id="worker-1",
    )
    assert started.managed_agent_id == "worker-1"

    with pytest.raises(ValidationError, match="requires an error"):
        TaskExecutionResult(disposition=TaskExecutionDisposition.BLOCKED)

    with pytest.raises(ValidationError, match="must not be blank"):
        TaskExecutionResult(disposition=TaskExecutionDisposition.BLOCKED, error="   ")
