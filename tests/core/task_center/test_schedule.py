from __future__ import annotations

from datetime import UTC, datetime, time

from vibe.core.task_center import (
    DailyTrigger,
    IntervalTrigger,
    ManualTrigger,
    Weekday,
    WeeklyTrigger,
    next_occurrence,
)


def test_interval_uses_stable_anchor_without_drift() -> None:
    anchor = datetime(2026, 1, 1, 12, tzinfo=UTC)
    trigger = IntervalTrigger(interval_seconds=300, anchor_at=anchor)

    assert next_occurrence(
        trigger, after=datetime(2026, 1, 1, 12, 12, tzinfo=UTC), created_at=anchor
    ) == datetime(2026, 1, 1, 12, 15, tzinfo=UTC)
    assert next_occurrence(trigger, after=anchor, created_at=anchor) == datetime(
        2026, 1, 1, 12, 5, tzinfo=UTC
    )


def test_manual_trigger_has_no_next_occurrence() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert next_occurrence(ManualTrigger(), after=now, created_at=now) is None


def test_daily_schedule_uses_requested_timezone() -> None:
    trigger = DailyTrigger(at=time(9), timezone="Europe/Paris")

    assert next_occurrence(
        trigger,
        after=datetime(2026, 1, 15, 7, 30, tzinfo=UTC),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    ) == datetime(2026, 1, 15, 8, 0, tzinfo=UTC)

    assert next_occurrence(
        trigger,
        after=datetime(2026, 1, 15, 8, 0, tzinfo=UTC),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    ) == datetime(2026, 1, 16, 8, 0, tzinfo=UTC)


def test_nonexistent_daily_wall_time_normalizes_forward() -> None:
    trigger = DailyTrigger(at=time(2, 30), timezone="Europe/Paris")

    assert next_occurrence(
        trigger,
        after=datetime(2026, 3, 29, 0, 0, tzinfo=UTC),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    ) == datetime(2026, 3, 29, 1, 30, tzinfo=UTC)


def test_weekly_schedule_selects_next_allowed_day() -> None:
    trigger = WeeklyTrigger(
        at=time(10), timezone="UTC", weekdays=(Weekday.MONDAY, Weekday.FRIDAY)
    )

    assert next_occurrence(
        trigger,
        after=datetime(2026, 1, 7, 12, tzinfo=UTC),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    ) == datetime(2026, 1, 9, 10, tzinfo=UTC)


def test_weekly_schedule_rechecks_weekday_after_skipped_local_date() -> None:
    trigger = WeeklyTrigger(
        at=time(9), timezone="Pacific/Apia", weekdays=(Weekday.FRIDAY,)
    )

    occurrence = next_occurrence(
        trigger,
        after=datetime(2011, 12, 29, 20, tzinfo=UTC),
        created_at=datetime(2011, 1, 1, tzinfo=UTC),
    )

    assert occurrence == datetime(2012, 1, 5, 19, tzinfo=UTC)
