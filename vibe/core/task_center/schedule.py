from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from vibe.core.task_center.models import (
    DailyTrigger,
    IntervalTrigger,
    TaskTrigger,
    WeeklyTrigger,
)


class TaskScheduleError(ValueError):
    pass


def validate_trigger_timezone(trigger: TaskTrigger) -> None:
    if not isinstance(trigger, DailyTrigger | WeeklyTrigger):
        return
    try:
        ZoneInfo(trigger.timezone)
    except ZoneInfoNotFoundError as error:
        raise TaskScheduleError(f"Unknown timezone: {trigger.timezone}") from error


def next_occurrence(
    trigger: TaskTrigger, *, after: datetime, created_at: datetime
) -> datetime | None:
    after = _aware_utc(after)
    created_at = _aware_utc(created_at)
    match trigger:
        case IntervalTrigger():
            anchor = trigger.anchor_at or created_at
            elapsed = (after - anchor).total_seconds()
            intervals = max(0, int(elapsed // trigger.interval_seconds) + 1)
            return anchor + timedelta(seconds=intervals * trigger.interval_seconds)
        case DailyTrigger():
            return _next_wall_clock(trigger, after, weekdays=None)
        case WeeklyTrigger():
            return _next_wall_clock(
                trigger, after, weekdays={int(day) for day in trigger.weekdays}
            )
        case _:
            return None


def _next_wall_clock(
    trigger: DailyTrigger | WeeklyTrigger, after: datetime, *, weekdays: set[int] | None
) -> datetime:
    try:
        zone = ZoneInfo(trigger.timezone)
    except ZoneInfoNotFoundError as error:
        raise TaskScheduleError(f"Unknown timezone: {trigger.timezone}") from error

    local_after = after.astimezone(zone)
    for offset in range(15):
        candidate_date = local_after.date() + timedelta(days=offset)
        if weekdays is not None and candidate_date.weekday() not in weekdays:
            continue
        candidate = _resolve_local_wall_time(candidate_date, trigger.at, zone)
        if (
            weekdays is not None
            and candidate.astimezone(zone).weekday() not in weekdays
        ):
            continue
        candidate_utc = candidate.astimezone(UTC)
        if candidate_utc > after:
            return candidate_utc
    raise TaskScheduleError("Unable to compute next weekly occurrence")


def _resolve_local_wall_time(
    candidate_date: date, local_time: time, zone: ZoneInfo
) -> datetime:
    naive = datetime.combine(candidate_date, local_time)
    candidate = naive.replace(tzinfo=zone, fold=0)
    round_trip = candidate.astimezone(UTC).astimezone(zone)
    if round_trip.replace(tzinfo=None) != naive:
        return round_trip
    return candidate


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise TaskScheduleError("timestamp must include a UTC offset")
    return value.astimezone(UTC)
