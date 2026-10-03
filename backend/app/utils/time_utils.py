"""
time_utils.py

Time and date utilities: UTC-first helpers, formatting, simple cron
expression evaluation, scheduling helpers, and timezone conversion.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterator, List, Optional
from zoneinfo import ZoneInfo

ISO_FORMAT: str = "%Y-%m-%dT%H:%M:%S.%fZ"
DATE_FORMAT: str = "%Y-%m-%d"
DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """Convert a datetime to UTC, assuming naive datetimes are already UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_timezone(dt: datetime, tz_name: str) -> datetime:
    """Convert a datetime into the given IANA timezone."""
    aware = to_utc(dt)
    return aware.astimezone(ZoneInfo(tz_name))


def format_iso(dt: datetime) -> str:
    """Format a datetime as an ISO-8601 UTC string."""
    return to_utc(dt).strftime(ISO_FORMAT)


def parse_iso(text: str) -> datetime:
    """Parse an ISO-8601 string into a timezone-aware UTC datetime."""
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    return to_utc(dt)


def format_human(dt: datetime, tz_name: Optional[str] = None) -> str:
    """Format a datetime in a human-readable form, optionally in a given timezone."""
    target = to_timezone(dt, tz_name) if tz_name else to_utc(dt)
    return target.strftime(DATETIME_FORMAT)


def timestamp_ms(dt: Optional[datetime] = None) -> int:
    """Return Unix epoch milliseconds for the given (or current) datetime."""
    target = dt or utc_now()
    return int(to_utc(target).timestamp() * 1000)


def from_timestamp_ms(ms: int) -> datetime:
    """Construct a UTC datetime from Unix epoch milliseconds."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def elapsed_seconds(start: datetime, end: Optional[datetime] = None) -> float:
    """Return the number of seconds elapsed between two datetimes."""
    target_end = to_utc(end) if end else utc_now()
    return (target_end - to_utc(start)).total_seconds()


def is_expired(expires_at: datetime, *, now: Optional[datetime] = None) -> bool:
    """Return True if `expires_at` is in the past relative to `now`."""
    reference = to_utc(now) if now else utc_now()
    return to_utc(expires_at) <= reference


def add_seconds(dt: datetime, seconds: float) -> datetime:
    """Return a new datetime offset by the given number of seconds."""
    return to_utc(dt) + timedelta(seconds=seconds)


def floor_to_minute(dt: datetime) -> datetime:
    """Truncate a datetime down to the start of its minute."""
    aware = to_utc(dt)
    return aware.replace(second=0, microsecond=0)


def start_of_day(dt: datetime) -> datetime:
    """Return the start (00:00:00) of the given datetime's UTC day."""
    aware = to_utc(dt)
    return aware.replace(hour=0, minute=0, second=0, microsecond=0)


def end_of_day(dt: datetime) -> datetime:
    """Return the end (23:59:59.999999) of the given datetime's UTC day."""
    return start_of_day(dt) + timedelta(days=1, microseconds=-1)


# ---------------------------------------------------------------------------
# Minimal cron support (standard 5-field cron: m h dom mon dow)
# ---------------------------------------------------------------------------
class CronParseError(Exception):
    """Raised when a cron expression cannot be parsed."""


def _parse_field(field: str, min_val: int, max_val: int) -> List[int]:
    if field == "*":
        return list(range(min_val, max_val + 1))
    values: set[int] = set()
    for part in field.split(","):
        if "/" in part:
            range_part, step_str = part.split("/")
            step = int(step_str)
            if range_part == "*":
                start, end = min_val, max_val
            elif "-" in range_part:
                start_str, end_str = range_part.split("-")
                start, end = int(start_str), int(end_str)
            else:
                start, end = int(range_part), max_val
            values.update(range(start, end + 1, step))
        elif "-" in part:
            start_str, end_str = part.split("-")
            values.update(range(int(start_str), int(end_str) + 1))
        else:
            values.add(int(part))
    out_of_range = [v for v in values if v < min_val or v > max_val]
    if out_of_range:
        raise CronParseError(f"Values out of range {min_val}-{max_val}: {out_of_range}")
    return sorted(values)


class CronSchedule:
    """Minimal 5-field cron expression evaluator (minute hour dom month dow)."""

    def __init__(self, expression: str) -> None:
        fields = expression.strip().split()
        if len(fields) != 5:
            raise CronParseError(f"Expected 5 fields, got {len(fields)}: {expression}")
        minute, hour, dom, month, dow = fields
        self.minutes = _parse_field(minute, 0, 59)
        self.hours = _parse_field(hour, 0, 23)
        self.days_of_month = _parse_field(dom, 1, 31)
        self.months = _parse_field(month, 1, 12)
        self.days_of_week = _parse_field(dow, 0, 6)
        self.expression = expression

    def matches(self, dt: datetime) -> bool:
        """Return True if the given UTC datetime matches this cron schedule."""
        aware = to_utc(dt)
        return (
            aware.minute in self.minutes
            and aware.hour in self.hours
            and aware.day in self.days_of_month
            and aware.month in self.months
            and (aware.weekday() + 1) % 7 in self.days_of_week
        )

    def next_run_after(self, dt: datetime, *, max_lookahead_minutes: int = 10_080) -> Optional[datetime]:
        """Find the next matching datetime after `dt`, scanning minute-by-minute."""
        cursor = floor_to_minute(to_utc(dt)) + timedelta(minutes=1)
        for _ in range(max_lookahead_minutes):
            if self.matches(cursor):
                return cursor
            cursor += timedelta(minutes=1)
        return None

    def iter_next(self, dt: datetime, count: int) -> Iterator[datetime]:
        """Yield the next `count` matching run times after `dt`."""
        cursor = dt
        produced = 0
        while produced < count:
            nxt = self.next_run_after(cursor)
            if nxt is None:
                return
            yield nxt
            cursor = nxt
            produced += 1
