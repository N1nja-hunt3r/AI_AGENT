from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    _HAS_ZONEINFO = True
except ImportError:
    _HAS_ZONEINFO = False


class CronError(Exception):
    pass


class InvalidCronExpressionError(CronError):
    pass


class InvalidTimezoneError(CronError):
    pass


_FIELD_RANGES: dict[str, tuple[int, int]] = {
    "minute": (0, 59),
    "hour": (0, 23),
    "day": (1, 31),
    "month": (1, 12),
    "weekday": (0, 6),
}

_MONTH_ALIASES: dict[str, str] = {
    "jan": "1", "feb": "2", "mar": "3", "apr": "4", "may": "5", "jun": "6",
    "jul": "7", "aug": "8", "sep": "9", "oct": "10", "nov": "11", "dec": "12",
}

_WEEKDAY_ALIASES: dict[str, str] = {
    "sun": "0", "mon": "1", "tue": "2", "wed": "3", "thu": "4", "fri": "5", "sat": "6",
}


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ParsedCron:
    expression: str
    minutes: frozenset[int]
    hours: frozenset[int]
    days: frozenset[int]
    months: frozenset[int]
    weekdays: frozenset[int]
    day_is_wildcard: bool
    weekday_is_wildcard: bool


def _expand_field(raw: str, field_name: str, aliases: Optional[dict[str, str]] = None) -> frozenset[int]:
    lo, hi = _FIELD_RANGES[field_name]
    values: set[int] = set()
    raw = raw.strip().lower()

    if aliases:
        for alias, num in aliases.items():
            raw = raw.replace(alias, num)

    for part in raw.split(","):
        part = part.strip()
        if not part:
            raise InvalidCronExpressionError(f"empty segment in field '{field_name}'")

        step = 1
        if "/" in part:
            base, step_str = part.split("/", 1)
            try:
                step = int(step_str)
            except ValueError as exc:
                raise InvalidCronExpressionError(f"invalid step '{step_str}' in field '{field_name}'") from exc
            if step <= 0:
                raise InvalidCronExpressionError(f"step must be positive in field '{field_name}'")
        else:
            base = part

        if base == "*":
            range_lo, range_hi = lo, hi
        elif "-" in base:
            try:
                range_lo_s, range_hi_s = base.split("-", 1)
                range_lo, range_hi = int(range_lo_s), int(range_hi_s)
            except ValueError as exc:
                raise InvalidCronExpressionError(f"invalid range '{base}' in field '{field_name}'") from exc
        else:
            try:
                range_lo = range_hi = int(base)
            except ValueError as exc:
                raise InvalidCronExpressionError(f"invalid value '{base}' in field '{field_name}'") from exc

        if range_lo < lo or range_hi > hi or range_lo > range_hi:
            raise InvalidCronExpressionError(
                f"value(s) out of range for field '{field_name}' ({lo}-{hi}): got '{part}'"
            )

        for v in range(range_lo, range_hi + 1, step):
            values.add(v if field_name != "weekday" else v % 7)

    return frozenset(values)


class CronManager:
    """Parses, validates, and schedules standard 5-field cron expressions with timezone support."""

    def __init__(self) -> None:
        self._created_at = time.time()
        self._parse_count = 0
        self._validate_count = 0
        self._next_run_count = 0
        self._cache: dict[str, ParsedCron] = {}

    def parse(self, expression: str) -> ParsedCron:
        self._parse_count += 1
        cached = self._cache.get(expression)
        if cached is not None:
            return cached

        fields = expression.strip().split()
        if len(fields) != 5:
            raise InvalidCronExpressionError(
                f"cron expression must have exactly 5 fields (minute hour day month weekday), got {len(fields)}: '{expression}'"
            )

        minute_raw, hour_raw, day_raw, month_raw, weekday_raw = fields

        minutes = _expand_field(minute_raw, "minute")
        hours = _expand_field(hour_raw, "hour")
        days = _expand_field(day_raw, "day")
        months = _expand_field(month_raw, "month", aliases=_MONTH_ALIASES)
        weekdays = _expand_field(weekday_raw, "weekday", aliases=_WEEKDAY_ALIASES)

        parsed = ParsedCron(
            expression=expression, minutes=minutes, hours=hours, days=days, months=months, weekdays=weekdays,
            day_is_wildcard=(day_raw.strip() == "*"), weekday_is_wildcard=(weekday_raw.strip() == "*"),
        )
        self._cache[expression] = parsed
        return parsed

    def validate(self, expression: str) -> bool:
        self._validate_count += 1
        try:
            self.parse(expression)
            return True
        except CronError:
            return False

    def _resolve_timezone(self, timezone: Optional[str]) -> Optional[Any]:
        if timezone is None:
            return None
        if not _HAS_ZONEINFO:
            raise InvalidTimezoneError("zoneinfo module is unavailable in this environment")
        try:
            return ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise InvalidTimezoneError(f"unknown timezone '{timezone}'") from exc
        except Exception as exc:
            raise InvalidTimezoneError(f"invalid timezone '{timezone}': {exc}") from exc

    def next_run(self, expression: str, after: Optional[float] = None, timezone: Optional[str] = None) -> float:
        self._next_run_count += 1
        parsed = self.parse(expression)
        tz = self._resolve_timezone(timezone)

        start_epoch = after if after is not None else time.time()
        dt = datetime.fromtimestamp(start_epoch, tz=tz) if tz is not None else datetime.fromtimestamp(start_epoch)
        dt = dt.replace(second=0, microsecond=0) + timedelta(minutes=1)

        max_iterations = 60 * 24 * 366 * 4
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            if dt.month not in parsed.months:
                dt = self._advance_to_next_month(dt)
                continue

            day_matches = dt.day in parsed.days
            weekday_matches = (dt.weekday() + 1) % 7 in parsed.weekdays

            if parsed.day_is_wildcard and parsed.weekday_is_wildcard:
                day_ok = True
            elif parsed.day_is_wildcard:
                day_ok = weekday_matches
            elif parsed.weekday_is_wildcard:
                day_ok = day_matches
            else:
                day_ok = day_matches or weekday_matches

            if not day_ok:
                dt = (dt + timedelta(days=1)).replace(hour=0, minute=0)
                continue

            if dt.hour not in parsed.hours:
                dt = self._advance_to_next_hour(dt)
                continue

            if dt.minute not in parsed.minutes:
                dt = dt + timedelta(minutes=1)
                continue

            return dt.timestamp()

        raise CronError(f"could not find next run time for expression '{expression}' within search bounds")

    def _advance_to_next_month(self, dt: datetime) -> datetime:
        year, month = dt.year, dt.month
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
        return dt.replace(year=year, month=month, day=1, hour=0, minute=0)

    def _advance_to_next_hour(self, dt: datetime) -> datetime:
        return (dt.replace(minute=0) + timedelta(hours=1))

    def next_n_runs(self, expression: str, n: int, after: Optional[float] = None, timezone: Optional[str] = None) -> list[float]:
        results: list[float] = []
        cursor = after
        for _ in range(n):
            run_time = self.next_run(expression, after=cursor, timezone=timezone)
            results.append(run_time)
            cursor = run_time
        return results

    def describe(self, expression: str) -> str:
        parsed = self.parse(expression)
        parts = []
        parts.append(f"minutes={sorted(parsed.minutes)}" if len(parsed.minutes) < 60 else "every minute")
        parts.append(f"hours={sorted(parsed.hours)}" if len(parsed.hours) < 24 else "every hour")
        if not parsed.day_is_wildcard:
            parts.append(f"days={sorted(parsed.days)}")
        if len(parsed.months) < 12:
            parts.append(f"months={sorted(parsed.months)}")
        if not parsed.weekday_is_wildcard:
            parts.append(f"weekdays={sorted(parsed.weekdays)}")
        return ", ".join(parts)

    def health_check(self) -> HealthStatus:
        try:
            probe_next = self.next_run("*/5 * * * *")
            healthy = probe_next > time.time() - 1
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "parse_count": self._parse_count,
                "validate_count": self._validate_count,
                "next_run_count": self._next_run_count,
                "cache_size": len(self._cache),
                "zoneinfo_available": _HAS_ZONEINFO,
            }
            return HealthStatus(healthy=healthy, component="cron_manager", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="cron_manager", details={"error": str(exc)})


__all__ = [
    "CronManager",
    "ParsedCron",
    "CronError",
    "InvalidCronExpressionError",
    "InvalidTimezoneError",
    "HealthStatus",
]
