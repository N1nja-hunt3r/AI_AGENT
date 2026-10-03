from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class LimitKind(str, Enum):
    REQUEST = "request"
    TOKEN = "token"
    IP = "ip"


class RateLimitError(Exception):
    pass


@dataclass
class RateLimitConfig:
    max_amount: int
    window_seconds: float
    burst_amount: int = 0

    def __post_init__(self) -> None:
        if self.max_amount <= 0:
            raise ValueError("max_amount must be positive")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.burst_amount < 0:
            raise ValueError("burst_amount must be non-negative")


@dataclass
class RateLimitDecision:
    allowed: bool
    key: str
    kind: LimitKind
    remaining: int
    limit: int
    retry_after_seconds: float = 0.0
    window_seconds: float = 0.0


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class _SlidingWindowCounter:
    """Sliding-window log with optional burst allowance on top of the base limit."""

    __slots__ = ("events", "config")

    def __init__(self, config: RateLimitConfig) -> None:
        self.events: deque[tuple[float, int]] = deque()
        self.config = config

    def _prune(self, now: float) -> None:
        cutoff = now - self.config.window_seconds
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def current_amount(self, now: Optional[float] = None) -> int:
        now = now if now is not None else time.time()
        self._prune(now)
        return sum(amount for _, amount in self.events)

    def effective_limit(self) -> int:
        return self.config.max_amount + self.config.burst_amount

    def try_consume(self, amount: int, now: Optional[float] = None) -> tuple[bool, int, float]:
        now = now if now is not None else time.time()
        self._prune(now)
        used = sum(a for _, a in self.events)
        limit = self.effective_limit()
        if used + amount > limit:
            retry_after = 0.0
            if self.events:
                oldest_ts, oldest_amt = self.events[0]
                retry_after = max(0.0, (oldest_ts + self.config.window_seconds) - now)
            return False, max(0, limit - used), retry_after
        self.events.append((now, amount))
        return True, max(0, limit - used - amount), 0.0

    def reset(self) -> None:
        self.events.clear()


class RateLimiter:
    """Async-safe sliding-window rate limiter for requests, tokens, and IPs."""

    def __init__(
        self,
        *,
        request_config: Optional[RateLimitConfig] = None,
        token_config: Optional[RateLimitConfig] = None,
        ip_config: Optional[RateLimitConfig] = None,
    ) -> None:
        self.default_configs: dict[LimitKind, RateLimitConfig] = {
            LimitKind.REQUEST: request_config or RateLimitConfig(max_amount=60, window_seconds=60.0, burst_amount=10),
            LimitKind.TOKEN: token_config or RateLimitConfig(max_amount=100_000, window_seconds=60.0, burst_amount=20_000),
            LimitKind.IP: ip_config or RateLimitConfig(max_amount=120, window_seconds=60.0, burst_amount=20),
        }
        self._counters: dict[tuple[LimitKind, str], _SlidingWindowCounter] = {}
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._allow_checks = 0
        self._consume_calls = 0
        self._denied_count = 0

    def _get_counter(self, kind: LimitKind, key: str, config: Optional[RateLimitConfig]) -> _SlidingWindowCounter:
        cache_key = (kind, key)
        counter = self._counters.get(cache_key)
        if counter is None:
            counter = _SlidingWindowCounter(config or self.default_configs[kind])
            self._counters[cache_key] = counter
        elif config is not None:
            counter.config = config
        return counter

    def allow(
        self,
        key: str,
        *,
        kind: LimitKind = LimitKind.REQUEST,
        amount: int = 1,
        config: Optional[RateLimitConfig] = None,
    ) -> RateLimitDecision:
        self._allow_checks += 1
        counter = self._get_counter(kind, key, config)
        now = time.time()
        used = counter.current_amount(now)
        limit = counter.effective_limit()
        remaining = max(0, limit - used)
        allowed = (used + amount) <= limit
        retry_after = 0.0
        if not allowed and counter.events:
            oldest_ts, _ = counter.events[0]
            retry_after = max(0.0, (oldest_ts + counter.config.window_seconds) - now)
        return RateLimitDecision(
            allowed=allowed, key=key, kind=kind, remaining=remaining, limit=limit,
            retry_after_seconds=round(retry_after, 3), window_seconds=counter.config.window_seconds,
        )

    def consume(
        self,
        key: str,
        *,
        kind: LimitKind = LimitKind.REQUEST,
        amount: int = 1,
        config: Optional[RateLimitConfig] = None,
    ) -> RateLimitDecision:
        self._consume_calls += 1
        counter = self._get_counter(kind, key, config)
        ok, remaining, retry_after = counter.try_consume(amount)
        if not ok:
            self._denied_count += 1
        return RateLimitDecision(
            allowed=ok, key=key, kind=kind, remaining=remaining, limit=counter.effective_limit(),
            retry_after_seconds=round(retry_after, 3), window_seconds=counter.config.window_seconds,
        )

    def reset(self, key: Optional[str] = None, *, kind: Optional[LimitKind] = None) -> int:
        if key is None and kind is None:
            count = len(self._counters)
            self._counters.clear()
            return count
        count = 0
        for cache_key in list(self._counters.keys()):
            k, ck = cache_key
            if (kind is None or k == kind) and (key is None or ck == key):
                self._counters[cache_key].reset()
                del self._counters[cache_key]
                count += 1
        return count

    def remaining(self, key: str, *, kind: LimitKind = LimitKind.REQUEST) -> int:
        counter = self._counters.get((kind, key))
        if counter is None:
            return self.default_configs[kind].max_amount + self.default_configs[kind].burst_amount
        limit = counter.effective_limit()
        return max(0, limit - counter.current_amount())

    async def allow_async(
        self, key: str, *, kind: LimitKind = LimitKind.REQUEST, amount: int = 1, config: Optional[RateLimitConfig] = None
    ) -> RateLimitDecision:
        async with self._lock:
            return self.allow(key, kind=kind, amount=amount, config=config)

    async def consume_async(
        self, key: str, *, kind: LimitKind = LimitKind.REQUEST, amount: int = 1, config: Optional[RateLimitConfig] = None
    ) -> RateLimitDecision:
        async with self._lock:
            return self.consume(key, kind=kind, amount=amount, config=config)

    async def reset_async(self, key: Optional[str] = None, *, kind: Optional[LimitKind] = None) -> int:
        async with self._lock:
            return self.reset(key, kind=kind)

    def check_ip(self, ip_address: str, *, amount: int = 1) -> RateLimitDecision:
        return self.consume(ip_address, kind=LimitKind.IP, amount=amount)

    def check_request(self, client_id: str, *, amount: int = 1) -> RateLimitDecision:
        return self.consume(client_id, kind=LimitKind.REQUEST, amount=amount)

    def check_tokens(self, client_id: str, *, tokens: int) -> RateLimitDecision:
        return self.consume(client_id, kind=LimitKind.TOKEN, amount=tokens)

    def health_check(self) -> HealthStatus:
        try:
            probe_key = "__health_check__"
            decision = self.allow(probe_key, kind=LimitKind.REQUEST, amount=0)
            self.reset(probe_key, kind=LimitKind.REQUEST)
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "tracked_keys": len(self._counters),
                "allow_checks": self._allow_checks,
                "consume_calls": self._consume_calls,
                "denied_count": self._denied_count,
                "request_limit": self.default_configs[LimitKind.REQUEST].max_amount,
                "token_limit": self.default_configs[LimitKind.TOKEN].max_amount,
                "ip_limit": self.default_configs[LimitKind.IP].max_amount,
            }
            return HealthStatus(healthy=decision.allowed, component="rate_limit", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="rate_limit", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        async with self._lock:
            return self.health_check()


__all__ = [
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitDecision",
    "LimitKind",
    "RateLimitError",
    "HealthStatus",
]
