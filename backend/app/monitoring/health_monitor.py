from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Union


class HealthState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentKind(str, Enum):
    CAPABILITY = "capability"
    AGENT = "agent"
    DATABASE = "database"
    SERVICE = "service"
    SECURITY = "security"
    OTHER = "other"


class HealthMonitorError(Exception):
    pass


class CheckNotFoundError(HealthMonitorError):
    pass


CheckFn = Callable[[], Union[bool, dict[str, Any], "HealthStatus"]]
AsyncCheckFn = Callable[[], Awaitable[Union[bool, dict[str, Any], "HealthStatus"]]]


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ComponentCheck:
    name: str
    kind: ComponentKind
    check_fn: Optional[CheckFn] = None
    async_check_fn: Optional[AsyncCheckFn] = None
    critical: bool = True
    timeout_seconds: float = 5.0


@dataclass
class ComponentResult:
    name: str
    kind: ComponentKind
    state: HealthState
    critical: bool
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: float = field(default_factory=time.time)
    error: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "state": self.state.value,
            "critical": self.critical,
            "details": self.details,
            "checked_at": self.checked_at,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 6),
        }


@dataclass
class SystemReport:
    overall_state: HealthState
    components: list[ComponentResult]
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_state": self.overall_state.value,
            "generated_at": self.generated_at,
            "components": [c.to_dict() for c in self.components],
            "healthy_count": sum(1 for c in self.components if c.state == HealthState.HEALTHY),
            "degraded_count": sum(1 for c in self.components if c.state == HealthState.DEGRADED),
            "unhealthy_count": sum(1 for c in self.components if c.state == HealthState.UNHEALTHY),
            "total_count": len(self.components),
        }


def _normalize_result(raw: Union[bool, dict[str, Any], HealthStatus, None]) -> tuple[bool, dict[str, Any]]:
    if raw is None:
        return False, {"error": "check returned None"}
    if isinstance(raw, HealthStatus):
        return raw.healthy, dict(raw.details)
    if isinstance(raw, bool):
        return raw, {}
    if isinstance(raw, dict):
        healthy = bool(raw.get("healthy", False))
        return healthy, {k: v for k, v in raw.items() if k != "healthy"}
    return False, {"error": f"unrecognized check result type: {type(raw).__name__}"}


class HealthMonitor:
    """Aggregates health checks across capabilities, agents, databases, services, and security modules."""

    def __init__(self) -> None:
        self._checks: dict[str, ComponentCheck] = {}
        self._last_results: dict[str, ComponentResult] = {}
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._check_run_count = 0

    def register_check(
        self,
        name: str,
        *,
        kind: ComponentKind = ComponentKind.OTHER,
        check_fn: Optional[CheckFn] = None,
        async_check_fn: Optional[AsyncCheckFn] = None,
        critical: bool = True,
        timeout_seconds: float = 5.0,
    ) -> ComponentCheck:
        if check_fn is None and async_check_fn is None:
            raise HealthMonitorError("must provide check_fn or async_check_fn")
        registration = ComponentCheck(
            name=name, kind=kind, check_fn=check_fn, async_check_fn=async_check_fn,
            critical=critical, timeout_seconds=timeout_seconds,
        )
        self._checks[name] = registration
        return registration

    def register_capability(self, name: str, check_fn: CheckFn, *, critical: bool = True) -> ComponentCheck:
        return self.register_check(name, kind=ComponentKind.CAPABILITY, check_fn=check_fn, critical=critical)

    def register_agent(self, name: str, check_fn: CheckFn, *, critical: bool = True) -> ComponentCheck:
        return self.register_check(name, kind=ComponentKind.AGENT, check_fn=check_fn, critical=critical)

    def register_database(self, name: str, check_fn: CheckFn, *, critical: bool = True) -> ComponentCheck:
        return self.register_check(name, kind=ComponentKind.DATABASE, check_fn=check_fn, critical=critical)

    def register_service(self, name: str, check_fn: CheckFn, *, critical: bool = True) -> ComponentCheck:
        return self.register_check(name, kind=ComponentKind.SERVICE, check_fn=check_fn, critical=critical)

    def register_security(self, name: str, check_fn: CheckFn, *, critical: bool = True) -> ComponentCheck:
        return self.register_check(name, kind=ComponentKind.SECURITY, check_fn=check_fn, critical=critical)

    def unregister(self, name: str) -> bool:
        self._last_results.pop(name, None)
        return self._checks.pop(name, None) is not None

    def check(self, name: str) -> ComponentResult:
        self._check_run_count += 1
        registration = self._checks.get(name)
        if registration is None:
            raise CheckNotFoundError(f"no health check registered under name '{name}'")
        if registration.check_fn is None:
            raise HealthMonitorError(f"check '{name}' has no synchronous check_fn registered")

        started = time.time()
        try:
            raw = registration.check_fn()
            healthy, details = _normalize_result(raw)
            state = HealthState.HEALTHY if healthy else (
                HealthState.UNHEALTHY if registration.critical else HealthState.DEGRADED
            )
            result = ComponentResult(
                name=name, kind=registration.kind, state=state, critical=registration.critical,
                details=details, duration_seconds=time.time() - started,
            )
        except Exception as exc:
            result = ComponentResult(
                name=name, kind=registration.kind, state=HealthState.UNHEALTHY, critical=registration.critical,
                details={}, error=str(exc), duration_seconds=time.time() - started,
            )

        self._last_results[name] = result
        return result

    def check_all(self) -> list[ComponentResult]:
        return [self.check(name) for name in self._checks if self._checks[name].check_fn is not None]

    async def check_async(self, name: str) -> ComponentResult:
        self._check_run_count += 1
        registration = self._checks.get(name)
        if registration is None:
            raise CheckNotFoundError(f"no health check registered under name '{name}'")

        started = time.time()
        try:
            if registration.async_check_fn is not None:
                raw = await asyncio.wait_for(registration.async_check_fn(), timeout=registration.timeout_seconds)
            elif registration.check_fn is not None:
                raw = await asyncio.wait_for(
                    asyncio.to_thread(registration.check_fn), timeout=registration.timeout_seconds
                )
            else:
                raise HealthMonitorError(f"check '{name}' has no check function registered")

            healthy, details = _normalize_result(raw)
            state = HealthState.HEALTHY if healthy else (
                HealthState.UNHEALTHY if registration.critical else HealthState.DEGRADED
            )
            result = ComponentResult(
                name=name, kind=registration.kind, state=state, critical=registration.critical,
                details=details, duration_seconds=time.time() - started,
            )
        except asyncio.TimeoutError:
            result = ComponentResult(
                name=name, kind=registration.kind, state=HealthState.UNHEALTHY, critical=registration.critical,
                details={}, error=f"check timed out after {registration.timeout_seconds}s",
                duration_seconds=time.time() - started,
            )
        except Exception as exc:
            result = ComponentResult(
                name=name, kind=registration.kind, state=HealthState.UNHEALTHY, critical=registration.critical,
                details={}, error=str(exc), duration_seconds=time.time() - started,
            )

        async with self._lock:
            self._last_results[name] = result
        return result

    async def check_all_async(self) -> list[ComponentResult]:
        names = list(self._checks.keys())
        return await asyncio.gather(*(self.check_async(name) for name in names))

    def _compute_overall(self, results: list[ComponentResult]) -> HealthState:
        if not results:
            return HealthState.UNKNOWN
        if any(r.state == HealthState.UNHEALTHY and r.critical for r in results):
            return HealthState.UNHEALTHY
        if any(r.state in (HealthState.UNHEALTHY, HealthState.DEGRADED) for r in results):
            return HealthState.DEGRADED
        return HealthState.HEALTHY

    def report(self) -> SystemReport:
        results = self.check_all()
        return SystemReport(overall_state=self._compute_overall(results), components=results)

    async def report_async(self) -> SystemReport:
        results = await self.check_all_async()
        return SystemReport(overall_state=self._compute_overall(results), components=results)

    def status(self, name: Optional[str] = None) -> Union[ComponentResult, SystemReport, None]:
        if name is not None:
            return self._last_results.get(name)
        results = list(self._last_results.values())
        return SystemReport(overall_state=self._compute_overall(results), components=results)

    def list_components(self, *, kind: Optional[ComponentKind] = None) -> list[str]:
        names = list(self._checks.keys())
        if kind is not None:
            names = [n for n in names if self._checks[n].kind == kind]
        return names

    def health_check(self) -> HealthStatus:
        try:
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "registered_check_count": len(self._checks),
                "last_result_count": len(self._last_results),
                "check_run_count": self._check_run_count,
                "components_by_kind": {
                    kind.value: len([c for c in self._checks.values() if c.kind == kind])
                    for kind in ComponentKind
                },
            }
            return HealthStatus(healthy=True, component="health_monitor", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="health_monitor", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        return await asyncio.to_thread(self.health_check)


__all__ = [
    "HealthMonitor",
    "HealthState",
    "ComponentKind",
    "ComponentCheck",
    "ComponentResult",
    "SystemReport",
    "HealthMonitorError",
    "CheckNotFoundError",
    "HealthStatus",
]
