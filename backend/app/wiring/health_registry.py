"""
health_registry.py - Global health registry and dashboard for all services.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"

class ServiceType(str, Enum):
    LLM = "llm"
    MEMORY = "memory"
    RAG = "rag"
    TOOLS = "tools"
    AGENTS = "agents"
    COMPUTER = "computer"
    DATABASE = "database"
    VECTORDB = "vectordb"
    SECURITY = "security"
    MONITORING = "monitoring"
    AUTOMATION = "automation"
    CAPABILITIES = "capabilities"
    CUSTOM = "custom"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HealthMetric:
    name: str
    value: Any
    unit: str = ""
    timestamp: float = field(default_factory=time.time)

@dataclass
class HealthCheckResult:
    service_id: str
    service_name: str
    service_type: ServiceType
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    metrics: List[HealthMetric] = field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_id": self.service_id,
            "service_name": self.service_name,
            "service_type": self.service_type.value,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "metrics": [{"name": m.name, "value": m.value, "unit": m.unit} for m in self.metrics],
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp,
            "error": self.error,
        }

@dataclass
class ServiceRegistration:
    service_id: str
    service_name: str
    service_type: ServiceType
    checker: Callable
    check_interval: float = 60.0
    timeout: float = 10.0
    critical: bool = False
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_check: Optional[HealthCheckResult] = None
    consecutive_failures: int = 0
    registered_at: float = field(default_factory=time.time)
    enabled: bool = True

@dataclass
class SystemHealthReport:
    timestamp: float = field(default_factory=time.time)
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    services: List[HealthCheckResult] = field(default_factory=list)
    healthy_count: int = 0
    degraded_count: int = 0
    unhealthy_count: int = 0
    unknown_count: int = 0
    total_services: int = 0
    uptime_seconds: float = 0.0
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.value,
            "summary": {
                "total": self.total_services,
                "healthy": self.healthy_count,
                "degraded": self.degraded_count,
                "unhealthy": self.unhealthy_count,
                "unknown": self.unknown_count,
            },
            "uptime_seconds": round(self.uptime_seconds, 2),
            "services": [s.to_dict() for s in self.services],
        }

    def dashboard(self) -> str:
        status_icons = {
            HealthStatus.HEALTHY: "🟢",
            HealthStatus.DEGRADED: "🟡",
            HealthStatus.UNHEALTHY: "🔴",
            HealthStatus.UNKNOWN: "⚪",
            HealthStatus.STARTING: "🔵",
            HealthStatus.STOPPING: "🟠",
        }
        lines = [
            "=" * 65,
            f"  HEALTH DASHBOARD  [{time.strftime('%Y-%m-%d %H:%M:%S')}]",
            f"  Overall: {status_icons.get(self.overall_status, '?')} {self.overall_status.value.upper()}  |  "
            f"Uptime: {self.uptime_seconds:.0f}s",
            f"  Services: {self.healthy_count}✓ {self.degraded_count}⚠ {self.unhealthy_count}✗ {self.unknown_count}?",
            "=" * 65,
        ]
        by_type: Dict[str, List[HealthCheckResult]] = {}
        for s in self.services:
            by_type.setdefault(s.service_type.value, []).append(s)

        for stype, svcs in sorted(by_type.items()):
            lines.append(f"\n  [{stype.upper()}]")
            for svc in svcs:
                icon = status_icons.get(svc.status, "?")
                latency = f" ({svc.latency_ms:.0f}ms)" if svc.latency_ms else ""
                msg = svc.message[:50] if svc.message else ""
                lines.append(f"    {icon} {svc.service_name:<25} {msg}{latency}")
        lines.append("\n" + "=" * 65)
        return "\n".join(lines)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class HealthRegistryError(Exception):
    pass

# ---------------------------------------------------------------------------
# Built-in health checkers
# ---------------------------------------------------------------------------

async def _check_llm_health() -> Dict[str, Any]:
    try:
        from llm_connector import get_connector
        connector = get_connector()
        results = await connector.health_check()
        healthy = any(results.values()) if results else False
        return {
            "healthy": healthy,
            "message": f"Adapters: {list(results.keys())}",
            "adapters": results,
            "requests": connector.token_usage.request_count,
            "total_cost_usd": connector.token_usage.total_cost_usd,
        }
    except Exception as exc:
        return {"healthy": False, "message": str(exc)}

async def _check_memory_health() -> Dict[str, Any]:
    try:
        from memory_connector import get_memory_connector
        connector = get_memory_connector()
        return await connector.health_check()
    except Exception as exc:
        return {"healthy": False, "message": str(exc)}

async def _check_rag_health() -> Dict[str, Any]:
    try:
        from rag_connector import get_rag_connector
        connector = get_rag_connector()
        return await connector.health_check()
    except Exception as exc:
        return {"healthy": False, "message": str(exc)}

async def _check_tools_health() -> Dict[str, Any]:
    try:
        from tool_connector import get_tool_connector
        connector = get_tool_connector()
        return await connector.health_check()
    except Exception as exc:
        return {"healthy": False, "message": str(exc)}

async def _check_agents_health() -> Dict[str, Any]:
    try:
        from agent_connector import get_agent_connector
        connector = get_agent_connector()
        return await connector.health_check()
    except Exception as exc:
        return {"healthy": False, "message": str(exc)}

async def _check_computer_health() -> Dict[str, Any]:
    try:
        from computer_connector import get_computer_connector
        connector = get_computer_connector()
        return await connector.health_check()
    except Exception as exc:
        return {"healthy": False, "message": str(exc)}

async def _check_security_health() -> Dict[str, Any]:
    import os
    issues = []
    if not os.environ.get("SECRET_KEY"):
        issues.append("SECRET_KEY missing")
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    env = os.environ.get("ENVIRONMENT", "development")
    if env in ("prod", "production") and debug:
        issues.append("DEBUG in production")
    return {
        "healthy": len(issues) == 0,
        "message": "; ".join(issues) if issues else "Security OK",
        "issues": issues,
        "environment": env,
        "debug": debug,
    }

async def _check_monitoring_health() -> Dict[str, Any]:
    import os
    sentry = bool(os.environ.get("SENTRY_DSN"))
    datadog = bool(os.environ.get("DATADOG_API_KEY"))
    return {
        "healthy": True,
        "message": "Monitoring active" if (sentry or datadog) else "No external monitoring",
        "sentry": sentry,
        "datadog": datadog,
    }

async def _check_automation_health() -> Dict[str, Any]:
    avail = []
    for pkg in ["pyautogui", "selenium", "playwright"]:
        try:
            __import__(pkg)
            avail.append(pkg)
        except ImportError:
            pass
    return {
        "healthy": True,
        "message": f"Available: {avail}" if avail else "No automation packages",
        "available": avail,
    }

async def _check_capabilities_health() -> Dict[str, Any]:
    caps = {}
    for pkg in ["sentence_transformers", "chromadb", "openai", "anthropic",
                "redis", "httpx", "fastapi", "pydantic"]:
        try:
            __import__(pkg)
            caps[pkg] = True
        except ImportError:
            caps[pkg] = False
    available = sum(1 for v in caps.values() if v)
    return {
        "healthy": True,
        "message": f"{available}/{len(caps)} packages available",
        "capabilities": caps,
    }

def _result_from_dict(
    service_id: str,
    service_name: str,
    service_type: ServiceType,
    data: Dict[str, Any],
    latency_ms: float,
) -> HealthCheckResult:
    is_healthy = data.get("healthy", True)
    status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
    return HealthCheckResult(
        service_id=service_id,
        service_name=service_name,
        service_type=service_type,
        status=status,
        message=data.get("message", ""),
        details={k: v for k, v in data.items() if k not in ("healthy", "message")},
        latency_ms=latency_ms,
    )

# ---------------------------------------------------------------------------
# HealthRegistry – Singleton
# ---------------------------------------------------------------------------

class HealthRegistry:
    _instance: Optional["HealthRegistry"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._services: Dict[str, ServiceRegistration] = {}
        self._history: Dict[str, List[HealthCheckResult]] = {}
        self._history_limit = 100
        self._start_time = time.time()
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitoring = False
        self._alert_hooks: List[Callable] = []
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "HealthRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def get_instance_async(cls) -> "HealthRegistry":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def initialize(self, register_defaults: bool = True) -> "HealthRegistry":
        if self._initialized:
            return self
        if register_defaults:
            self._register_defaults()
        self._initialized = True
        logger.info("HealthRegistry initialized with %d services", len(self._services))
        return self

    def _register_defaults(self) -> None:
        defaults = [
            ("llm", "LLM Service", ServiceType.LLM, _check_llm_health, 60.0, True),
            ("memory", "Memory Service", ServiceType.MEMORY, _check_memory_health, 120.0, False),
            ("rag", "RAG Service", ServiceType.RAG, _check_rag_health, 120.0, False),
            ("tools", "Tool Service", ServiceType.TOOLS, _check_tools_health, 120.0, False),
            ("agents", "Agent Service", ServiceType.AGENTS, _check_agents_health, 120.0, False),
            ("computer", "Computer Service", ServiceType.COMPUTER, _check_computer_health, 300.0, False),
            ("security", "Security", ServiceType.SECURITY, _check_security_health, 300.0, True),
            ("monitoring", "Monitoring", ServiceType.MONITORING, _check_monitoring_health, 300.0, False),
            ("automation", "Automation", ServiceType.AUTOMATION, _check_automation_health, 300.0, False),
            ("capabilities", "Capabilities", ServiceType.CAPABILITIES, _check_capabilities_health, 600.0, False),
        ]
        for sid, name, stype, checker, interval, critical in defaults:
            self.register(
                service_id=sid,
                service_name=name,
                service_type=stype,
                checker=checker,
                check_interval=interval,
                critical=critical,
            )

    def _ensure(self) -> None:
        if not self._initialized:
            self.initialize()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        service_id: str,
        service_name: str,
        service_type: ServiceType,
        checker: Callable,
        check_interval: float = 60.0,
        timeout: float = 10.0,
        critical: bool = False,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        if service_id in self._services and not overwrite:
            logger.warning("Service already registered: %s", service_id)
            return
        reg = ServiceRegistration(
            service_id=service_id,
            service_name=service_name,
            service_type=service_type,
            checker=checker,
            check_interval=check_interval,
            timeout=timeout,
            critical=critical,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._services[service_id] = reg
        self._history[service_id] = []
        logger.debug("Registered service: %s (%s)", service_id, service_type.value)

    def unregister(self, service_id: str) -> bool:
        if service_id in self._services:
            del self._services[service_id]
            self._history.pop(service_id, None)
            return True
        return False

    def enable(self, service_id: str) -> None:
        if service_id in self._services:
            self._services[service_id].enabled = True

    def disable(self, service_id: str) -> None:
        if service_id in self._services:
            self._services[service_id].enabled = False

    # ------------------------------------------------------------------
    # Health checking
    # ------------------------------------------------------------------

    async def check_service(self, service_id: str) -> HealthCheckResult:
        self._ensure()
        reg = self._services.get(service_id)
        if not reg:
            return HealthCheckResult(
                service_id=service_id,
                service_name=service_id,
                service_type=ServiceType.CUSTOM,
                status=HealthStatus.UNKNOWN,
                message="Service not registered",
            )
        t0 = time.perf_counter()
        try:
            raw = await asyncio.wait_for(reg.checker(), timeout=reg.timeout)
            latency = (time.perf_counter() - t0) * 1000
            if isinstance(raw, dict):
                result = _result_from_dict(
                    service_id, reg.service_name, reg.service_type, raw, latency
                )
            elif isinstance(raw, bool):
                result = HealthCheckResult(
                    service_id=service_id,
                    service_name=reg.service_name,
                    service_type=reg.service_type,
                    status=HealthStatus.HEALTHY if raw else HealthStatus.UNHEALTHY,
                    message="OK" if raw else "Failed",
                    latency_ms=latency,
                )
            else:
                result = HealthCheckResult(
                    service_id=service_id,
                    service_name=reg.service_name,
                    service_type=reg.service_type,
                    status=HealthStatus.HEALTHY,
                    message=str(raw),
                    latency_ms=latency,
                )
        except asyncio.TimeoutError:
            latency = (time.perf_counter() - t0) * 1000
            result = HealthCheckResult(
                service_id=service_id,
                service_name=reg.service_name,
                service_type=reg.service_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timeout after {reg.timeout}s",
                latency_ms=latency,
                error="TimeoutError",
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            result = HealthCheckResult(
                service_id=service_id,
                service_name=reg.service_name,
                service_type=reg.service_type,
                status=HealthStatus.UNHEALTHY,
                message=str(exc)[:200],
                latency_ms=latency,
                error=type(exc).__name__,
            )

        if result.status == HealthStatus.UNHEALTHY:
            reg.consecutive_failures += 1
        else:
            reg.consecutive_failures = 0

        reg.last_check = result
        self._store_history(service_id, result)

        if result.status == HealthStatus.UNHEALTHY:
            await self._fire_alert(reg, result)

        return result

    async def check_all(
        self,
        service_type: Optional[ServiceType] = None,
        tags: Optional[List[str]] = None,
    ) -> SystemHealthReport:
        self._ensure()
        sids = list(self._services.keys())
        if service_type:
            sids = [s for s in sids if self._services[s].service_type == service_type]
        if tags:
            sids = [s for s in sids if any(t in self._services[s].tags for t in tags)]
        enabled_sids = [s for s in sids if self._services[s].enabled]

        tasks = [self.check_service(sid) for sid in enabled_sids]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        report = SystemHealthReport(
            services=list(results),
            total_services=len(results),
            uptime_seconds=time.time() - self._start_time,
        )
        for r in results:
            if r.status == HealthStatus.HEALTHY:
                report.healthy_count += 1
            elif r.status == HealthStatus.DEGRADED:
                report.degraded_count += 1
            elif r.status == HealthStatus.UNHEALTHY:
                report.unhealthy_count += 1
            else:
                report.unknown_count += 1

        critical_services = [self._services[s] for s in enabled_sids if self._services[s].critical]
        critical_unhealthy = [
            r for r in results
            if r.status == HealthStatus.UNHEALTHY
            and any(cs.service_id == r.service_id for cs in critical_services)
        ]
        if critical_unhealthy:
            report.overall_status = HealthStatus.UNHEALTHY
        elif report.unhealthy_count > 0 or report.degraded_count > 0:
            report.overall_status = HealthStatus.DEGRADED
        elif report.healthy_count > 0:
            report.overall_status = HealthStatus.HEALTHY
        else:
            report.overall_status = HealthStatus.UNKNOWN

        return report

    def get_last_check(self, service_id: str) -> Optional[HealthCheckResult]:
        reg = self._services.get(service_id)
        return reg.last_check if reg else None

    def get_history(self, service_id: str, limit: int = 20) -> List[HealthCheckResult]:
        return self._history.get(service_id, [])[-limit:]

    def get_all_statuses(self) -> Dict[str, HealthStatus]:
        self._ensure()
        statuses: Dict[str, HealthStatus] = {}
        for sid, reg in self._services.items():
            if reg.last_check:
                statuses[sid] = reg.last_check.status
            else:
                statuses[sid] = HealthStatus.UNKNOWN
        return statuses

    def _store_history(self, service_id: str, result: HealthCheckResult) -> None:
        history = self._history.setdefault(service_id, [])
        history.append(result)
        if len(history) > self._history_limit:
            self._history[service_id] = history[-self._history_limit:]

    # ------------------------------------------------------------------
    # Background monitoring
    # ------------------------------------------------------------------

    async def start_monitoring(self) -> None:
        self._ensure()
        if self._monitoring:
            return
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started")

    async def stop_monitoring(self) -> None:
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")

    async def _monitor_loop(self) -> None:
        last_checks: Dict[str, float] = {}
        while self._monitoring:
            now = time.time()
            for sid, reg in list(self._services.items()):
                if not reg.enabled:
                    continue
                last = last_checks.get(sid, 0)
                if now - last >= reg.check_interval:
                    try:
                        await self.check_service(sid)
                        last_checks[sid] = now
                    except Exception as exc:
                        logger.warning("Monitor check failed for %s: %s", sid, exc)
            await asyncio.sleep(5.0)

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def add_alert_hook(self, hook: Callable) -> None:
        self._alert_hooks.append(hook)

    async def _fire_alert(self, reg: ServiceRegistration, result: HealthCheckResult) -> None:
        for hook in self._alert_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(reg, result)
                else:
                    hook(reg, result)
            except Exception as exc:
                logger.warning("Alert hook error: %s", exc)

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    async def get_dashboard(self) -> str:
        report = await self.check_all()
        return report.dashboard()

    async def get_report(self) -> SystemHealthReport:
        return await self.check_all()

    def get_report_json(self) -> str:
        statuses = self.get_all_statuses()
        return json.dumps({
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self._start_time,
            "services": {k: v.value for k, v in statuses.items()},
        }, indent=2)

    # ------------------------------------------------------------------
    # Service list
    # ------------------------------------------------------------------

    def list_services(self) -> List[Dict[str, Any]]:
        self._ensure()
        return [
            {
                "id": sid,
                "name": reg.service_name,
                "type": reg.service_type.value,
                "enabled": reg.enabled,
                "critical": reg.critical,
                "check_interval": reg.check_interval,
                "consecutive_failures": reg.consecutive_failures,
                "last_status": reg.last_check.status.value if reg.last_check else "unknown",
                "tags": reg.tags,
            }
            for sid, reg in self._services.items()
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "HealthRegistry":
        self._ensure()
        await self.start_monitoring()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop_monitoring()

    def __repr__(self) -> str:
        return f"HealthRegistry(services={list(self._services.keys())}, monitoring={self._monitoring})"


def get_health_registry() -> HealthRegistry:
    registry = HealthRegistry.get_instance()
    if not registry._initialized:
        registry.initialize()
    return registry


async def get_health_report() -> SystemHealthReport:
    return await get_health_registry().get_all_statuses_report() if False else await get_health_registry().check_all()


async def get_dashboard() -> str:
    return await get_health_registry().get_dashboard()


async def is_healthy(service_id: Optional[str] = None) -> bool:
    registry = get_health_registry()
    if service_id:
        result = await registry.check_service(service_id)
        return result.status == HealthStatus.HEALTHY
    report = await registry.check_all()
    return report.overall_status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
