"""
bootstrap.py

Top-level system bootstrap: initializes security, monitoring,
database connections, services, capabilities, agents, the reasoning
kernel and automation subsystem, then runs warmup and an aggregate
health check before the application starts serving traffic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pathlib import Path

from dotenv import load_dotenv

from app.integration.agent_loader import AgentLoader, get_agent_loader
from app.integration.capability_loader import CapabilityLoader, get_capability_loader
from app.integration.dependency_container import DependencyContainer, get_container
from app.integration.kernel_initializer import Controller, KernelInitializer, get_kernel_initializer
from app.integration.monitoring_loader import MonitoringLoader, get_monitoring_loader
from app.integration.registry import HealthStatus
from app.integration.security_loader import SecurityLoader, get_security_loader
from app.integration.service_loader import ServiceLoader, get_service_loader
from app.integration.shutdown import ShutdownReport, run_shutdown
from app.integration.startup import StartupReport, run_startup

logger = logging.getLogger("bootstrap")


@dataclass
class AutomationConfig:
    enabled: bool = True
    scheduled_jobs: List[str] = field(default_factory=list)


class AutomationEngine:
    """Coordinates scheduled and event-driven automation jobs."""

    def __init__(self, config: Optional[AutomationConfig] = None) -> None:
        self._config = config or AutomationConfig()
        self._running = False

    async def start(self) -> None:
        if not self._config.enabled:
            logger.info("Automation engine disabled via configuration.")
            return
        self._running = True
        logger.info("Automation engine started with %d job(s).", len(self._config.scheduled_jobs))

    async def stop(self) -> None:
        self._running = False

    async def health_check(self) -> bool:
        return self._running or not self._config.enabled


@dataclass
class BootstrapReport:
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    security_ready: bool = False
    monitoring_ready: bool = False
    services_loaded: int = 0
    capabilities_loaded: int = 0
    agents_loaded: int = 0
    kernel_ready: bool = False
    automation_ready: bool = False
    startup_report: Optional[StartupReport] = None
    health: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def ok(self) -> bool:
        return not self.errors


class SystemBootstrap:
    """Owns the end-to-end lifecycle of the AI system's subsystems."""

    def __init__(self, container: Optional[DependencyContainer] = None) -> None:
        self.container: DependencyContainer = container or get_container()
        self.security_loader: Optional[SecurityLoader] = None
        self.monitoring_loader: Optional[MonitoringLoader] = None
        self.service_loader: Optional[ServiceLoader] = None
        self.capability_loader: Optional[CapabilityLoader] = None
        self.agent_loader: Optional[AgentLoader] = None
        self.kernel_initializer: Optional[KernelInitializer] = None
        self.automation: AutomationEngine = AutomationEngine()
        self.controller: Optional[Controller] = None
        self._lock = asyncio.Lock()
        self._ready = False

    async def initialize(
        self,
        *,
        database_dsn: Optional[str] = None,
        vector_db_dsn: Optional[str] = None,
        model_names: Optional[List[str]] = None,
    ) -> BootstrapReport:
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info("Loaded environment from %s", env_path)

        import os
        if os.environ.get("NVIDIA_DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.environ["NVIDIA_DEEPSEEK_API_KEY"]
            os.environ["OPENAI_BASE_URL"] = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

        try:
            from app.wiring.provider_initializer import init_provider_registry
            provider_registry = init_provider_registry()
            self.container.register_singleton("provider:registry", provider_registry)
            logger.info("Provider registry initialized with %d providers", len(provider_registry.list_providers()))
        except Exception as exc:
            logger.exception("Provider registry initialization failed: %s", exc)

        async with self._lock:
            report = BootstrapReport()

            self.security_loader = await get_security_loader()
            self.monitoring_loader = await get_monitoring_loader()
            self.service_loader = await get_service_loader()
            self.capability_loader = await get_capability_loader()
            self.agent_loader = await get_agent_loader()
            self.kernel_initializer = get_kernel_initializer()

            try:
                await self.monitoring_loader.initialize()
                report.monitoring_ready = True
            except Exception as exc:
                logger.exception("Monitoring initialization failed.")
                report.errors["monitoring"] = str(exc)

            try:
                await self.security_loader.initialize()
                report.security_ready = True
            except Exception as exc:
                logger.exception("Security initialization failed.")
                report.errors["security"] = str(exc)

            startup_report = await run_startup(
                self.container,
                database_dsn=database_dsn,
                vector_db_dsn=vector_db_dsn,
                model_names=model_names,
            )
            report.startup_report = startup_report
            report.capabilities_loaded = startup_report.capabilities_loaded
            report.agents_loaded = startup_report.agents_loaded
            if not startup_report.ok:
                report.errors.update({f"startup.{k}": v for k, v in startup_report.errors.items()})

            try:
                loaded_services = await self.service_loader.load_all()
                report.services_loaded = len(loaded_services)
            except Exception as exc:
                logger.exception("Service loading failed.")
                report.errors["services"] = str(exc)

            try:
                self.controller = await self.kernel_initializer.initialize()
                report.kernel_ready = True
            except Exception as exc:
                logger.exception("Kernel initialization failed.")
                report.errors["kernel"] = str(exc)

            try:
                await self.automation.start()
                report.automation_ready = True
            except Exception as exc:
                logger.exception("Automation engine failed to start.")
                report.errors["automation"] = str(exc)

            report.health = await self._collect_health()
            report.finished_at = time.time()
            self._ready = report.ok
            logger.info(
                "System bootstrap completed in %.2fs (ok=%s).",
                report.duration_seconds,
                report.ok,
            )
            return report

    async def _collect_health(self) -> Dict[str, Any]:
        health: Dict[str, Any] = {}
        try:
            if self.security_loader:
                health["security"] = {
                    k: v.value for k, v in (await self.security_loader.health_check()).items()
                }
            if self.monitoring_loader:
                health["monitoring"] = {
                    k: v.value for k, v in (await self.monitoring_loader.health_check()).items()
                }
            if self.service_loader:
                health["services"] = {
                    k: v.value for k, v in (await self.service_loader.health_check()).items()
                }
            if self.capability_loader:
                health["capabilities"] = {
                    k: v.value for k, v in (await self.capability_loader.health_check()).items()
                }
            if self.agent_loader:
                health["agents"] = {
                    k: v.value for k, v in (await self.agent_loader.health_check()).items()
                }
            if self.kernel_initializer:
                health["kernel"] = {
                    k: v.value for k, v in (await self.kernel_initializer.health_check()).items()
                }
            health["automation"] = (
                HealthStatus.HEALTHY.value if await self.automation.health_check() else HealthStatus.UNHEALTHY.value
            )
        except Exception:
            logger.exception("Failed to collect aggregate health status.")
        return health

    async def health_check(self) -> Dict[str, Any]:
        return await self._collect_health()

    async def shutdown(self) -> ShutdownReport:
        async with self._lock:
            await self.automation.stop()
            if self.monitoring_loader:
                await self.monitoring_loader.shutdown()
            if self.security_loader:
                await self.security_loader.shutdown()
            report = await run_shutdown(self.container)
            self._ready = False
            return report

    @property
    def is_ready(self) -> bool:
        return self._ready


_bootstrap_lock = asyncio.Lock()
_bootstrap: Optional[SystemBootstrap] = None


async def get_bootstrap() -> SystemBootstrap:
    global _bootstrap
    async with _bootstrap_lock:
        if _bootstrap is None:
            _bootstrap = SystemBootstrap()
        return _bootstrap
