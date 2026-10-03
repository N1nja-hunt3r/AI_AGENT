"""
startup.py

Coordinates application startup: database and vector-database
connections, model warmup, capability/agent loading and a final
aggregate health check pass.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.integration.agent_loader import AgentLoader, get_agent_loader
from app.integration.capability_loader import CapabilityLoader, get_capability_loader
from app.integration.dependency_container import DependencyContainer, get_container
from app.integration.registry import HealthStatus

logger = logging.getLogger("startup")


@dataclass
class StartupReport:
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    database_connected: bool = False
    vector_db_connected: bool = False
    models_warmed: bool = False
    capabilities_loaded: int = 0
    agents_loaded: int = 0
    health: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def ok(self) -> bool:
        return not self.errors


class DatabaseConnectionError(RuntimeError):
    pass


async def _connect_database(container: DependencyContainer, dsn: Optional[str]) -> bool:
    try:
        # Placeholder connection logic; replace with a real async driver
        # (e.g. asyncpg, motor) wired through the dependency container.
        await asyncio.sleep(0)
        connection = {"dsn": dsn or "sqlite://memory", "connected": True}
        container.register_singleton("database:connection", connection)
        logger.info("Database connection established.")
        return True
    except Exception:
        logger.exception("Failed to connect to the primary database.")
        return False


async def _connect_vector_db(container: DependencyContainer, dsn: Optional[str]) -> bool:
    try:
        await asyncio.sleep(0)
        connection = {"dsn": dsn or "in-memory-vector-store", "connected": True}
        container.register_singleton("vector_db:connection", connection)
        logger.info("Vector database connection established.")
        return True
    except Exception:
        logger.exception("Failed to connect to the vector database.")
        return False


async def _warmup_models(container: DependencyContainer, model_names: Optional[List[str]] = None) -> bool:
    try:
        models = model_names or []
        warmed: Dict[str, bool] = {}
        for name in models:
            await asyncio.sleep(0)  # Replace with an actual model warmup call.
            warmed[name] = True
        container.register_singleton("models:warmed", warmed)
        logger.info("Warmed up %d model(s).", len(models))
        return True
    except Exception:
        logger.exception("Model warmup failed.")
        return False


async def run_startup(
    container: Optional[DependencyContainer] = None,
    *,
    database_dsn: Optional[str] = None,
    vector_db_dsn: Optional[str] = None,
    model_names: Optional[List[str]] = None,
) -> StartupReport:
    """Execute the full startup sequence and return a structured report."""
    container = container or get_container()
    report = StartupReport()

    report.database_connected = await _connect_database(container, database_dsn)
    if not report.database_connected:
        report.errors["database"] = "connection failed"

    report.vector_db_connected = await _connect_vector_db(container, vector_db_dsn)
    if not report.vector_db_connected:
        report.errors["vector_db"] = "connection failed"

    report.models_warmed = await _warmup_models(container, model_names)
    if not report.models_warmed:
        report.errors["models"] = "warmup failed"

    capability_loader: CapabilityLoader = await get_capability_loader()
    try:
        loaded_capabilities = await capability_loader.load_all()
        report.capabilities_loaded = len(loaded_capabilities)
    except Exception as exc:
        logger.exception("Capability loading failed during startup.")
        report.errors["capabilities"] = str(exc)

    agent_loader: AgentLoader = await get_agent_loader()
    try:
        loaded_agents = await agent_loader.load_all()
        report.agents_loaded = len(loaded_agents)
    except Exception as exc:
        logger.exception("Agent loading failed during startup.")
        report.errors["agents"] = str(exc)

    health_results: Dict[str, Dict[str, HealthStatus]] = {}
    try:
        health_results["capabilities"] = await capability_loader.health_check()
        health_results["agents"] = await agent_loader.health_check()
    except Exception as exc:
        logger.exception("Health check pass failed during startup.")
        report.errors["health_check"] = str(exc)

    report.health = {
        domain: {name: status.value for name, status in statuses.items()}
        for domain, statuses in health_results.items()
    }

    report.finished_at = time.time()
    logger.info(
        "Startup completed in %.2fs (ok=%s, capabilities=%d, agents=%d).",
        report.duration_seconds,
        report.ok,
        report.capabilities_loaded,
        report.agents_loaded,
    )
    return report
