"""
shutdown.py

Coordinates graceful application shutdown: persisting state, closing
database/vector-database connections, shutting down agents and
running final cleanup hooks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.integration.agent_loader import AgentLoader, get_agent_loader
from app.integration.dependency_container import DependencyContainer, get_container

logger = logging.getLogger("shutdown")


@dataclass
class ShutdownReport:
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    database_closed: bool = False
    vector_db_closed: bool = False
    state_saved: bool = False
    agents_shutdown: int = 0
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def ok(self) -> bool:
        return not self.errors


async def _save_state(container: DependencyContainer) -> bool:
    try:
        snapshot: Dict[str, Any] = {
            "models_warmed": container.try_resolve("models:warmed"),
            "saved_at": time.time(),
        }
        container.register_singleton("shutdown:last_snapshot", snapshot, tags={"kind": "state"})
        logger.info("Application state snapshot saved.")
        return True
    except Exception:
        logger.exception("Failed to save application state.")
        return False


async def _close_database(container: DependencyContainer) -> bool:
    try:
        connection = container.try_resolve("database:connection")
        if connection is not None:
            connection["connected"] = False
        logger.info("Database connection closed.")
        return True
    except Exception:
        logger.exception("Failed to close database connection.")
        return False


async def _close_vector_db(container: DependencyContainer) -> bool:
    try:
        connection = container.try_resolve("vector_db:connection")
        if connection is not None:
            connection["connected"] = False
        logger.info("Vector database connection closed.")
        return True
    except Exception:
        logger.exception("Failed to close vector database connection.")
        return False


async def _shutdown_agents(agent_loader: AgentLoader) -> int:
    count = 0
    for name in agent_loader.list_agents():
        try:
            agent = agent_loader.get_agent(name)
            if hasattr(agent, "shutdown"):
                await agent.shutdown()
            count += 1
        except Exception:
            logger.exception("Failed to shut down agent '%s'.", name)
    return count


async def run_shutdown(container: Optional[DependencyContainer] = None) -> ShutdownReport:
    """Execute the full shutdown sequence and return a structured report."""
    container = container or get_container()
    report = ShutdownReport()

    report.state_saved = await _save_state(container)
    if not report.state_saved:
        report.errors["state"] = "save failed"

    agent_loader: AgentLoader = await get_agent_loader()
    try:
        report.agents_shutdown = await _shutdown_agents(agent_loader)
    except Exception as exc:
        logger.exception("Agent shutdown sequence failed.")
        report.errors["agents"] = str(exc)

    report.database_closed = await _close_database(container)
    if not report.database_closed:
        report.errors["database"] = "close failed"

    report.vector_db_closed = await _close_vector_db(container)
    if not report.vector_db_closed:
        report.errors["vector_db"] = "close failed"

    report.finished_at = time.time()
    logger.info(
        "Shutdown completed in %.2fs (ok=%s, agents_shutdown=%d).",
        report.duration_seconds,
        report.ok,
        report.agents_shutdown,
    )
    return report
