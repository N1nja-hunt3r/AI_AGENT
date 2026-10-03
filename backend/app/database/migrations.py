"""Programmatic Alembic migration management with versioning and health checks."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class MigrationManager:
    """Wraps Alembic to manage schema versioning, upgrades, and downgrades."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        script_location: str = "migrations",
        config_path: Optional[str] = None,
    ) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL", "sqlite:///app.db")
        self.script_location = script_location
        self.config_path = config_path or os.getenv("ALEMBIC_CONFIG", "alembic.ini")
        self._engine: Optional[Engine] = None

    def _sync_url(self) -> str:
        """Return a sync-driver URL for Alembic, which does not run async."""
        return self.database_url.replace("+aiosqlite", "").replace("+asyncpg", "")

    def get_config(self) -> Config:
        cfg = Config(self.config_path) if Path(self.config_path).exists() else Config()
        cfg.set_main_option("script_location", self.script_location)
        cfg.set_main_option("sqlalchemy.url", self._sync_url())
        return cfg

    def get_engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self._sync_url(), pool_pre_ping=True)
        return self._engine

    def init(self) -> None:
        """Initialize a new migrations directory."""
        cfg = self.get_config()
        if not Path(self.script_location).exists():
            command.init(cfg, self.script_location)
            logger.info("Initialized Alembic migrations at %s", self.script_location)

    def revision(self, message: str, autogenerate: bool = True) -> None:
        """Create a new migration revision."""
        cfg = self.get_config()
        command.revision(cfg, message=message, autogenerate=autogenerate)
        logger.info("Created migration revision: %s", message)

    def upgrade(self, revision: str = "head") -> None:
        """Upgrade the database to a given revision (default: latest)."""
        cfg = self.get_config()
        command.upgrade(cfg, revision)
        logger.info("Upgraded database to revision: %s", revision)

    def downgrade(self, revision: str) -> None:
        """Downgrade the database to a given revision."""
        cfg = self.get_config()
        command.downgrade(cfg, revision)
        logger.info("Downgraded database to revision: %s", revision)

    def current_revision(self) -> Optional[str]:
        """Return the current database revision, if any."""
        engine = self.get_engine()
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            return context.get_current_revision()

    def head_revision(self) -> Optional[str]:
        """Return the latest available revision from migration scripts."""
        cfg = self.get_config()
        script = ScriptDirectory.from_config(cfg)
        return script.get_current_head()

    def is_up_to_date(self) -> bool:
        """Check whether the database is at the latest migration revision."""
        return self.current_revision() == self.head_revision()

    def history(self) -> List[str]:
        """List all known migration revisions, newest first."""
        cfg = self.get_config()
        script = ScriptDirectory.from_config(cfg)
        return [rev.revision for rev in script.walk_revisions()]

    def health_check(self) -> bool:
        """Verify the database is reachable and schema is at the latest revision."""
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return self.is_up_to_date()
        except Exception as exc:  # noqa: BLE001
            logger.error("Migration health check failed: %s", exc)
            return False

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
