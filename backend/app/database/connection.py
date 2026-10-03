"""Database connection management with singleton engine, pooling, retries, and health checks."""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator, Optional
from urllib.parse import quote_plus

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.pool import NullPool, QueuePool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration sourced from environment variables."""

    driver: str = os.getenv("DB_DRIVER", "sqlite")
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "app.db")
    user: str = os.getenv("DB_USER", "")
    password: str = os.getenv("DB_PASSWORD", "")
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "10"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    pool_recycle: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))
    echo: bool = os.getenv("DB_ECHO", "false").lower() == "true"
    connect_retries: int = int(os.getenv("DB_CONNECT_RETRIES", "5"))
    retry_backoff_seconds: float = float(os.getenv("DB_RETRY_BACKOFF", "1.0"))
    health_check_timeout: float = float(os.getenv("DB_HEALTH_CHECK_TIMEOUT", "5.0"))

    @property
    def url(self) -> str:
        explicit = os.getenv("DATABASE_URL")
        if explicit:
            return explicit
        if self.driver == "sqlite":
            return f"sqlite+aiosqlite:///{self.name}"
        if self.driver == "postgresql":
            user = quote_plus(self.user)
            password = quote_plus(self.password)
            return f"postgresql+asyncpg://{user}:{password}@{self.host}:{self.port}/{self.name}"
        raise ValueError(f"Unsupported DB driver: {self.driver}")


class DatabaseConnection:
    """Singleton manager for the async SQLAlchemy engine and session factory."""

    _instance: Optional["DatabaseConnection"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "DatabaseConnection":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        if self._initialized:
            return
        self._config = config or DatabaseConfig()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._initialized = True

    @classmethod
    def get_instance(cls, config: Optional[DatabaseConfig] = None) -> "DatabaseConnection":
        return cls(config)

    def get_engine(self) -> AsyncEngine:
        """Return the singleton AsyncEngine, creating it on first access."""
        if self._engine is not None:
            return self._engine

        is_sqlite = self._config.driver == "sqlite"
        pool_kwargs = (
            {"poolclass": NullPool}
            if is_sqlite
            else {
                "poolclass": QueuePool,
                "pool_size": self._config.pool_size,
                "max_overflow": self._config.max_overflow,
                "pool_timeout": self._config.pool_timeout,
                "pool_recycle": self._config.pool_recycle,
                "pool_pre_ping": True,
            }
        )

        self._engine = create_async_engine(
            self._config.url,
            echo=self._config.echo,
            future=True,
            **pool_kwargs,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        logger.info("Database engine initialized for driver=%s", self._config.driver)
        return self._engine

    def get_session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            self.get_engine()
        assert self._session_factory is not None
        return self._session_factory

    async def get_session(self) -> AsyncIterator[AsyncSession]:
        """Yield a new AsyncSession from the singleton session factory."""
        factory = self.get_session_factory()
        async with factory() as session:
            yield session

    async def connect_with_retry(self) -> None:
        """Establish connectivity, retrying with exponential backoff."""
        engine = self.get_engine()
        last_error: Optional[Exception] = None
        for attempt in range(1, self._config.connect_retries + 1):
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                logger.info("Database connection established on attempt %d", attempt)
                return
            except (OperationalError, SQLAlchemyError, OSError) as exc:
                last_error = exc
                wait = self._config.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Database connection attempt %d/%d failed: %s. Retrying in %.1fs",
                    attempt,
                    self._config.connect_retries,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
        raise ConnectionError(
            f"Failed to connect to database after {self._config.connect_retries} attempts"
        ) from last_error

    async def health_check(self) -> bool:
        """Run a lightweight query to verify database connectivity."""
        try:
            engine = self.get_engine()
            async with engine.connect() as conn:
                await asyncio.wait_for(
                    conn.execute(text("SELECT 1")),
                    timeout=self._config.health_check_timeout,
                )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Health check failed: %s", exc)
            return False

    async def close(self) -> None:
        """Dispose of the engine and release all pooled connections."""
        if self._engine is not None:
            await self._engine.dispose()
            logger.info("Database engine disposed")
            self._engine = None
            self._session_factory = None


_db_connection: Optional[DatabaseConnection] = None


def get_db_connection() -> DatabaseConnection:
    """Module-level accessor for the singleton DatabaseConnection."""
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    return _db_connection
