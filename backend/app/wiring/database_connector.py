"""
database_connector.py - Production-grade database connector.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

try:
    from sqlalchemy.ext.asyncio import (
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.pool import NullPool
    from sqlalchemy import text as sa_text
    _SQLALCHEMY_AVAILABLE = True
except ImportError:
    _SQLALCHEMY_AVAILABLE = False
    AsyncEngine = None
    AsyncSession = None
    async_sessionmaker = None
    create_async_engine = None

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    aioredis = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DatabaseConfig:
    database_url: str = "sqlite+aiosqlite:///./app.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: float = 30.0
    pool_recycle: int = 3600
    echo: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_url: Optional[str] = None
    cache_ttl: int = 300
    backup_dir: str = "./backups"
    extra: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DatabaseConnectorError(Exception):
    pass

class ConnectionError(DatabaseConnectorError):
    pass

class SessionError(DatabaseConnectorError):
    pass

# ---------------------------------------------------------------------------
# Cache subsystem (cache.py compatibility)
# ---------------------------------------------------------------------------

class _CacheManager:
    def __init__(self, url: Optional[str] = None, ttl: int = 300) -> None:
        self._url = url
        self._ttl = ttl
        self._client: Any = None
        self._local: Dict[str, Any] = {}
        self._use_redis = bool(url and _REDIS_AVAILABLE)

    async def initialize(self) -> None:
        if self._use_redis:
            try:
                self._client = aioredis.from_url(self._url, decode_responses=True)
                await self._client.ping()
                logger.info("Cache connected: redis")
            except Exception as exc:
                logger.warning("Redis cache unavailable: %s; using local cache", exc)
                self._use_redis = False

    async def get(self, key: str) -> Optional[Any]:
        if self._use_redis and self._client:
            import json
            val = await self._client.get(key)
            return json.loads(val) if val else None
        return self._local.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if self._use_redis and self._client:
            import json
            await self._client.setex(key, ttl or self._ttl, json.dumps(value))
        else:
            self._local[key] = value

    async def delete(self, key: str) -> None:
        if self._use_redis and self._client:
            await self._client.delete(key)
        self._local.pop(key, None)

    async def flush(self) -> None:
        if self._use_redis and self._client:
            await self._client.flushdb()
        self._local.clear()

    async def health_check(self) -> bool:
        if self._use_redis and self._client:
            try:
                return await self._client.ping()
            except Exception:
                return False
        return True

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

# ---------------------------------------------------------------------------
# Session manager (session.py compatibility)
# ---------------------------------------------------------------------------

class _SessionManager:
    def __init__(self, engine: Any, config: DatabaseConfig) -> None:
        self._engine = engine
        self._config = config
        self._factory: Any = None

    def initialize(self) -> None:
        if not _SQLALCHEMY_AVAILABLE or not self._engine:
            return
        self._factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[Any, None]:
        if not self._factory:
            raise SessionError("Session factory not initialized")
        session = self._factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def execute_raw(self, query: str, params: Optional[Dict] = None) -> Any:
        async with self.get_session() as session:
            result = await session.execute(sa_text(query), params or {})
            return result.fetchall()

# ---------------------------------------------------------------------------
# CRUD (crud.py compatibility)
# ---------------------------------------------------------------------------

class _CRUDManager:
    def __init__(self, session_manager: _SessionManager) -> None:
        self._sm = session_manager

    async def create(self, model: Any, data: Dict[str, Any]) -> Any:
        async with self._sm.get_session() as session:
            obj = model(**data)
            session.add(obj)
            await session.flush()
            await session.refresh(obj)
            return obj

    async def get(self, model: Any, obj_id: Any) -> Optional[Any]:
        async with self._sm.get_session() as session:
            return await session.get(model, obj_id)

    async def update(self, obj: Any, data: Dict[str, Any]) -> Any:
        async with self._sm.get_session() as session:
            for k, v in data.items():
                setattr(obj, k, v)
            session.add(obj)
            return obj

    async def delete(self, obj: Any) -> bool:
        async with self._sm.get_session() as session:
            await session.delete(obj)
            return True

    async def list(
        self,
        model: Any,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Any]:
        from sqlalchemy import select
        async with self._sm.get_session() as session:
            stmt = select(model).limit(limit).offset(offset)
            if filters:
                for k, v in filters.items():
                    stmt = stmt.where(getattr(model, k) == v)
            result = await session.execute(stmt)
            return list(result.scalars().all())

# ---------------------------------------------------------------------------
# Backup manager (backup.py compatibility)
# ---------------------------------------------------------------------------

class _BackupManager:
    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._backup_dir = config.backup_dir

    def _ensure_dir(self) -> None:
        import os
        os.makedirs(self._backup_dir, exist_ok=True)

    async def backup(self, tag: Optional[str] = None) -> str:
        self._ensure_dir()
        import os
        import shutil
        ts = int(time.time())
        label = tag or "auto"
        db_url = self._config.database_url
        backup_path = os.path.join(self._backup_dir, f"backup_{label}_{ts}.db")

        if "sqlite" in db_url:
            db_file = db_url.split("///")[-1].lstrip("./")
            if os.path.exists(db_file):
                await asyncio.to_thread(shutil.copy2, db_file, backup_path)
                logger.info("Backup created: %s", backup_path)
                return backup_path

        backup_path = os.path.join(self._backup_dir, f"backup_{label}_{ts}.sql")
        f = await asyncio.to_thread(open, backup_path, "w")
        await asyncio.to_thread(f.write, f"-- backup {ts}\n")
        await asyncio.to_thread(f.close)
        logger.info("Backup placeholder created: %s", backup_path)
        return backup_path

    async def restore(self, backup_path: str) -> bool:
        import os
        import shutil
        if not os.path.exists(backup_path):
            logger.error("Backup not found: %s", backup_path)
            return False
        db_url = self._config.database_url
        if "sqlite" in db_url:
            db_file = db_url.split("///")[-1].lstrip("./")
            await asyncio.to_thread(shutil.copy2, backup_path, db_file)
            logger.info("Restored from: %s", backup_path)
            return True
        logger.warning("Restore only implemented for SQLite")
        return False

    async def list_backups(self) -> List[str]:
        import os
        self._ensure_dir()
        files = await asyncio.to_thread(os.listdir, self._backup_dir)
        return sorted(f for f in files if f.startswith("backup_"))

# ---------------------------------------------------------------------------
# Connection manager (connection.py compatibility)
# ---------------------------------------------------------------------------

class _ConnectionManager:
    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._engine: Any = None
        self._connected = False

    async def connect(self) -> Any:
        if not _SQLALCHEMY_AVAILABLE:
            logger.warning("SQLAlchemy not available; running in mock mode")
            self._connected = True
            return None

        for attempt in range(self._config.max_retries):
            try:
                url = self._config.database_url
                kwargs: Dict[str, Any] = {"echo": self._config.echo}
                if "sqlite" not in url:
                    kwargs.update({
                        "pool_size": self._config.pool_size,
                        "max_overflow": self._config.max_overflow,
                        "pool_timeout": self._config.pool_timeout,
                        "pool_recycle": self._config.pool_recycle,
                        "pool_pre_ping": True,
                    })
                else:
                    kwargs["poolclass"] = NullPool if _SQLALCHEMY_AVAILABLE else None
                self._engine = create_async_engine(url, **{k: v for k, v in kwargs.items() if v is not None})
                async with self._engine.connect() as conn:
                    await conn.execute(sa_text("SELECT 1"))
                self._connected = True
                logger.info("Database connected: %s", url.split("@")[-1])
                return self._engine
            except Exception as exc:
                wait = self._config.retry_delay * (2 ** attempt)
                logger.warning("DB connect attempt %d failed: %s; retry in %.1fs", attempt + 1, exc, wait)
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(wait)
        raise ConnectionError(f"Failed to connect after {self._config.max_retries} attempts")

    async def disconnect(self) -> None:
        if self._engine:
            await self._engine.dispose()
            self._connected = False
            logger.info("Database disconnected")

    async def health_check(self) -> bool:
        if not self._engine:
            return False
        try:
            async with self._engine.connect() as conn:
                await conn.execute(sa_text("SELECT 1"))
            return True
        except Exception:
            return False

    @property
    def engine(self) -> Any:
        return self._engine

    @property
    def is_connected(self) -> bool:
        return self._connected

# ---------------------------------------------------------------------------
# DatabaseConnector – Singleton
# ---------------------------------------------------------------------------

class DatabaseConnector:
    _instance: Optional["DatabaseConnector"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._config: DatabaseConfig = DatabaseConfig()
        self._connection_manager: Optional[_ConnectionManager] = None
        self._session_manager: Optional[_SessionManager] = None
        self._crud_manager: Optional[_CRUDManager] = None
        self._cache_manager: Optional[_CacheManager] = None
        self._backup_manager: Optional[_BackupManager] = None
        self._initialized = False
        self._hooks: List[Callable] = []

    @classmethod
    def get_instance(cls) -> "DatabaseConnector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    async def get_instance_async(cls) -> "DatabaseConnector":
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def configure(self, config: DatabaseConfig) -> "DatabaseConnector":
        self._config = config
        return self

    def configure_from_env(self) -> "DatabaseConnector":
        self._config = DatabaseConfig(
            database_url=os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./app.db"),
            pool_size=int(os.environ.get("DB_POOL_SIZE", "10")),
            max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", "20")),
            echo=os.environ.get("DB_ECHO", "false").lower() == "true",
            cache_url=os.environ.get("CACHE_URL"),
            backup_dir=os.environ.get("BACKUP_DIR", "./backups"),
        )
        return self

    async def connect(self) -> "DatabaseConnector":
        if self._initialized:
            return self
        if not self._config:
            self.configure_from_env()

        self._connection_manager = _ConnectionManager(self._config)
        engine = await self._connection_manager.connect()

        self._session_manager = _SessionManager(engine, self._config)
        self._session_manager.initialize()

        self._crud_manager = _CRUDManager(self._session_manager)
        self._cache_manager = _CacheManager(self._config.cache_url, self._config.cache_ttl)
        await self._cache_manager.initialize()

        self._backup_manager = _BackupManager(self._config)
        self._initialized = True
        logger.info("DatabaseConnector ready")
        return self

    async def disconnect(self) -> None:
        if self._connection_manager:
            await self._connection_manager.disconnect()
        if self._cache_manager:
            await self._cache_manager.close()
        self._initialized = False

    async def health_check(self) -> Dict[str, Any]:
        db_ok = await self._connection_manager.health_check() if self._connection_manager else False
        cache_ok = await self._cache_manager.health_check() if self._cache_manager else False
        return {
            "healthy": db_ok,
            "database": db_ok,
            "cache": cache_ok,
            "initialized": self._initialized,
        }

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[Any, None]:
        if not self._session_manager:
            raise SessionError("Not connected")
        async with self._session_manager.get_session() as session:
            yield session

    def get_crud(self) -> _CRUDManager:
        if not self._crud_manager:
            raise DatabaseConnectorError("Not connected")
        return self._crud_manager

    def get_cache(self) -> _CacheManager:
        if not self._cache_manager:
            raise DatabaseConnectorError("Not connected")
        return self._cache_manager

    async def backup(self, tag: Optional[str] = None) -> str:
        if not self._backup_manager:
            raise DatabaseConnectorError("Not connected")
        return await self._backup_manager.backup(tag)

    async def restore(self, backup_path: str) -> bool:
        if not self._backup_manager:
            raise DatabaseConnectorError("Not connected")
        return await self._backup_manager.restore(backup_path)

    async def execute_raw(self, query: str, params: Optional[Dict] = None) -> Any:
        if not self._session_manager:
            raise DatabaseConnectorError("Not connected")
        return await self._session_manager.execute_raw(query, params)

    async def __aenter__(self) -> "DatabaseConnector":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()

    def __repr__(self) -> str:
        return f"DatabaseConnector(initialized={self._initialized}, url={self._config.database_url.split('@')[-1]})"


def get_database_connector() -> DatabaseConnector:
    connector = DatabaseConnector.get_instance()
    return connector


async def get_db_session() -> AsyncGenerator[Any, None]:
    connector = get_database_connector()
    if not connector._initialized:
        await connector.connect()
    async with connector.get_session() as session:
        yield session
