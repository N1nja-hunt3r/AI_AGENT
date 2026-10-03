"""Async session lifecycle management with transaction and rollback support."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .connection import DatabaseConnection, get_db_connection

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages async session lifecycle, transactions, and health checks."""

    def __init__(self, connection: Optional[DatabaseConnection] = None) -> None:
        self._connection = connection or get_db_connection()

    def _factory(self) -> async_sessionmaker[AsyncSession]:
        return self._connection.get_session_factory()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session that auto-commits on success and rolls back on error."""
        factory = self._factory()
        session = factory()
        try:
            yield session
            await session.commit()
        except SQLAlchemyError:
            await session.rollback()
            logger.exception("Session error, transaction rolled back")
            raise
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    @asynccontextmanager
    async def transaction(self, session: AsyncSession) -> AsyncIterator[AsyncSession]:
        """Run an explicit nested transaction (savepoint) on an existing session."""
        if session.in_transaction():
            async with session.begin_nested():
                try:
                    yield session
                except Exception:
                    logger.exception("Nested transaction failed, rolling back savepoint")
                    raise
        else:
            async with session.begin():
                try:
                    yield session
                except Exception:
                    logger.exception("Transaction failed, rolling back")
                    raise

    async def commit(self, session: AsyncSession) -> None:
        """Commit the current session, rolling back on failure."""
        try:
            await session.commit()
        except SQLAlchemyError:
            await session.rollback()
            logger.exception("Commit failed, rolled back")
            raise

    async def rollback(self, session: AsyncSession) -> None:
        """Roll back the current session."""
        await session.rollback()

    async def close(self, session: AsyncSession) -> None:
        """Close a session, releasing its connection back to the pool."""
        await session.close()

    async def health_check(self) -> bool:
        """Verify a session can be opened and a trivial query executed."""
        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Session health check failed: %s", exc)
            return False


_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Module-level accessor for the SessionManager singleton."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
