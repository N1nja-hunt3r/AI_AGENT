"""
short_term.py
=============
Short-term memory store for the AI Operating System.

Responsibilities
----------------
- Store per-session conversation history (messages).
- Sliding-window management: cap history by message count or token budget.
- Context trimming: drop oldest messages when the window is exceeded.
- Recent-message retrieval for context_engine.py and memory_capability.py.
- Full session isolation: each session_id has an independent state.
- Health reporting compatible with the capability health-check protocol.

Future
------
- PostgreSQL persistence via asyncpg (stub included).
- Pluggable tokeniser (tiktoken / HuggingFace tokenizers).
- TTL-based session expiry.
- Multi-Agent shared-session support.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """Immutable unit of conversation stored in short-term memory."""

    role: Role
    content: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0                    # populated by tokeniser on insert

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=Role(data["role"]),
            content=data["content"],
            message_id=data.get("message_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count", 0),
        )


@dataclass
class SessionWindow:
    """In-memory sliding window for one isolated session."""

    session_id: str
    messages: List[Message] = field(default_factory=list)
    total_tokens: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_updated: float = field(default_factory=time.monotonic)

    def token_sum(self) -> int:
        return sum(m.token_count for m in self.messages)


# ---------------------------------------------------------------------------
# Tokeniser (pluggable)
# ---------------------------------------------------------------------------


class _BaseTokeniser:
    """Count tokens for window budget enforcement."""

    def count(self, text: str) -> int:  # noqa: D102
        raise NotImplementedError


class _NaiveTokeniser(_BaseTokeniser):
    """
    Whitespace-split estimator.
    Replace with tiktoken or HuggingFace tokenizers for production accuracy.
    """

    def count(self, text: str) -> int:
        return max(1, len(text.split()))


def _build_tokeniser(name: str = "naive") -> _BaseTokeniser:
    if name == "naive":
        return _NaiveTokeniser()

    if name == "tiktoken":
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding("cl100k_base")

            class _TikTokeniser(_BaseTokeniser):
                def count(self, text: str) -> int:
                    return len(enc.encode(text))

            return _TikTokeniser()
        except ImportError:
            logger.warning("tiktoken not installed; falling back to naive tokeniser.")
            return _NaiveTokeniser()

    raise ValueError(f"Unknown tokeniser: {name}")


# ---------------------------------------------------------------------------
# Persistence backend interface (Future: PostgreSQL)
# ---------------------------------------------------------------------------


class _PersistenceBackend:
    """Abstract persistence contract. Swap for asyncpg backend in production."""

    async def save_message(self, session_id: str, message: Message) -> None:
        """Persist a single message."""

    async def load_session(self, session_id: str) -> List[Message]:
        """Load all messages for a session on warm-up."""
        return []

    async def delete_session(self, session_id: str) -> None:
        """Remove all records for a session."""

    async def health(self) -> bool:
        return True


class _PostgreSQLBackend(_PersistenceBackend):
    """
    Stub – replace body with asyncpg calls once a DB is provisioned.

    Expected schema
    ---------------
    CREATE TABLE short_term_messages (
        message_id  UUID PRIMARY KEY,
        session_id  TEXT NOT NULL,
        role        TEXT NOT NULL,
        content     TEXT NOT NULL,
        token_count INTEGER NOT NULL DEFAULT 0,
        metadata    JSONB NOT NULL DEFAULT '{}',
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX ON short_term_messages (session_id, created_at);
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: Any = None          # asyncpg.Pool

    async def _ensure_pool(self) -> None:
        if self._pool is not None:
            return
        try:
            import asyncpg  # type: ignore

            self._pool = await asyncpg.create_pool(self._dsn)
        except ImportError:
            raise RuntimeError("asyncpg not installed. Run: pip install asyncpg")

    async def save_message(self, session_id: str, message: Message) -> None:
        await self._ensure_pool()
        import json

        await self._pool.execute(
            """
            INSERT INTO short_term_messages
                (message_id, session_id, role, content, token_count, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (message_id) DO NOTHING
            """,
            message.message_id,
            session_id,
            message.role.value,
            message.content,
            message.token_count,
            json.dumps(message.metadata),
        )

    async def load_session(self, session_id: str) -> List[Message]:
        await self._ensure_pool()
        rows = await self._pool.fetch(
            "SELECT * FROM short_term_messages WHERE session_id=$1 ORDER BY created_at",
            session_id,
        )
        return [
            Message(
                role=Role(r["role"]),
                content=r["content"],
                message_id=str(r["message_id"]),
                token_count=r["token_count"],
                metadata=r["metadata"] or {},
            )
            for r in rows
        ]

    async def delete_session(self, session_id: str) -> None:
        await self._ensure_pool()
        await self._pool.execute(
            "DELETE FROM short_term_messages WHERE session_id=$1", session_id
        )

    async def health(self) -> bool:
        try:
            await self._ensure_pool()
            await self._pool.fetchval("SELECT 1")
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# ShortTermMemory
# ---------------------------------------------------------------------------


class ShortTermMemory:
    """
    Per-session sliding-window conversation store.

    Compatible with
    ---------------
    - context_engine.py  →  get_recent_context(session_id) → List[dict]
    - memory_capability.py  →  all public async methods

    Parameters
    ----------
    max_messages:   Hard cap on messages per session (0 = unlimited).
    max_tokens:     Token budget per session window (0 = unlimited).
    tokeniser:      'naive' (default) | 'tiktoken'
    persistence:    Optional _PersistenceBackend instance.
    """

    def __init__(
        self,
        max_messages: int = 200,
        max_tokens: int = 8_000,
        tokeniser: str = "naive",
        persistence: Optional[_PersistenceBackend] = None,
    ) -> None:
        self._max_messages = max_messages
        self._max_tokens = max_tokens
        self._tokeniser = _build_tokeniser(tokeniser)
        self._persistence = persistence or _PersistenceBackend()
        self._sessions: Dict[str, SessionWindow] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_session(self, session_id: str) -> SessionWindow:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionWindow(session_id=session_id)
            logger.debug("[STM] New session created: %s", session_id)
        return self._sessions[session_id]

    def _count_tokens(self, text: str) -> int:
        return self._tokeniser.count(text)

    def _enforce_window(self, window: SessionWindow) -> int:
        """
        Drop oldest messages until both constraints are satisfied.
        Returns the number of messages evicted.
        """
        evicted = 0

        while window.messages:
            over_count = (
                self._max_messages > 0
                and len(window.messages) > self._max_messages
            )
            over_tokens = (
                self._max_tokens > 0
                and window.token_sum() > self._max_tokens
            )
            if not (over_count or over_tokens):
                break
            dropped = window.messages.pop(0)
            window.total_tokens -= dropped.token_count
            evicted += 1
            logger.debug(
                "[STM] Evicted message %s from session %s (role=%s)",
                dropped.message_id,
                window.session_id,
                dropped.role.value,
            )

        return evicted

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_message(
        self,
        session_id: str,
        role: str | Role,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Append a message to the session window.

        Parameters
        ----------
        session_id: Unique identifier for the conversation session.
        role:       Speaker role ('user', 'assistant', 'system', 'tool').
        content:    Message text.
        metadata:   Optional arbitrary key-value annotations.

        Returns
        -------
        The stored Message object.
        """
        if isinstance(role, str):
            role = Role(role)

        token_count = self._count_tokens(content)
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
            token_count=token_count,
        )

        async with self._lock:
            window = self._get_or_create_session(session_id)
            window.messages.append(message)
            window.total_tokens += token_count
            window.last_updated = time.monotonic()
            self._enforce_window(window)

        # Fire-and-forget persistence (non-blocking)
        asyncio.ensure_future(
            self._persistence.save_message(session_id, message)
        )

        logger.debug(
            "[STM] add_message session=%s role=%s tokens=%d total_tokens=%d",
            session_id,
            role.value,
            token_count,
            window.total_tokens,
        )
        return message

    async def get_recent_context(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        roles: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the most-recent messages for context_engine.py consumption.

        Parameters
        ----------
        session_id:   Target session.
        max_messages: Override window size for this retrieval only.
        max_tokens:   Override token budget for this retrieval only.
        roles:        If provided, filter to these roles only.

        Returns
        -------
        List of message dicts (newest last), ready for LLM API consumption.
        """
        async with self._lock:
            window = self._sessions.get(session_id)
            if window is None:
                return []

            messages: List[Message] = list(window.messages)

        # Optional role filter
        if roles:
            role_set = {Role(r) for r in roles}
            messages = [m for m in messages if m.role in role_set]

        # Optional local token budget (trim from oldest)
        if max_tokens and max_tokens > 0:
            budget = max_tokens
            trimmed: List[Message] = []
            for msg in reversed(messages):
                if budget - msg.token_count < 0:
                    break
                trimmed.insert(0, msg)
                budget -= msg.token_count
            messages = trimmed

        # Optional local message cap (take tail)
        if max_messages and max_messages > 0:
            messages = messages[-max_messages:]

        return [m.to_dict() for m in messages]

    async def trim_context(
        self,
        session_id: str,
        keep_messages: Optional[int] = None,
        keep_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Explicitly trim a session's window to the supplied limits.

        Parameters
        ----------
        session_id:    Target session.
        keep_messages: Keep only the N most-recent messages.
        keep_tokens:   Keep messages until token budget is reached (from tail).

        Returns
        -------
        Trim summary: messages_before, messages_after, tokens_before, tokens_after.
        """
        async with self._lock:
            window = self._sessions.get(session_id)
            if window is None:
                return {"session_id": session_id, "trimmed": 0, "not_found": True}

            before_count = len(window.messages)
            before_tokens = window.token_sum()

            # Token-based trim (from oldest)
            if keep_tokens and keep_tokens > 0:
                budget = keep_tokens
                kept: List[Message] = []
                for msg in reversed(window.messages):
                    if budget - msg.token_count < 0:
                        break
                    kept.insert(0, msg)
                    budget -= msg.token_count
                window.messages = kept

            # Message-count trim (keep tail)
            if keep_messages and keep_messages > 0:
                window.messages = window.messages[-keep_messages:]

            window.total_tokens = window.token_sum()
            window.last_updated = time.monotonic()
            after_count = len(window.messages)
            after_tokens = window.total_tokens

        summary = {
            "session_id": session_id,
            "messages_before": before_count,
            "messages_after": after_count,
            "tokens_before": before_tokens,
            "tokens_after": after_tokens,
            "trimmed": before_count - after_count,
        }
        logger.info("[STM] trim_context %s", summary)
        return summary

    async def clear(self, session_id: str) -> Dict[str, Any]:
        """
        Remove all messages for a session.

        Returns
        -------
        Summary dict with messages_cleared count.
        """
        async with self._lock:
            window = self._sessions.pop(session_id, None)

        count = len(window.messages) if window else 0
        asyncio.ensure_future(self._persistence.delete_session(session_id))
        logger.info("[STM] clear session=%s messages_cleared=%d", session_id, count)
        return {"session_id": session_id, "messages_cleared": count}

    async def health_check(self) -> Dict[str, Any]:
        """
        Return health status compatible with the capability health-check protocol.
        """
        backend_ok = await self._persistence.health()
        active_sessions = len(self._sessions)
        total_messages = sum(len(w.messages) for w in self._sessions.values())
        total_tokens = sum(w.token_sum() for w in self._sessions.values())

        status = "healthy" if backend_ok else "degraded"
        return {
            "status": status,
            "active_sessions": active_sessions,
            "total_messages_in_memory": total_messages,
            "total_tokens_in_memory": total_tokens,
            "max_messages_per_session": self._max_messages,
            "max_tokens_per_session": self._max_tokens,
            "persistence_backend_ok": backend_ok,
            "tokeniser": type(self._tokeniser).__name__,
        }

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    async def session_stats(self, session_id: str) -> Dict[str, Any]:
        """Return statistics for a single session."""
        async with self._lock:
            window = self._sessions.get(session_id)
            if window is None:
                return {"session_id": session_id, "found": False}
            return {
                "session_id": session_id,
                "found": True,
                "message_count": len(window.messages),
                "total_tokens": window.token_sum(),
                "created_at": window.created_at,
                "last_updated": window.last_updated,
            }

    async def list_sessions(self) -> List[str]:
        """Return all active session IDs."""
        async with self._lock:
            return list(self._sessions.keys())

    async def warm_up(self, session_id: str) -> int:
        """
        Load a session from the persistence backend into memory.
        Returns the number of messages loaded.
        """
        messages = await self._persistence.load_session(session_id)
        if not messages:
            return 0

        async with self._lock:
            window = self._get_or_create_session(session_id)
            window.messages = messages
            window.total_tokens = window.token_sum()
            self._enforce_window(window)

        logger.info(
            "[STM] warm_up session=%s loaded=%d", session_id, len(messages)
        )
        return len(messages)


# ---------------------------------------------------------------------------
# Factory helpers (called by memory_capability.py / registry.py)
# ---------------------------------------------------------------------------


def create_short_term_memory(
    *,
    max_messages: int = 200,
    max_tokens: int = 8_000,
    tokeniser: str = "naive",
    postgres_dsn: Optional[str] = None,
) -> ShortTermMemory:
    """
    Instantiate ShortTermMemory with optional PostgreSQL persistence.

    Parameters
    ----------
    max_messages:  Window size in messages (0 = unlimited).
    max_tokens:    Window size in tokens   (0 = unlimited).
    tokeniser:     'naive' | 'tiktoken'
    postgres_dsn:  If supplied, use PostgreSQL backend; else in-memory only.
    """
    backend: _PersistenceBackend
    if postgres_dsn:
        backend = _PostgreSQLBackend(postgres_dsn)
        logger.info("[STM] Using PostgreSQL persistence backend.")
    else:
        backend = _PersistenceBackend()
        logger.info("[STM] Using in-memory (no persistence) backend.")

    return ShortTermMemory(
        max_messages=max_messages,
        max_tokens=max_tokens,
        tokeniser=tokeniser,
        persistence=backend,
    )
