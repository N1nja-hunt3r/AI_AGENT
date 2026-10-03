from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional


class EventCategory(str, Enum):
    SECURITY = "security"
    APPROVAL = "approval"
    COMPUTER_ACTION = "computer_action"
    AUTH = "auth"
    SYSTEM = "system"


class EventSeverity(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    event_id: str
    category: EventCategory
    severity: EventSeverity
    action: str
    actor: str
    timestamp: float = field(default_factory=time.time)
    target: Optional[str] = None
    outcome: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

    def to_json(self) -> str:
        payload = asdict(self)
        payload["category"] = self.category.value
        payload["severity"] = self.severity.value
        payload["timestamp_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp))
        return json.dumps(payload, default=str, sort_keys=True)


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AuditLogger:
    """Structured JSON audit logger for security, approval, and computer-action events."""

    def __init__(
        self,
        *,
        log_path: Optional[str | Path] = None,
        max_in_memory: int = 10_000,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        logger_name: str = "audit_logger",
    ) -> None:
        self.max_in_memory = max_in_memory
        self._buffer: list[AuditEvent] = []
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._event_count = 0
        self._category_counts: dict[str, int] = {c.value: 0 for c in EventCategory}

        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._write_enabled = False

        if not any(isinstance(h, RotatingFileHandler) for h in self._logger.handlers):
            if log_path is not None:
                path = Path(log_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                handler = RotatingFileHandler(str(path), maxBytes=max_bytes, backupCount=backup_count)
                handler.setFormatter(logging.Formatter("%(message)s"))
                self._logger.addHandler(handler)
                self._write_enabled = True

    def _severity_to_level(self, severity: EventSeverity) -> int:
        return {
            EventSeverity.DEBUG: logging.DEBUG,
            EventSeverity.INFO: logging.INFO,
            EventSeverity.WARNING: logging.WARNING,
            EventSeverity.ERROR: logging.ERROR,
            EventSeverity.CRITICAL: logging.CRITICAL,
        }[severity]

    def _record(self, event: AuditEvent) -> AuditEvent:
        self._buffer.append(event)
        if len(self._buffer) > self.max_in_memory:
            self._buffer = self._buffer[-self.max_in_memory :]
        self._event_count += 1
        self._category_counts[event.category.value] = self._category_counts.get(event.category.value, 0) + 1
        if self._write_enabled:
            self._logger.log(self._severity_to_level(event.severity), event.to_json())
        return event

    def log_event(
        self,
        *,
        category: EventCategory,
        action: str,
        actor: str,
        severity: EventSeverity = EventSeverity.INFO,
        target: Optional[str] = None,
        outcome: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> AuditEvent:
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            category=category,
            severity=severity,
            action=action,
            actor=actor,
            target=target,
            outcome=outcome,
            details=details or {},
            correlation_id=correlation_id,
        )
        return self._record(event)

    def log_security_event(
        self,
        action: str,
        actor: str,
        *,
        severity: EventSeverity = EventSeverity.WARNING,
        target: Optional[str] = None,
        outcome: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> AuditEvent:
        return self.log_event(
            category=EventCategory.SECURITY, action=action, actor=actor, severity=severity,
            target=target, outcome=outcome, details=details, correlation_id=correlation_id,
        )

    def log_approval_event(
        self,
        action: str,
        actor: str,
        *,
        outcome: str,
        target: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> AuditEvent:
        return self.log_event(
            category=EventCategory.APPROVAL, action=action, actor=actor, severity=EventSeverity.INFO,
            target=target, outcome=outcome, details=details, correlation_id=correlation_id,
        )

    def log_computer_action(
        self,
        action: str,
        actor: str,
        *,
        target: Optional[str] = None,
        outcome: Optional[str] = None,
        severity: EventSeverity = EventSeverity.INFO,
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> AuditEvent:
        return self.log_event(
            category=EventCategory.COMPUTER_ACTION, action=action, actor=actor, severity=severity,
            target=target, outcome=outcome, details=details, correlation_id=correlation_id,
        )

    def log_auth_event(
        self,
        action: str,
        actor: str,
        *,
        outcome: str,
        severity: EventSeverity = EventSeverity.INFO,
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> AuditEvent:
        return self.log_event(
            category=EventCategory.AUTH, action=action, actor=actor, severity=severity,
            outcome=outcome, details=details, correlation_id=correlation_id,
        )

    def history(
        self,
        limit: int = 100,
        *,
        category: Optional[EventCategory] = None,
        actor: Optional[str] = None,
        severity: Optional[EventSeverity] = None,
        correlation_id: Optional[str] = None,
        since: Optional[float] = None,
    ) -> list[AuditEvent]:
        items = self._buffer
        if category is not None:
            items = [e for e in items if e.category == category]
        if actor is not None:
            items = [e for e in items if e.actor == actor]
        if severity is not None:
            items = [e for e in items if e.severity == severity]
        if correlation_id is not None:
            items = [e for e in items if e.correlation_id == correlation_id]
        if since is not None:
            items = [e for e in items if e.timestamp >= since]
        return items[-limit:]

    def export_json(self, limit: int = 1000) -> str:
        return json.dumps([json.loads(e.to_json()) for e in self.history(limit=limit)], sort_keys=True)

    async def log_event_async(self, **kwargs: Any) -> AuditEvent:
        async with self._lock:
            return self.log_event(**kwargs)

    async def log_security_event_async(self, action: str, actor: str, **kwargs: Any) -> AuditEvent:
        async with self._lock:
            return self.log_security_event(action, actor, **kwargs)

    async def log_approval_event_async(self, action: str, actor: str, *, outcome: str, **kwargs: Any) -> AuditEvent:
        async with self._lock:
            return self.log_approval_event(action, actor, outcome=outcome, **kwargs)

    async def log_computer_action_async(self, action: str, actor: str, **kwargs: Any) -> AuditEvent:
        async with self._lock:
            return self.log_computer_action(action, actor, **kwargs)

    async def history_async(self, limit: int = 100, **kwargs: Any) -> list[AuditEvent]:
        async with self._lock:
            return self.history(limit=limit, **kwargs)

    def health_check(self) -> HealthStatus:
        try:
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "event_count": self._event_count,
                "buffer_size": len(self._buffer),
                "max_in_memory": self.max_in_memory,
                "category_counts": dict(self._category_counts),
                "file_logging_enabled": self._write_enabled,
            }
            return HealthStatus(healthy=True, component="audit_logger", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="audit_logger", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        async with self._lock:
            return self.health_check()


__all__ = [
    "AuditLogger",
    "AuditEvent",
    "EventCategory",
    "EventSeverity",
    "HealthStatus",
]
