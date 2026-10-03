"""
logger_utils.py

Structured logging utilities: JSON-formatted log records, rotating
file handlers, contextual fields, and a simple setup entry point.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_context_fields_ctx: ContextVar[Dict[str, Any]] = ContextVar("context_fields", default={})

_DEFAULT_LOG_RECORD_KEYS = frozenset(logging.LogRecord(
    "", 0, "", 0, "", (), None
).__dict__.keys())


def set_request_id(request_id: Optional[str] = None) -> str:
    """Bind a request id to the current context, generating one if absent."""
    value = request_id or str(uuid.uuid4())
    _request_id_ctx.set(value)
    return value


def get_request_id() -> Optional[str]:
    """Return the request id bound to the current context, if any."""
    return _request_id_ctx.get()


def bind_context(**fields: Any) -> None:
    """Merge additional structured fields into the current logging context."""
    current = dict(_context_fields_ctx.get())
    current.update(fields)
    _context_fields_ctx.set(current)


def clear_context() -> None:
    """Clear all bound contextual logging fields."""
    _context_fields_ctx.set({})
    _request_id_ctx.set(None)


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def __init__(self, *, service_name: str = "app", static_fields: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self._service_name = service_name
        self._static_fields = static_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self._service_name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        payload.update(self._static_fields)

        request_id = get_request_id()
        if request_id:
            payload["request_id"] = request_id
        context_fields = _context_fields_ctx.get()
        if context_fields:
            payload.update(context_fields)

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _DEFAULT_LOG_RECORD_KEYS and not key.startswith("_")
        }
        if extras:
            payload.update(extras)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for local development console output."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

    def format(self, record: logging.LogRecord) -> str:
        request_id = get_request_id()
        base = super().format(record)
        return f"{base} | request_id={request_id}" if request_id else base


def configure_logging(
    *,
    level: str = "INFO",
    json_format: bool = True,
    service_name: str = "app",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    propagate: bool = False,
) -> logging.Logger:
    """
    Configure the root application logger with console output and optional
    rotating file output. Idempotent: clears existing handlers first.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter: logging.Formatter = (
        JSONFormatter(service_name=service_name) if json_format else TextFormatter()
    )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.propagate = propagate
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured to use the global handler setup."""
    return logging.getLogger(name)


class LogTimer:
    """Context manager that logs the duration of a code block."""

    def __init__(self, logger: logging.Logger, label: str, *, level: int = logging.INFO) -> None:
        self._logger = logger
        self._label = label
        self._level = level
        self._start: float = 0.0

    def __enter__(self) -> "LogTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed_ms = (time.monotonic() - self._start) * 1000
        status = "failed" if exc_type else "completed"
        self._logger.log(
            self._level,
            "%s %s in %.2fms",
            self._label,
            status,
            elapsed_ms,
            extra={"duration_ms": round(elapsed_ms, 2), "status": status},
        )


class ThreadSafeCounterHandler(logging.Handler):
    """A lightweight handler that tallies emitted log records by level."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        with self._lock:
            self._counts[record.levelname] = self._counts.get(record.levelname, 0) + 1

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts)
