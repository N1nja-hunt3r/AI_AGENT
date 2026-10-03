from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


_LEVEL_MAP: dict[LogLevel, int] = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.CRITICAL: logging.CRITICAL,
}

_correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("correlation_id", default=None)
_context_var: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("log_context", default={})


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class _JsonFormatter(logging.Formatter):
    def __init__(self, service_name: str) -> None:
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "epoch": record.created,
            "level": record.levelname.lower(),
            "service": self.service_name,
            "logger": record.name,
            "message": record.getMessage(),
        }
        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id:
            payload["correlation_id"] = correlation_id
        context = getattr(record, "context", None)
        if context:
            payload["context"] = context
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            payload["fields"] = extra_fields
        if record.exc_info:
            payload["exception"] = "".join(traceback.format_exception(*record.exc_info))
        payload["module"] = record.module
        payload["function"] = record.funcName
        payload["line"] = record.lineno
        return json.dumps(payload, default=str, sort_keys=True)


class _ConsoleFormatter(logging.Formatter):
    _COLORS: dict[int, str] = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[41m\033[97m",
    }
    _RESET = "\033[0m"

    def __init__(self, *, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        level = record.levelname.ljust(8)
        correlation_id = getattr(record, "correlation_id", None)
        cid_part = f" [{correlation_id}]" if correlation_id else ""
        base = f"{ts} {level} {record.name}{cid_part} - {record.getMessage()}"
        context = getattr(record, "context", None)
        if context:
            base += f" | context={json.dumps(context, default=str)}"
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            base += f" | fields={json.dumps(extra_fields, default=str)}"
        if record.exc_info:
            base += "\n" + "".join(traceback.format_exception(*record.exc_info))
        if self.use_color:
            color = self._COLORS.get(record.levelno, "")
            return f"{color}{base}{self._RESET}"
        return base


class Logger:
    """Production structured logger: JSON file logging with rotation, console logging, correlation IDs, async support."""

    def __init__(
        self,
        name: str = "app",
        *,
        service_name: str = "service",
        level: LogLevel = LogLevel.INFO,
        log_file: Optional[str | Path] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        console_enabled: bool = True,
        console_color: bool = True,
        console_json: bool = False,
    ) -> None:
        self.name = name
        self.service_name = service_name
        self._logger = logging.getLogger(f"{service_name}.{name}.{uuid.uuid4().hex[:8]}")
        self._logger.setLevel(_LEVEL_MAP[level])
        self._logger.propagate = False
        self._logger.handlers.clear()

        self._file_enabled = False
        if log_file is not None:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(str(path), maxBytes=max_bytes, backupCount=backup_count)
            file_handler.setFormatter(_JsonFormatter(service_name))
            self._logger.addHandler(file_handler)
            self._file_enabled = True

        self._console_enabled = console_enabled
        if console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            if console_json:
                console_handler.setFormatter(_JsonFormatter(service_name))
            else:
                console_handler.setFormatter(_ConsoleFormatter(use_color=console_color))
            self._logger.addHandler(console_handler)

        self._created_at = time.time()
        self._counts: dict[str, int] = {lvl.value: 0 for lvl in LogLevel}
        self._lock = asyncio.Lock()

    @staticmethod
    def new_correlation_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def set_correlation_id(correlation_id: Optional[str]) -> contextvars.Token[Optional[str]]:
        return _correlation_id_var.set(correlation_id)

    @staticmethod
    def get_correlation_id() -> Optional[str]:
        return _correlation_id_var.get()

    @staticmethod
    def reset_correlation_id(token: contextvars.Token[Optional[str]]) -> None:
        _correlation_id_var.reset(token)

    @staticmethod
    def set_context(**kwargs: Any) -> contextvars.Token[dict[str, Any]]:
        merged = {**_context_var.get(), **kwargs}
        return _context_var.set(merged)

    @staticmethod
    def get_context() -> dict[str, Any]:
        return dict(_context_var.get())

    @staticmethod
    def reset_context(token: contextvars.Token[dict[str, Any]]) -> None:
        _context_var.reset(token)

    @staticmethod
    def clear_context() -> None:
        _context_var.set({})

    def _log(
        self,
        level: LogLevel,
        message: str,
        *,
        correlation_id: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        exc_info: bool = False,
        **fields: Any,
    ) -> None:
        self._counts[level.value] += 1
        cid = correlation_id if correlation_id is not None else _correlation_id_var.get()
        merged_context = {**_context_var.get(), **(context or {})}
        extra = {
            "correlation_id": cid,
            "context": merged_context or None,
            "extra_fields": fields or None,
        }
        self._logger.log(_LEVEL_MAP[level], message, exc_info=exc_info, extra=extra)

    def debug(self, message: str, *, correlation_id: Optional[str] = None, context: Optional[dict[str, Any]] = None, **fields: Any) -> None:
        self._log(LogLevel.DEBUG, message, correlation_id=correlation_id, context=context, **fields)

    def info(self, message: str, *, correlation_id: Optional[str] = None, context: Optional[dict[str, Any]] = None, **fields: Any) -> None:
        self._log(LogLevel.INFO, message, correlation_id=correlation_id, context=context, **fields)

    def warning(self, message: str, *, correlation_id: Optional[str] = None, context: Optional[dict[str, Any]] = None, **fields: Any) -> None:
        self._log(LogLevel.WARNING, message, correlation_id=correlation_id, context=context, **fields)

    def error(
        self,
        message: str,
        *,
        correlation_id: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        exc_info: bool = False,
        **fields: Any,
    ) -> None:
        self._log(LogLevel.ERROR, message, correlation_id=correlation_id, context=context, exc_info=exc_info, **fields)

    def critical(
        self,
        message: str,
        *,
        correlation_id: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        exc_info: bool = False,
        **fields: Any,
    ) -> None:
        self._log(LogLevel.CRITICAL, message, correlation_id=correlation_id, context=context, exc_info=exc_info, **fields)

    async def debug_async(self, message: str, **kwargs: Any) -> None:
        async with self._lock:
            await asyncio.to_thread(self.debug, message, **kwargs)

    async def info_async(self, message: str, **kwargs: Any) -> None:
        async with self._lock:
            await asyncio.to_thread(self.info, message, **kwargs)

    async def warning_async(self, message: str, **kwargs: Any) -> None:
        async with self._lock:
            await asyncio.to_thread(self.warning, message, **kwargs)

    async def error_async(self, message: str, **kwargs: Any) -> None:
        async with self._lock:
            await asyncio.to_thread(self.error, message, **kwargs)

    async def critical_async(self, message: str, **kwargs: Any) -> None:
        async with self._lock:
            await asyncio.to_thread(self.critical, message, **kwargs)

    def set_level(self, level: LogLevel) -> None:
        self._logger.setLevel(_LEVEL_MAP[level])

    def health_check(self) -> HealthStatus:
        try:
            handlers_ok = len(self._logger.handlers) > 0
            details = {
                "uptime_seconds": round(time.time() - self._created_at, 2),
                "name": self.name,
                "service_name": self.service_name,
                "level": logging.getLevelName(self._logger.level),
                "file_logging_enabled": self._file_enabled,
                "console_logging_enabled": self._console_enabled,
                "handler_count": len(self._logger.handlers),
                "log_counts": dict(self._counts),
            }
            return HealthStatus(healthy=handlers_ok, component="logger", details=details)
        except Exception as exc:
            return HealthStatus(healthy=False, component="logger", details={"error": str(exc)})

    async def health_check_async(self) -> HealthStatus:
        return await asyncio.to_thread(self.health_check)


__all__ = [
    "Logger",
    "LogLevel",
    "HealthStatus",
]
