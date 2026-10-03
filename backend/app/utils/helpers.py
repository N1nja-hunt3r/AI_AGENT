"""
helpers.py

General-purpose utility functions: identifier generation, formatting,
serialization helpers, data conversion, and lightweight health checks.
"""

from __future__ import annotations

import hashlib
import re
import socket
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, TypeVar

T = TypeVar("T")

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


# ---------------------------------------------------------------------------
# Identifier generation
# ---------------------------------------------------------------------------
def generate_uuid() -> str:
    """Generate a random UUID4 string."""
    return str(uuid.uuid4())


def generate_short_id(length: int = 8) -> str:
    """Generate a short hex identifier derived from a UUID4."""
    if length <= 0:
        raise ValueError("length must be positive")
    return uuid.uuid4().hex[:length]


def generate_namespaced_id(namespace: str, length: int = 8) -> str:
    """Generate a namespaced identifier, e.g. 'agent_3f9a2b1c'."""
    clean_ns = slugify(namespace) or "id"
    return f"{clean_ns}_{generate_short_id(length)}"


def deterministic_id(*parts: str) -> str:
    """Generate a deterministic SHA-256-based identifier from input parts."""
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# String formatting
# ---------------------------------------------------------------------------
def slugify(text: str) -> str:
    """Convert text into a lowercase, hyphen-delimited slug."""
    lowered = text.strip().lower()
    slug = _SLUG_RE.sub("-", lowered).strip("-")
    return slug


def camel_to_snake(text: str) -> str:
    """Convert camelCase or PascalCase to snake_case."""
    return _CAMEL_RE.sub("_", text).lower()


def snake_to_camel(text: str, upper_first: bool = False) -> str:
    """Convert snake_case to camelCase or PascalCase."""
    parts = [p for p in text.split("_") if p]
    if not parts:
        return text
    head = parts[0].capitalize() if upper_first else parts[0]
    tail = "".join(p.capitalize() for p in parts[1:])
    return head + tail


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max_length, appending a suffix if shortened."""
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    cut = max(max_length - len(suffix), 0)
    return text[:cut] + suffix


def format_bytes(num_bytes: float) -> str:
    """Format a byte count into a human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string."""
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.0f}s"
    hours, mins = divmod(minutes, 60)
    return f"{int(hours)}h {int(mins)}m"


def mask_secret(value: str, visible_chars: int = 4) -> str:
    """Mask a secret string, leaving only the last `visible_chars` visible."""
    if not value:
        return ""
    if len(value) <= visible_chars:
        return "*" * len(value)
    return ("*" * (len(value) - visible_chars)) + value[-visible_chars:]


# ---------------------------------------------------------------------------
# Serialization / data conversion
# ---------------------------------------------------------------------------
def to_serializable(value: Any) -> Any:
    """Recursively convert a value into a JSON-serializable structure."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return [to_serializable(v) for v in value]
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return to_serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return to_serializable(value.model_dump())
    if hasattr(value, "__dict__"):
        return to_serializable(vars(value))
    return str(value)


def flatten_dict(data: Mapping[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested mapping into a single-level dict with joined keys."""
    items: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            items.update(flatten_dict(value, full_key, sep=sep))
        else:
            items[full_key] = value
    return items


def chunk_list(items: List[T], size: int) -> List[List[T]]:
    """Split a list into chunks of at most `size` elements."""
    if size <= 0:
        raise ValueError("size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def remove_none_values(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a new dict excluding keys whose values are None."""
    return {k: v for k, v in data.items() if v is not None}


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------
def check_tcp_port(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check whether a TCP host:port is reachable."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def build_health_report(
    name: str,
    healthy: bool,
    *,
    latency_ms: Optional[float] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a standardized health-check report dictionary."""
    report: Dict[str, Any] = {
        "component": name,
        "status": "healthy" if healthy else "unhealthy",
        "checked_at": utc_now_iso(),
    }
    if latency_ms is not None:
        report["latency_ms"] = round(latency_ms, 2)
    if error:
        report["error"] = error
    if metadata:
        report["metadata"] = metadata
    return report


def aggregate_health_reports(reports: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multiple health reports into a single summary."""
    reports_list = list(reports)
    unhealthy = [r for r in reports_list if r.get("status") != "healthy"]
    return {
        "status": "healthy" if not unhealthy else "degraded",
        "total_components": len(reports_list),
        "unhealthy_components": [r["component"] for r in unhealthy],
        "checked_at": utc_now_iso(),
        "details": reports_list,
    }
