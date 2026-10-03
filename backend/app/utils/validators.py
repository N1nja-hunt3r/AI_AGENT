"""
validators.py

Input validation helpers: email, URL, file, JSON, prompt content,
and minimal schema validation, all returning structured results
rather than relying on exceptions for control flow where possible.
"""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$"
)

_PROMPT_INJECTION_MARKERS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard the system prompt",
    "you are now in developer mode",
)

DEFAULT_ALLOWED_SCHEMES: tuple[str, ...] = ("http", "https")
DEFAULT_MAX_PROMPT_CHARS: int = 1_000_000


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.valid = False
        self.errors.append(message)

    def __bool__(self) -> bool:
        return self.valid


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
def is_valid_email(value: str) -> bool:
    """Return True if `value` looks like a syntactically valid email address."""
    if not value or len(value) > 254:
        return False
    return bool(_EMAIL_RE.match(value.strip()))


def validate_email(value: str) -> ValidationResult:
    result = ValidationResult(valid=True)
    if not value:
        result.add_error("email must not be empty")
    elif len(value) > 254:
        result.add_error("email exceeds maximum length of 254 characters")
    elif not _EMAIL_RE.match(value.strip()):
        result.add_error(f"'{value}' is not a valid email address")
    return result


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------
def is_valid_url(value: str, allowed_schemes: Sequence[str] = DEFAULT_ALLOWED_SCHEMES) -> bool:
    """Return True if `value` is a well-formed URL with an allowed scheme."""
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    if parsed.scheme not in allowed_schemes:
        return False
    if not parsed.netloc:
        return False
    return True


def validate_url(value: str, allowed_schemes: Sequence[str] = DEFAULT_ALLOWED_SCHEMES) -> ValidationResult:
    result = ValidationResult(valid=True)
    if not value:
        result.add_error("url must not be empty")
        return result
    parsed = urlparse(value)
    if parsed.scheme not in allowed_schemes:
        result.add_error(
            f"scheme '{parsed.scheme}' not allowed; expected one of {list(allowed_schemes)}"
        )
    if not parsed.netloc:
        result.add_error("url is missing a network location (host)")
    return result


def is_private_ip_host(host: str) -> bool:
    """Return True if a hostname/IP literal resolves to a private/internal range."""
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        return host in {"localhost"}


# ---------------------------------------------------------------------------
# File
# ---------------------------------------------------------------------------
def validate_file_path(
    path: str,
    *,
    must_exist: bool = False,
    allowed_extensions: Optional[Sequence[str]] = None,
    max_size_bytes: Optional[int] = None,
) -> ValidationResult:
    """Validate a file path against existence, extension, and size constraints."""
    result = ValidationResult(valid=True)
    if not path:
        result.add_error("file path must not be empty")
        return result

    if allowed_extensions:
        ext = os.path.splitext(path)[1].lower()
        normalized = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in allowed_extensions}
        if ext not in normalized:
            result.add_error(f"extension '{ext}' not in allowed set {sorted(normalized)}")

    if must_exist and not os.path.isfile(path):
        result.add_error(f"file does not exist: {path}")
        return result

    if max_size_bytes is not None and os.path.isfile(path):
        size = os.path.getsize(path)
        if size > max_size_bytes:
            result.add_error(f"file size {size} exceeds maximum {max_size_bytes} bytes")

    return result


def has_path_traversal(path: str) -> bool:
    """Detect potential directory traversal sequences in a path."""
    normalized = os.path.normpath(path)
    return ".." in normalized.split(os.sep)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def is_json_serializable(value: Any) -> bool:
    """Return True if a value can be serialized to JSON without error."""
    import json

    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def validate_json_keys(data: Mapping[str, Any], required: Sequence[str]) -> ValidationResult:
    result = ValidationResult(valid=True)
    missing = [k for k in required if k not in data]
    if missing:
        result.add_error(f"missing required keys: {missing}")
    return result


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------
def validate_prompt(
    prompt: str,
    *,
    max_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    min_chars: int = 1,
    check_injection_markers: bool = True,
) -> ValidationResult:
    """Validate prompt text against length bounds and known injection markers."""
    result = ValidationResult(valid=True)
    if len(prompt) < min_chars:
        result.add_error(f"prompt shorter than minimum length {min_chars}")
    if len(prompt) > max_chars:
        result.add_error(f"prompt exceeds maximum length {max_chars}")
    if check_injection_markers:
        lowered = prompt.lower()
        found = [marker for marker in _PROMPT_INJECTION_MARKERS if marker in lowered]
        if found:
            result.add_error(f"prompt contains suspicious instruction-override phrasing: {found}")
    return result


# ---------------------------------------------------------------------------
# Generic field validation
# ---------------------------------------------------------------------------
def validate_string_length(
    value: str, *, min_length: int = 0, max_length: Optional[int] = None, field_name: str = "value"
) -> ValidationResult:
    result = ValidationResult(valid=True)
    if len(value) < min_length:
        result.add_error(f"{field_name} must be at least {min_length} characters")
    if max_length is not None and len(value) > max_length:
        result.add_error(f"{field_name} must be at most {max_length} characters")
    return result


def validate_numeric_range(
    value: float, *, min_value: Optional[float] = None, max_value: Optional[float] = None, field_name: str = "value"
) -> ValidationResult:
    result = ValidationResult(valid=True)
    if min_value is not None and value < min_value:
        result.add_error(f"{field_name} must be >= {min_value}")
    if max_value is not None and value > max_value:
        result.add_error(f"{field_name} must be <= {max_value}")
    return result


def combine_results(results: Sequence[ValidationResult]) -> ValidationResult:
    """Combine multiple ValidationResult instances into a single result."""
    combined = ValidationResult(valid=True)
    for r in results:
        if not r.valid:
            combined.valid = False
            combined.errors.extend(r.errors)
    return combined
