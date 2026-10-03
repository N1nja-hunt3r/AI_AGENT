"""
env_manager.py

Environment variable and secrets management: typed accessors, .env
file loading, validation against required keys, and runtime reload.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")
_SENSITIVE_MARKERS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")


class EnvValidationError(Exception):
    """Raised when required environment variables are missing or invalid."""

    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = list(missing)
        super().__init__(f"Missing required environment variables: {self.missing}")


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def parse_env_file(path: Union[str, Path]) -> Dict[str, str]:
    """Parse a .env-style file into a dictionary without mutating os.environ."""
    file_path = Path(path)
    result: Dict[str, str] = {}
    if not file_path.exists():
        return result
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_LINE_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), _strip_quotes(match.group(2))
        result[key] = value
    return result


def is_sensitive_key(key: str) -> bool:
    """Heuristically determine whether an environment variable name is sensitive."""
    upper = key.upper()
    return any(marker in upper for marker in _SENSITIVE_MARKERS)


def mask_value(value: str, visible_chars: int = 4) -> str:
    """Mask a sensitive value, leaving only the trailing characters visible."""
    if not value:
        return ""
    if len(value) <= visible_chars:
        return "*" * len(value)
    return ("*" * (len(value) - visible_chars)) + value[-visible_chars:]


class EnvManager:
    """Thread-safe manager for layered environment variable access."""

    def __init__(self, env_file: Optional[Union[str, Path]] = None) -> None:
        self._env_file = Path(env_file) if env_file else None
        self._lock = threading.RLock()
        self._overlay: Dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        """Reload variables from the configured .env file into an overlay."""
        with self._lock:
            self._overlay = parse_env_file(self._env_file) if self._env_file else {}

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a variable, preferring real process environment over .env overlay."""
        with self._lock:
            if key in os.environ:
                return os.environ[key]
            return self._overlay.get(key, default)

    def get_required(self, key: str) -> str:
        """Get a required variable, raising EnvValidationError if absent."""
        value = self.get(key)
        if value is None:
            raise EnvValidationError([key])
        return value

    def get_bool(self, key: str, default: bool = False) -> bool:
        value = self.get(key)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"Environment variable '{key}' is not a valid int: {value}") from exc

    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        value = self.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Environment variable '{key}' is not a valid float: {value}") from exc

    def get_list(self, key: str, *, separator: str = ",", default: Optional[List[str]] = None) -> List[str]:
        value = self.get(key)
        if value is None:
            return default or []
        return [item.strip() for item in value.split(separator) if item.strip()]

    def set_runtime(self, key: str, value: str, *, persist_to_environ: bool = True) -> None:
        """Set a variable for the current process, optionally into os.environ."""
        with self._lock:
            self._overlay[key] = value
            if persist_to_environ:
                os.environ[key] = value

    def unset_runtime(self, key: str) -> None:
        """Remove a variable from both the overlay and process environment."""
        with self._lock:
            self._overlay.pop(key, None)
            os.environ.pop(key, None)

    def validate_required(self, keys: Sequence[str]) -> None:
        """Raise EnvValidationError if any of the given keys are unset."""
        missing = [key for key in keys if self.get(key) is None]
        if missing:
            raise EnvValidationError(missing)

    def all_keys(self) -> List[str]:
        """Return the union of process and overlay environment variable names."""
        with self._lock:
            return sorted(set(os.environ.keys()) | set(self._overlay.keys()))

    def snapshot(self, *, mask_sensitive: bool = True) -> Dict[str, str]:
        """Return a dict snapshot of all visible variables, masking secrets by default."""
        with self._lock:
            merged: Dict[str, str] = {**self._overlay, **os.environ}
        if not mask_sensitive:
            return merged
        return {
            key: (mask_value(value) if is_sensitive_key(key) else value)
            for key, value in merged.items()
        }

    def diff_from_file(self) -> Dict[str, Any]:
        """Compare process environment values against the configured .env file."""
        if not self._env_file:
            return {}
        file_values = parse_env_file(self._env_file)
        diffs: Dict[str, Any] = {}
        for key, file_value in file_values.items():
            process_value = os.environ.get(key)
            if process_value is not None and process_value != file_value:
                diffs[key] = {"file": "***" if is_sensitive_key(key) else file_value,
                              "process": "***" if is_sensitive_key(key) else process_value}
        return diffs


_default_manager: Optional[EnvManager] = None
_default_manager_lock = threading.Lock()


def get_env_manager(env_file: Optional[Union[str, Path]] = ".env") -> EnvManager:
    """Return the process-wide singleton EnvManager, creating it if needed."""
    global _default_manager
    with _default_manager_lock:
        if _default_manager is None:
            _default_manager = EnvManager(env_file)
        return _default_manager


def reset_env_manager() -> None:
    """Reset the singleton EnvManager (primarily for testing)."""
    global _default_manager
    with _default_manager_lock:
        _default_manager = None
