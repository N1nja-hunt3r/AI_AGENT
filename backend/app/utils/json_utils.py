"""
json_utils.py

JSON serialization, deserialization, safe parsing, and lightweight
schema validation helpers.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class JSONDecodeError(Exception):
    """Raised when a string cannot be parsed as valid JSON."""


class SchemaValidationError(Exception):
    """Raised when a JSON value fails schema validation."""

    def __init__(self, message: str, path: str = "$") -> None:
        self.path = path
        super().__init__(f"{path}: {message}")


class AppJSONEncoder(json.JSONEncoder):
    """JSON encoder supporting datetimes, enums, decimals, and dataclasses."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return vars(obj)
        return super().default(obj)


def to_json(value: Any, *, indent: Optional[int] = None, sort_keys: bool = False) -> str:
    """Serialize a Python value to a JSON string using AppJSONEncoder."""
    return json.dumps(value, cls=AppJSONEncoder, indent=indent, sort_keys=sort_keys, ensure_ascii=False)


def from_json(text: str) -> Any:
    """Deserialize a JSON string, raising JSONDecodeError on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise JSONDecodeError(f"Invalid JSON: {exc}") from exc


def safe_parse_json(text: str, default: Any = None) -> Any:
    """Attempt to parse JSON, returning `default` instead of raising."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def extract_json_from_text(text: str) -> Optional[Any]:
    """
    Extract and parse the first JSON object/array embedded in free-form text,
    handling markdown code fences commonly emitted by LLMs.
    """
    fence_match = _JSON_FENCE_RE.search(text)
    candidates: List[str] = []
    if fence_match:
        candidates.append(fence_match.group(1))

    stripped = text.strip()
    candidates.append(stripped)

    start_chars = {"{": "}", "[": "]"}
    for ch, close in start_chars.items():
        start_idx = text.find(ch)
        end_idx = text.rfind(close)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidates.append(text[start_idx : end_idx + 1])

    for candidate in candidates:
        parsed = safe_parse_json(candidate, default=_SENTINEL)
        if parsed is not _SENTINEL:
            return parsed
    return None


_SENTINEL = object()


def is_valid_json(text: str) -> bool:
    """Return True if the given text is syntactically valid JSON."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def merge_json_objects(*objects: Mapping[str, Any]) -> Dict[str, Any]:
    """Shallow-merge multiple JSON-like mappings, later ones taking precedence."""
    result: Dict[str, Any] = {}
    for obj in objects:
        result.update(obj)
    return result


def pretty_print(value: Any) -> str:
    """Return an indented, human-readable JSON representation of a value."""
    return to_json(value, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Lightweight schema validation
# ---------------------------------------------------------------------------
JSONSchemaType = Union[type, Tuple[type, ...]]


def validate_required_keys(data: Mapping[str, Any], required: Sequence[str], path: str = "$") -> None:
    """Validate that all required keys are present in a mapping."""
    missing = [k for k in required if k not in data]
    if missing:
        raise SchemaValidationError(f"missing required keys: {missing}", path)


def validate_type(value: Any, expected: JSONSchemaType, path: str = "$") -> None:
    """Validate that a value matches an expected Python type or tuple of types."""
    if not isinstance(value, expected):
        expected_name = (
            expected.__name__
            if isinstance(expected, type)
            else " | ".join(t.__name__ for t in expected)
        )
        raise SchemaValidationError(
            f"expected type {expected_name}, got {type(value).__name__}", path
        )


def validate_schema(data: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """
    Validate `data` against a minimal declarative schema dict, e.g.:
        {"type": "object", "required": ["id", "name"], "properties": {
            "id": {"type": str}, "name": {"type": str}}}
    """
    expected_type = schema.get("type")
    if expected_type == "object":
        validate_type(data, dict, path)
        required = schema.get("required", [])
        validate_required_keys(data, required, path)
        properties = schema.get("properties", {})
        for key, sub_schema in properties.items():
            if key in data:
                validate_schema(data[key], sub_schema, f"{path}.{key}")
    elif expected_type == "array":
        validate_type(data, list, path)
        item_schema = schema.get("items")
        if item_schema:
            for idx, item in enumerate(data):
                validate_schema(item, item_schema, f"{path}[{idx}]")
    elif isinstance(expected_type, type) or (
        isinstance(expected_type, tuple) and all(isinstance(t, type) for t in expected_type)
    ):
        validate_type(data, expected_type, path)


def try_validate_schema(data: Any, schema: Mapping[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate schema without raising; returns (is_valid, error_message)."""
    try:
        validate_schema(data, schema)
        return True, None
    except SchemaValidationError as exc:
        return False, str(exc)
