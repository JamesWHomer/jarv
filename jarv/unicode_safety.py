"""Helpers for keeping API and persisted JSON text UTF-8 encodable."""

from collections.abc import Mapping
from typing import Any


def sanitize_text(value: str) -> str:
    """Replace lone surrogate code points so text can be encoded as UTF-8."""
    return value.encode("utf-8", errors="replace").decode("utf-8")


def sanitize_json_value(value: Any) -> Any:
    """Recursively sanitize strings in JSON-like data structures."""
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            sanitize_text(key) if isinstance(key, str) else key: sanitize_json_value(item)
            for key, item in value.items()
        }
    return value
