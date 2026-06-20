"""Shared conversion from Jarv Responses-style history items to provider formats."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any


def parse_json_arguments(value: Any, *, fallback: Any = None) -> Any:
    """Parse tool-call arguments from a JSON string or pass through dict values."""
    if not isinstance(value, str):
        return value if value is not None else ({} if fallback is None else fallback)
    try:
        return json.loads(value or "{}")
    except json.JSONDecodeError:
        return {} if fallback is None else fallback


def append_grouped(
    items: list[dict],
    role: str,
    blocks: list[dict],
    *,
    content_key: str = "content",
    parts_key: str | None = None,
) -> None:
    """Append blocks to the last message when the role matches, else start a new one."""
    if not blocks:
        return
    if items and items[-1]["role"] == role:
        existing = items[-1][content_key]
        if isinstance(existing, list):
            existing.extend(blocks)
            return
    if parts_key:
        items.append({"role": role, parts_key: list(blocks)})
    else:
        items.append({"role": role, content_key: list(blocks)})


def iter_history_segments(input_items: list[dict]) -> Iterator[tuple[str, Any]]:
    """Walk canonical history items as typed segments."""
    i = 0
    while i < len(input_items):
        item = input_items[i]
        role = item.get("role")
        typ = item.get("type")

        if role in ("user", "assistant"):
            yield ("message", role, item)
            i += 1
            continue

        if typ == "reasoning":
            yield ("reasoning", item)
            i += 1
            continue

        if typ == "function_call":
            calls: list[dict] = []
            while i < len(input_items) and input_items[i].get("type") == "function_call":
                calls.append(input_items[i])
                i += 1
            yield ("function_calls", calls)
            continue

        if typ == "function_call_output":
            outputs: list[dict] = []
            while (
                i < len(input_items)
                and input_items[i].get("type") == "function_call_output"
            ):
                outputs.append(input_items[i])
                i += 1
            yield ("function_outputs", outputs)
            continue

        i += 1


def convert_tools(
    tools: list[dict],
    *,
    convert_one: Callable[[dict, dict], dict | None],
) -> list[dict]:
    """Convert Responses-style tool definitions with a per-tool callback."""
    result: list[dict] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if tool.get("type") != "function" or not function.get("name"):
            continue
        converted = convert_one(tool, function)
        if converted is not None:
            result.append(converted)
    return result
