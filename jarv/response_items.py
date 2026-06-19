"""Responses API input and stored history item builders."""

from __future__ import annotations

import hashlib
from typing import Any

from .tool_outputs import ToolOutput


def responses_input_id(item_id: str, prefix: str) -> str:
    """Return an id that is valid for Responses API input items."""
    valid_prefix = f"{prefix}_"
    if item_id.startswith(valid_prefix) and len(item_id) <= 64:
        return item_id
    digest_len = 64 - len(valid_prefix)
    digest = hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:digest_len]
    return f"{valid_prefix}{digest}"


def reasoning_history_item(item: Any, metadata: dict | None = None) -> dict:
    result = {
        "type": "reasoning",
        "id": getattr(item, "id"),
        "summary": getattr(item, "summary", []),
        **(metadata or {}),
    }
    provider_content = getattr(item, "provider_content", None)
    if provider_content:
        result["provider_content"] = provider_content
    return result


def function_call_history_item(item: Any, metadata: dict | None = None) -> dict:
    result = {
        "type": "function_call",
        "id": getattr(item, "id"),
        "call_id": getattr(item, "call_id"),
        "name": getattr(item, "name"),
        "arguments": getattr(item, "arguments"),
        **(metadata or {}),
    }
    provider_content = getattr(item, "provider_content", None)
    if provider_content:
        result["provider_content"] = provider_content
    return result


def function_call_output_item(
    call_id: str,
    output: ToolOutput,
    metadata: dict | None = None,
) -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
        **(metadata or {}),
    }


def status_history_item(
    text: str,
    phase: str,
    metadata: dict | None = None,
) -> dict:
    return {
        "type": "status",
        "phase": phase,
        "content": text,
        **(metadata or {}),
    }


def to_response_input_item(item: dict) -> dict | None:
    """Convert one stored history item to a Responses API input item."""
    role = item.get("role")
    typ = item.get("type")
    try:
        if role == "user":
            return {"role": "user", "content": str(item.get("content", ""))}
        if role == "assistant":
            return {"role": "assistant", "content": str(item.get("content") or "")}
        if typ == "reasoning" and "id" in item:
            result = {
                "type": "reasoning",
                "id": responses_input_id(str(item["id"]), "rs"),
                "summary": item.get("summary", []),
            }
            if item.get("provider_content"):
                result["provider_content"] = item["provider_content"]
            return result
        if typ == "function_call":
            result = {
                "type": "function_call",
                "id": responses_input_id(str(item["id"]), "fc"),
                "call_id": item["call_id"],
                "name": item["name"],
                "arguments": item["arguments"],
            }
            if item.get("provider_content"):
                result["provider_content"] = item["provider_content"]
            return result
        if typ == "function_call_output":
            return {
                "type": "function_call_output",
                "call_id": item["call_id"],
                "output": item["output"],
            }
    except KeyError:
        return None
    return None
