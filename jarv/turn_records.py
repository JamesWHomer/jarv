"""Shared turn record assembly for root agents and subagents."""

from .response_items import (
    function_call_history_item,
    function_call_output_item,
    reasoning_history_item,
    to_response_input_item,
)
from .tool_outputs import ToolOutput


def stream_usage_output_text(reply_text: str, tool_calls: list) -> str:
    if reply_text:
        return reply_text
    return "\n".join(f"{item.name} {item.arguments}" for item in tool_calls)


def append_reasoning_input_items(
    target: list[dict],
    reasoning_items: list,
    *,
    history: list | None = None,
    metadata: dict | None = None,
) -> None:
    metadata = metadata or {}
    for item in reasoning_items:
        stored_item = reasoning_history_item(item, metadata)
        if history is not None:
            history.append(stored_item)
        api_item = to_response_input_item(stored_item)
        if api_item is not None:
            target.append(api_item)


def append_assistant_response_input_items(
    target: list[dict],
    reasoning_items: list,
    reply_text: str,
    tool_calls: list,
    *,
    history: list | None = None,
    metadata: dict | None = None,
) -> None:
    """Record the whole assistant response before any tool results."""
    append_reasoning_input_items(
        target, reasoning_items, history=history, metadata=metadata,
    )
    stored_items = []
    if reply_text:
        stored_items.append({"role": "assistant", "content": reply_text, **(metadata or {})})
    stored_items.extend(function_call_history_item(item, metadata) for item in tool_calls)
    for stored_item in stored_items:
        if history is not None:
            history.append(stored_item)
        api_item = to_response_input_item(stored_item)
        if api_item is not None:
            target.append(api_item)


def append_tool_result_input_items(
    target: list[dict],
    item,
    output: ToolOutput,
    *,
    history: list | None = None,
    metadata: dict | None = None,
    include_call: bool = True,
) -> None:
    metadata = metadata or {}
    stored_items = []
    if include_call:
        stored_items.append(function_call_history_item(item, metadata))
    stored_items.append(function_call_output_item(item.call_id, output, metadata))
    for stored_item in stored_items:
        if history is not None:
            history.append(stored_item)
        api_item = to_response_input_item(stored_item)
        if api_item is not None:
            target.append(api_item)
