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


def append_tool_result_input_items(
    target: list[dict],
    item,
    output: ToolOutput,
    *,
    history: list | None = None,
    metadata: dict | None = None,
) -> None:
    metadata = metadata or {}
    for stored_item in (
        function_call_history_item(item, metadata),
        function_call_output_item(item.call_id, output, metadata),
    ):
        if history is not None:
            history.append(stored_item)
        api_item = to_response_input_item(stored_item)
        if api_item is not None:
            target.append(api_item)
