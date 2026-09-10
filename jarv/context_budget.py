"""Token-budget context windowing and history compaction."""

from __future__ import annotations

from typing import Any

from .config import get_setting, setting_default
from .tool_outputs import summarize_tool_output
from .retained_outputs import RetainedOutputStore
from .usage import estimate_context_breakdown, estimate_item_tokens, resolve_context_window


_COMPACTION_PREFIX = "[Compacted earlier conversation]"
_COMPACTION_TARGET_RATIO = 0.72


def _config_ratio(config: dict, key: str) -> float:
    try:
        value = float(get_setting(config, key))
    except (TypeError, ValueError):
        value = float(setting_default(key))
    return max(0.05, min(value, 0.95))


def fixed_context_tokens(model: str, instructions: str, tools: list) -> int:
    breakdown = estimate_context_breakdown(model, instructions, tools, [])
    return int(breakdown.get("system", 0)) + int(breakdown.get("tools", 0))


def input_token_budget(model: str, config: dict, instructions: str, tools: list) -> int:
    """Maximum estimated input tokens for system, tools, and conversation."""
    window = resolve_context_window(model, config)
    budget_ratio = _config_ratio(config, "context_budget_ratio")
    output_reserve = _config_ratio(config, "context_output_reserve_ratio")
    fixed = fixed_context_tokens(model, instructions, tools)
    budget = int(window * budget_ratio) - int(window * output_reserve) - fixed
    return max(budget, 256)


def history_token_budget(model: str, config: dict, instructions: str, tools: list) -> int:
    """Token budget reserved for stored conversation history."""
    return input_token_budget(model, config, instructions, tools)


def turn_input_token_budget(model: str, config: dict, instructions: str, tools: list) -> int:
    """Token budget for kwargs['input'] within an active turn."""
    return input_token_budget(model, config, instructions, tools)


def _to_api_item(stored_item: dict) -> dict | None:
    from .response_items import to_response_input_item

    return to_response_input_item(stored_item)


def history_to_api_items(history: list) -> list[dict]:
    items: list[dict] = []
    for stored in history:
        if not isinstance(stored, dict):
            continue
        api_item = _to_api_item(stored)
        if api_item is not None:
            items.append(api_item)
    return items


def _align_slice_to_user(items: list[dict]) -> list[dict]:
    for index, item in enumerate(items):
        if isinstance(item, dict) and item.get("role") == "user":
            return items[index:]
    return []


def estimate_api_items_tokens(model: str, items: list) -> int:
    total = 0
    for item in items:
        if isinstance(item, dict):
            total += estimate_item_tokens(model, item)
    return total


def estimate_history_tokens(model: str, history: list) -> int:
    return estimate_api_items_tokens(model, history_to_api_items(history))


def trim_items_to_budget(items: list[dict], model: str, budget: int) -> list[dict]:
    """Keep complete turns, always preserving the newest even if it exceeds budget.

    The budget is soft when a single turn cannot fit: losing executed actions
    is worse than letting the caller handle an oversized context.
    """
    if budget <= 0 or not items:
        return []

    items = _align_slice_to_user(items)
    ranges = iter_turn_ranges(items)
    start = len(items)
    used = 0
    for turn_start, turn_end in reversed(ranges):
        count = estimate_api_items_tokens(model, items[turn_start:turn_end])
        if start < len(items) and used + count > budget:
            break
        start = turn_start
        used += count
    return items[start:]


def _should_compact_history(
    model: str,
    config: dict,
    instructions: str,
    tools: list,
    history: list,
) -> bool:
    threshold = _config_ratio(config, "context_compaction_threshold")
    fill = estimated_context_fill_ratio(model, config, instructions, tools, history)
    return fill is not None and fill >= threshold


def build_input(
    history: list,
    *,
    model: str,
    config: dict,
    instructions: str = "",
    tools: list | None = None,
    retained_store: RetainedOutputStore | None = None,
) -> list[dict]:
    """Convert stored history to Responses API input within the token budget."""
    tools = tools or []
    source = history
    if _should_compact_history(model, config, instructions, tools, history):
        target_tokens = int(
            history_token_budget(model, config, instructions, tools) * _COMPACTION_TARGET_RATIO
        )
        source = compact_history_view(
            history,
            model=model,
            config=config,
            instructions=instructions,
            tools=tools,
            target_tokens=target_tokens,
        )
    api_items = history_to_api_items(source)
    budget = history_token_budget(model, config, instructions, tools)
    return _trim_with_retained_outputs(api_items, model, budget, retained_store)


def _trim_with_retained_outputs(
    items: list[dict], model: str, budget: int,
    retained_store: RetainedOutputStore | None,
) -> list[dict]:
    kept = trim_items_to_budget(items, model, budget)
    if retained_store is None:
        return kept
    # Reduce the largest results first. Never rewrite call arguments, IDs, or
    # reasoning signatures. Copies leave the full stored transcript untouched.
    for index in sorted(
        range(len(kept)),
        key=lambda i: estimate_item_tokens(model, kept[i]), reverse=True,
    ):
        if estimate_api_items_tokens(model, kept) <= budget:
            break
        item = kept[index]
        if item.get("type") != "function_call_output":
            continue
        output = item.get("output")
        # Preserve multimodal payloads; flattening them would lose image data.
        if not isinstance(output, str) or output.startswith("[Retained tool result"):
            continue
        preview = output[:160] + "\n...\n" + output[-160:]
        if len(output) <= len(preview) + 160:
            continue
        output_id = retained_store.put(output)
        kept[index] = {
            **item,
            "output": (
                f'[Retained tool result: {output_id}]\n'
                f'Retrieve with read(input="{output_id}", offset=0, size=2000).\n'
                f"{preview}"
            ),
        }
    return kept


def trim_turn_input(
    input_items: list,
    *,
    model: str,
    config: dict,
    instructions: str,
    tools: list,
    retained_store: RetainedOutputStore | None = None,
) -> list[dict]:
    """Trim in-turn ``kwargs['input']`` growth to the active token budget."""
    budget = turn_input_token_budget(model, config, instructions, tools)
    return _trim_with_retained_outputs(
        [item for item in input_items if isinstance(item, dict)],
        model,
        budget,
        retained_store,
    )


def _is_turn_start(item: dict) -> bool:
    return item.get("role") == "user"


def iter_turn_ranges(history: list) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(history):
        while index < len(history) and not _is_turn_start(history[index]):
            index += 1
        if index >= len(history):
            break
        start = index
        index += 1
        while index < len(history) and not _is_turn_start(history[index]):
            index += 1
        ranges.append((start, index))
    return ranges


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def summarize_turn_items(turn_items: list[dict], *, max_chars_per_item: int = 400) -> str:
    lines = [_COMPACTION_PREFIX]
    for item in turn_items:
        role = item.get("role")
        typ = item.get("type")
        if role == "user":
            lines.append(f"User: {_truncate_text(str(item.get('content') or ''), max_chars_per_item)}")
        elif role == "assistant":
            lines.append(
                f"Assistant: {_truncate_text(str(item.get('content') or ''), max_chars_per_item)}"
            )
        elif typ == "function_call":
            lines.append(
                "Tool "
                f"{item.get('name', '')}: "
                f"{_truncate_text(str(item.get('arguments') or ''), max_chars_per_item // 2)}"
            )
        elif typ == "function_call_output":
            lines.append(
                "Tool output: "
                f"{_truncate_text(summarize_tool_output(item.get('output')), max_chars_per_item)}"
            )
        elif typ == "reasoning":
            lines.append("Reasoning: [omitted]")
    return "\n".join(lines)


def compact_oldest_turns(
    history: list,
    *,
    model: str,
    config: dict,
    instructions: str,
    tools: list,
    target_tokens: int,
) -> bool:
    """Replace the oldest complete turn with a compact summary until under budget.

    Mutates ``history`` in place. Callers should pass a copy when the original
    stored transcript must be preserved.
    """
    modified = False
    while estimate_history_tokens(model, history) > target_tokens:
        ranges = iter_turn_ranges(history)
        candidates = [
            (start, end) for start, end in ranges[:-1]
            if history[start].get("type") != "compacted_summary"
        ]
        if not candidates:
            break
        start, end = candidates[0]
        summary = summarize_turn_items(history[start:end])
        history[start:end] = [{
            "role": "user",
            "type": "compacted_summary",
            "content": summary,
        }]
        modified = True
    return modified


def compact_history_view(
    history: list,
    *,
    model: str,
    config: dict,
    instructions: str,
    tools: list,
    target_tokens: int,
) -> list:
    """Return a compacted copy of ``history`` for API input without mutating storage."""
    view = list(history)
    compact_oldest_turns(
        view,
        model=model,
        config=config,
        instructions=instructions,
        tools=tools,
        target_tokens=target_tokens,
    )
    return view


def estimated_context_fill_ratio(
    model: str,
    config: dict,
    instructions: str,
    tools: list,
    history: list,
) -> float | None:
    window = resolve_context_window(model, config)
    if window <= 0:
        return None
    fixed = fixed_context_tokens(model, instructions, tools)
    history_tokens = estimate_history_tokens(model, history)
    return (fixed + history_tokens) / window


def context_budget_status(
    model: str,
    config: dict,
    instructions: str,
    tools: list,
    history: list,
) -> dict[str, Any]:
    """Return budget diagnostics for tests and observability."""
    window = resolve_context_window(model, config)
    fixed = fixed_context_tokens(model, instructions, tools)
    history_tokens = estimate_history_tokens(model, history)
    budget = history_token_budget(model, config, instructions, tools)
    fill = estimated_context_fill_ratio(model, config, instructions, tools, history)
    return {
        "context_window": window,
        "fixed_tokens": fixed,
        "history_tokens": history_tokens,
        "history_budget": budget,
        "fill_ratio": fill,
    }
