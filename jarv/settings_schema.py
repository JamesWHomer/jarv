"""Settings row schema and choice metadata."""

from __future__ import annotations

from .config_schema import (
    COMMAND_SAFETY_CHOICES,
    READ_ONLY_COMMAND_DISPLAY_CHOICES,
    TOOL_CALL_DISPLAY_CHOICES,
    TOOL_NAMES,
)

SETTINGS_SAFETY_CHOICES = tuple(
    (value, label)
    for value, label in (
        ("risky", "flag risky"),
        ("all", "confirm all"),
        ("none", "no prompts"),
    )
    if value in COMMAND_SAFETY_CHOICES
)

SETTINGS_READ_ONLY_DISPLAY_CHOICES = tuple(
    (value, value) for value in READ_ONLY_COMMAND_DISPLAY_CHOICES
)

SETTINGS_TOOL_CALL_DISPLAY_CHOICES = tuple(
    (value, value) for value in TOOL_CALL_DISPLAY_CHOICES
)

SETTINGS_TOOL_LABELS = {
    "run_command": ("Run commands", "execute shell commands"),
    "web_search": ("Web search", "search the web"),
    "read": ("Read", "read files, URLs, artifacts, and retained output"),
    "spawn": ("Subagents", "fan out work to parallel subagents"),
    "ask_user": ("Ask user", "pause to request clarification"),
}


def settings_service_tier_choices(config: dict) -> tuple[tuple[str, str], ...]:
    from .provider_catalog import service_tier_choices

    provider = str(config.get("provider", "openai"))
    return tuple((tier, tier) for tier in service_tier_choices(provider))


def settings_service_tier_description(config: dict) -> str:
    provider = str(config.get("provider", "openai"))
    if provider == "anthropic":
        return "priority uses committed capacity, then falls back to standard"
    if len(settings_service_tier_choices(config)) == 1:
        return "this provider uses standard processing"
    return "standard cost, flex savings, or priority latency"


def settings_rows(config: dict) -> list[dict]:
    from .reasoning import reasoning_effort_choices, reasoning_effort_description

    rows = [
        {
            "section": "account",
            "label": "Provider",
            "key": "provider",
            "kind": "setup",
            "step": "provider",
            "desc": "choose an API provider",
        },
        {
            "section": "account",
            "label": "API key",
            "key": "api_key",
            "kind": "setup",
            "step": "key",
            "desc": "store or replace the active provider key",
        },
        {
            "section": "account",
            "label": "Base URL",
            "key": "base_url",
            "kind": "text",
            "empty": "provider default",
            "desc": "optional custom endpoint",
        },
        {
            "section": "behaviour",
            "label": "Model",
            "key": "model",
            "kind": "setup",
            "step": "model",
            "desc": "pick from the provider presets or enter a model",
        },
        {
            "section": "behaviour",
            "label": "Reasoning effort",
            "key": "reasoning_effort",
            "kind": "choice",
            "choices": reasoning_effort_choices(config),
            "desc": reasoning_effort_description(config),
        },
        {
            "section": "behaviour",
            "label": "System prompt",
            "key": "system_prompt",
            "kind": "text",
            "multiline": True,
            "desc": "instructions sent before each request",
        },
        {
            "section": "display",
            "label": "Read-only commands",
            "key": "read_only_command_display",
            "kind": "choice",
            "choices": SETTINGS_READ_ONLY_DISPLAY_CHOICES,
            "desc": "fullscreen temporary view or permanent print output",
        },
        {
            "section": "display",
            "label": "Tool calls",
            "key": "tool_call_display",
            "kind": "choice",
            "choices": SETTINGS_TOOL_CALL_DISPLAY_CHOICES,
            "desc": "resize-safe print layout or bordered fullscreen cards",
        },
        {
            "section": "display",
            "label": "Print usage",
            "key": "print_usage_after_agent",
            "kind": "bool",
            "desc": "print token totals after completed agent runs",
        },
        {
            "section": "command review",
            "label": "Command approval",
            "key": "command_safety",
            "kind": "choice",
            "choices": SETTINGS_SAFETY_CHOICES,
            "desc": "default: flag only risky commands",
        },
        {
            "section": "command review",
            "label": "Auditor",
            "key": "audit",
            "kind": "bool",
            "desc": "LLM reviews flagged commands first",
        },
        {
            "section": "command review",
            "label": "Audit auto-accept",
            "key": "auditor_auto_approve",
            "kind": "bool",
            "desc": "auto-run commands the auditor marks safe",
        },
        {
            "section": "command review",
            "label": "Auditor model",
            "key": "auditor_model",
            "kind": "text",
            "empty": "default",
            "desc": "use the active model unless overridden",
        },
        {
            "section": "runtime",
            "label": "Command timeout",
            "key": "command_timeout",
            "kind": "int",
            "desc": "seconds before shell commands are killed",
        },
        {
            "section": "runtime",
            "label": "Web timeout",
            "key": "web_timeout",
            "kind": "int",
            "desc": "seconds before web requests are cancelled",
        },
        {
            "section": "runtime",
            "label": "History limit",
            "key": "max_history",
            "kind": "int",
            "desc": "recent stored items sent as context",
        },
        {
            "section": "runtime",
            "label": "Stdin limit",
            "key": "max_stdin_chars",
            "kind": "int",
            "desc": "piped stdin chars attached to one-shot prompts",
        },
        {
            "section": "runtime",
            "label": "Tool output limit",
            "key": "max_tool_output_chars",
            "kind": "int",
            "desc": "tool output chars returned to the model",
        },
        *[
            {
                "section": "tools",
                "label": SETTINGS_TOOL_LABELS[name][0],
                "key": f"tool:{name}",
                "tool_name": name,
                "kind": "tool_bool",
                "desc": SETTINGS_TOOL_LABELS[name][1],
            }
            for name in TOOL_NAMES
        ],
        {
            "section": "subagents",
            "label": "Max depth",
            "key": "max_subagent_depth",
            "kind": "int",
            "desc": "maximum nested spawn depth",
        },
        {
            "section": "subagents",
            "label": "Parallel workers",
            "key": "subagent_thread_pool_max_workers",
            "kind": "int",
            "desc": "subagents per spawn batch",
        },
        {
            "section": "updates",
            "label": "Update checks",
            "key": "check_updates",
            "kind": "bool",
            "desc": "background check on one-shot runs",
        },
    ]
    tier_choices = settings_service_tier_choices(config)
    if len(tier_choices) > 1:
        rows.insert(
            2,
            {
                "section": "account",
                "label": "Processing tier",
                "key": "service_tier",
                "kind": "choice",
                "choices": tier_choices,
                "desc": settings_service_tier_description(config),
            },
        )
    return rows
