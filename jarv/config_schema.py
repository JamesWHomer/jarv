"""Single source of truth for jarv configuration keys, defaults, and validation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarv, a helpful CLI assistant. "
    "You can run shell commands when needed to answer questions or complete tasks. "
    "Be concise and direct. "
    "When several tool calls are independent, issue them in the same response instead of one tool call per turn. "
    "When the user asks about jarv commands, behavior, config, updating, or usage, "
    "run `jarv /help` before answering. Do not invent unsupported commands."
)

TOOL_NAMES = ("run_command", "web_search", "read", "spawn", "ask_user")
READ_ONLY_COMMAND_DISPLAY_CHOICES = ("fullscreen", "print")
LEGACY_READ_ONLY_COMMAND_DISPLAY_CHOICES = ("auto", "inline")
TOOL_CALL_DISPLAY_CHOICES = ("fullscreen", "print", "auto")
COMMAND_SAFETY_CHOICES = ("risky", "all", "none")


@dataclass(frozen=True)
class ConfigField:
    key: str
    default: Any
    validator: str = "any"
    choices: tuple[str, ...] = ()


CONFIG_FIELDS: tuple[ConfigField, ...] = (
    ConfigField("provider", "openai"),
    ConfigField("api_key", ""),
    ConfigField("api_keys", {}),
    ConfigField("base_url", ""),
    ConfigField("model", "gpt-5.4-mini"),
    ConfigField("service_tiers", {}, validator="service_tiers"),
    ConfigField("reasoning_effort", ""),
    ConfigField("max_history", 40, validator="positive_int"),
    ConfigField("context_budget_ratio", 0.75, validator="ratio"),
    ConfigField("context_compaction_threshold", 0.85, validator="ratio"),
    ConfigField("context_output_reserve_ratio", 0.15, validator="ratio"),
    ConfigField("context_window_fallback", 128_000, validator="positive_int"),
    ConfigField("max_stdin_chars", 200_000, validator="positive_int"),
    ConfigField("max_tool_output_chars", 20_000, validator="positive_int"),
    ConfigField("disabled_tools", [], validator="disabled_tools"),
    ConfigField("command_timeout", 60, validator="positive_int"),
    ConfigField("web_timeout", 15, validator="positive_int"),
    ConfigField(
        "command_safety",
        "risky",
        validator="choices",
        choices=COMMAND_SAFETY_CHOICES,
    ),
    ConfigField("audit", True, validator="bool"),
    ConfigField("auditor_auto_approve", True, validator="bool"),
    ConfigField("auditor_model", ""),
    ConfigField("system_prompt", DEFAULT_SYSTEM_PROMPT),
    ConfigField("max_subagent_depth", 4, validator="non_negative_int"),
    ConfigField("subagent_thread_pool_max_workers", 8, validator="positive_int"),
    ConfigField("check_updates", True, validator="bool"),
    ConfigField(
        "read_only_command_display",
        "fullscreen",
        validator="choices",
        choices=READ_ONLY_COMMAND_DISPLAY_CHOICES,
    ),
    ConfigField(
        "tool_call_display",
        "auto",
        validator="choices",
        choices=TOOL_CALL_DISPLAY_CHOICES,
    ),
    ConfigField("print_usage_after_agent", False, validator="bool"),
)

CONFIG_FIELD_BY_KEY = {field.key: field for field in CONFIG_FIELDS}


def build_default_config() -> dict:
    return {field.key: copy.deepcopy(field.default) for field in CONFIG_FIELDS}


def validate_config_fields(
    config: dict,
    *,
    report: Callable[[str], None],
) -> bool:
    """Validate and coerce values described by CONFIG_FIELDS."""
    ok = True
    for field in CONFIG_FIELDS:
        key = field.key
        if field.validator == "positive_int":
            try:
                value = int(config.get(key, field.default))
                if value <= 0:
                    raise ValueError
                config[key] = value
            except (TypeError, ValueError):
                report(f"[red]Config '{key}' must be a positive integer.[/red]")
                ok = False
        elif field.validator == "non_negative_int":
            try:
                value = int(config.get(key, field.default))
                if value < 0:
                    raise ValueError
                config[key] = value
            except (TypeError, ValueError):
                report(f"[red]Config '{key}' must be a non-negative integer.[/red]")
                ok = False
        elif field.validator == "ratio":
            try:
                value = float(config.get(key, field.default))
                if not (0.0 < value < 1.0):
                    raise ValueError
                config[key] = value
            except (TypeError, ValueError):
                report(f"[red]Config '{key}' must be a number between 0 and 1.[/red]")
                ok = False
        elif field.validator == "choices":
            value = config.get(key, field.default)
            if value not in field.choices:
                choices = ", ".join(field.choices)
                report(f"[red]Config '{key}' must be one of: {choices}.[/red]")
                ok = False
        elif field.validator == "disabled_tools":
            disabled_tools = config.get(key, field.default)
            if not isinstance(disabled_tools, list):
                report("[red]Config 'disabled_tools' must be a list.[/red]")
                ok = False
            else:
                invalid_tools = [
                    name
                    for name in disabled_tools
                    if not isinstance(name, str) or name not in TOOL_NAMES
                ]
                if invalid_tools:
                    choices = ", ".join(TOOL_NAMES)
                    report(
                        "[red]Config 'disabled_tools' contains unknown tools. "
                        f"Available tools: {choices}.[/red]"
                    )
                    ok = False
                elif len(set(disabled_tools)) != len(disabled_tools):
                    config[key] = list(dict.fromkeys(disabled_tools))
        elif field.validator == "service_tiers":
            service_tiers = config.get(key, field.default)
            if not isinstance(service_tiers, dict):
                report("[red]Config 'service_tiers' must be an object.[/red]")
                ok = False
    return ok
