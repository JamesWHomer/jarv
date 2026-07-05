"""Single source of truth for jarv configuration keys, defaults, validation, and UI metadata."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarv, a helpful CLI assistant. "
    "You can run shell commands when needed to answer questions or complete tasks. "
    "Be concise and direct. "
    "When several tool calls are independent, issue them in the same response instead of one tool call per turn. "
    "When the user asks about jarv commands, behavior, config, updating, or usage, "
    "run `jarv /help` before answering. Do not invent unsupported commands."
)

TOOL_NAMES = ("run_command", "web_search", "read", "edit", "spawn", "ask_user")
READ_ONLY_COMMAND_DISPLAY_CHOICES = ("fullscreen", "print")
LEGACY_READ_ONLY_COMMAND_DISPLAY_CHOICES = ("auto", "inline")
TOOL_CALL_DISPLAY_CHOICES = ("fullscreen", "print", "auto")
COMMAND_SAFETY_CHOICES = ("risky", "all", "none")

SETTINGS_SAFETY_CHOICES = (
    ("risky", "flag risky"),
    ("all", "confirm all"),
    ("none", "no prompts"),
)

SETTINGS_TOOL_LABELS = {
    "run_command": ("Run commands", "execute shell commands"),
    "web_search": ("Web search", "search the web"),
    "read": ("Read", "read files, URLs, artifacts, and retained output"),
    "edit": ("Edit files", "make exact text replacements in files"),
    "spawn": ("Subagents", "fan out work to parallel subagents"),
    "ask_user": ("Ask user", "pause to request clarification"),
}


@dataclass(frozen=True)
class ConfigField:
    key: str
    default: Any
    validator: str = "any"
    choices: tuple[str, ...] = ()
    label: str = ""
    section: str = ""
    desc: str = ""
    ui_kind: str = ""
    about: str = ""
    empty: str = ""
    multiline: bool = False
    settings_choices: tuple[tuple[str, str], ...] = field(default_factory=tuple)


CONFIG_FIELDS: tuple[ConfigField, ...] = (
    ConfigField("provider", "openai", label="Provider", section="account", desc="choose an API provider", ui_kind="setup", about="API provider. Options: openai, openrouter, anthropic, gemini, groq, deepseek, together, fireworks, ollama, lm_studio, vllm."),
    ConfigField("api_key", "", label="API key", section="account", desc="store or replace the active provider key", ui_kind="setup", about="API key. Can also be provided via provider-specific env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)."),
    ConfigField("api_keys", {}),
    ConfigField("base_url", "", label="Base URL", section="account", desc="optional custom endpoint", ui_kind="text", empty="provider default", about="Custom API base URL. Overrides the provider's default endpoint."),
    ConfigField("model", "gpt-5.4-mini", label="Model", section="behaviour", desc="pick from the provider presets or enter a model", ui_kind="setup", about="Model name."),
    ConfigField("service_tiers", {}, validator="service_tiers"),
    ConfigField("reasoning_effort", "", label="Reasoning effort", section="behaviour", desc="model-supported reasoning effort", ui_kind="choice", about="Model-supported reasoning effort. Empty uses the provider/model default; `none` explicitly disables reasoning only where supported."),
    ConfigField("max_history", 40, validator="positive_int", label="History limit", section="runtime", desc="recent stored items sent as context", ui_kind="int", about="Maximum stored history items included as model context (item cap before token trimming). It does not delete saved history."),
    ConfigField("context_budget_ratio", 0.75, validator="ratio", about="Share of the context window used for input."),
    ConfigField("context_compaction_threshold", 0.85, validator="ratio", about="Fill ratio that triggers history compaction."),
    ConfigField("context_output_reserve_ratio", 0.15, validator="ratio", about="Context window share reserved for model output."),
    ConfigField("context_window_fallback", 128_000, validator="positive_int", about="Context window when model metadata is unknown."),
    ConfigField("max_stdin_chars", 200_000, validator="positive_int", label="Stdin limit", section="runtime", desc="piped stdin chars attached to one-shot prompts", ui_kind="int", about="Maximum piped stdin characters attached to a one-shot prompt."),
    ConfigField("max_tool_output_chars", 20_000, validator="positive_int", label="Tool output limit", section="runtime", desc="tool output chars returned to the model", ui_kind="int", about="Maximum generic tool output characters returned to the model and the default combined head/tail budget for `run_command`."),
    ConfigField("disabled_tools", [], validator="disabled_tools", about="Tool names omitted from root agents and subagents. Use `/settings` to toggle `run_command`, `web_search`, `read`, `edit`, `spawn`, and `ask_user`."),
    ConfigField("command_timeout", 60, validator="positive_int", label="Command timeout", section="runtime", desc="kill non-interactive commands; check in during interactive commands", ui_kind="int", about="Seconds before a non-interactive shell command is killed, or before an interactive command asks the model what to do next."),
    ConfigField("web_timeout", 15, validator="positive_int", label="Web timeout", section="runtime", desc="seconds before web requests are cancelled", ui_kind="int", about="Seconds before a web search or URL read is killed."),
    ConfigField(
        "command_safety",
        "risky",
        validator="choices",
        choices=COMMAND_SAFETY_CHOICES,
        label="Command approval",
        section="command review",
        desc="default: flag only risky commands",
        ui_kind="choice",
        settings_choices=SETTINGS_SAFETY_CHOICES,
        about="Command confirmation level. `all` = confirm every command, `risky` = confirm only dangerous commands, `none` = no confirmation.",
    ),
    ConfigField("audit", True, validator="bool", label="Auditor", section="command review", desc="LLM reviews flagged commands first", ui_kind="bool", about="When `true`, flagged commands are sent to a fast LLM auditor (uses extra tokens). Applies to `run_command` only; flagged `edit` calls show a diff for manual approval."),
    ConfigField("auditor_auto_approve", True, validator="bool", label="Audit auto-accept", section="command review", desc="auto-run commands the auditor marks safe", ui_kind="bool", about="When `true`, the auditor auto-approves commands it deems safe. When `false`, the auditor only shows a recommendation."),
    ConfigField("auditor_model", "", label="Auditor model", section="command review", desc="use the active model unless overridden", ui_kind="text", empty="default", about="Model used for the auditor. Empty = use the active model."),
    ConfigField("system_prompt", DEFAULT_SYSTEM_PROMPT, label="System prompt", section="behaviour", desc="instructions sent before each request", ui_kind="text", multiline=True, about="Instructions sent to the model before each request."),
    ConfigField("max_subagent_depth", 4, validator="non_negative_int", label="Max depth", section="subagents", desc="maximum nested spawn depth", ui_kind="int", about="Maximum recursion depth for `spawn` (root is 0)."),
    ConfigField("subagent_thread_pool_max_workers", 8, validator="positive_int", label="Parallel workers", section="subagents", desc="subagents per spawn batch", ui_kind="int", about="Max parallel children in one `spawn` batch."),
    ConfigField("check_updates", True, validator="bool", label="Update checks", section="updates", desc="background check on one-shot runs", ui_kind="bool", about="When `true`, a one-shot `jarv <question>` run fires a non-blocking background update check."),
    ConfigField(
        "read_only_command_display",
        "fullscreen",
        validator="choices",
        choices=READ_ONLY_COMMAND_DISPLAY_CHOICES,
        label="Read-only commands",
        section="display",
        desc="fullscreen temporary view or permanent print output",
        ui_kind="choice",
        settings_choices=tuple((value, value) for value in READ_ONLY_COMMAND_DISPLAY_CHOICES),
        about="How `/help`, `/about`, `/usage`, and `/config` are displayed in an interactive terminal.",
    ),
    ConfigField(
        "tool_call_display",
        "auto",
        validator="choices",
        choices=TOOL_CALL_DISPLAY_CHOICES,
        label="Tool calls",
        section="display",
        desc="resize-safe print layout or bordered fullscreen cards",
        ui_kind="choice",
        settings_choices=tuple((value, value) for value in TOOL_CALL_DISPLAY_CHOICES),
        about="How agent tool calls are rendered.",
    ),
    ConfigField("print_usage_after_agent", False, validator="bool", label="Print usage", section="display", desc="print token totals after completed agent runs", ui_kind="bool", about="When `true`, print a compact token usage line after each completed agent run."),
)

CONFIG_FIELD_BY_KEY = {field.key: field for field in CONFIG_FIELDS}


def build_default_config() -> dict:
    return {field.key: copy.deepcopy(field.default) for field in CONFIG_FIELDS}


_MISSING = object()


def setting_default(key: str) -> Any:
    """Return the schema default for ``key``.

    Mutable defaults are deep-copied so callers can safely store or mutate the
    result without aliasing the shared schema value.
    """
    field = CONFIG_FIELD_BY_KEY.get(key)
    if field is None:
        raise KeyError(f"unknown config key: {key!r}")
    return copy.deepcopy(field.default)


def get_setting(config: dict, key: str, default: Any = _MISSING) -> Any:
    """Read ``config[key]``, falling back to the schema default when absent.

    Single source of truth for the ``config.get(key, DEFAULT_CONFIG[key])`` idiom:
    when ``key`` is present its stored value is returned verbatim (including empty
    or ``None`` values), otherwise the schema default for ``key`` is used. Pass an
    explicit ``default`` for keys outside :data:`CONFIG_FIELDS`.
    """
    if key in config:
        return config[key]
    if default is not _MISSING:
        return default
    return setting_default(key)


def config_about_lines(config: dict | None = None) -> list[str]:
    """Return /about documentation lines for documented config keys."""
    config = config or build_default_config()
    lines: list[str] = []
    for field in CONFIG_FIELDS:
        if not field.about:
            continue
        default = config.get(field.key, field.default)
        if field.key == "model":
            default_repr = f"`{default}`"
        elif isinstance(default, str) and default:
            default_repr = f"`{default}`"
        elif isinstance(default, bool):
            default_repr = "`true`" if default else "`false`"
        else:
            default_repr = f"`{default!r}`"
        line = f"- `{field.key}` - {field.about}"
        if field.key not in ("api_key", "api_keys", "system_prompt"):
            line += f" Default: {default_repr}."
        lines.append(line)
    lines.append(
        "- `service_tiers` - Per-provider processing tier. Values are `standard`, `flex`, or `priority`; missing providers use `standard`."
    )
    return lines


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
