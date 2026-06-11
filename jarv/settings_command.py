"""Interactive settings command."""

import sys
import threading
from collections.abc import Callable

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich import box

from .command_input import _read_key_with_repeats, mouse_capture
from .config import (
    CONFIG_FILE,
    DEFAULT_CONFIG,
    READ_ONLY_COMMAND_DISPLAY_CHOICES,
    load_config,
    save_config,
)
from .display import console, jarv_panel, refresh_on_resize, section_rule, terminal_size
from .text_editor import (
    apply_text_editor_key,
    initialize_text_editor,
    render_single_line,
    render_visual_lines,
)


class _ModelCatalogRefresher:
    """Deduplicate delayed catalog refreshes and keep network work off the UI thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._timers: dict[str, threading.Timer] = {}
        self._inflight: set[str] = set()
        self._latest: dict[str, tuple[int, dict, Callable]] = {}
        self._closed = False

    def request(
        self,
        config: dict,
        callback: Callable[[str, list[tuple[str, str]], int], None],
        *,
        delay: float = 0,
    ) -> int:
        from .model_catalog import catalog_cache_key

        snapshot = dict(config)
        key = catalog_cache_key(snapshot)
        with self._lock:
            if self._closed:
                return self._generation
            self._generation += 1
            generation = self._generation
            self._latest[key] = (generation, snapshot, callback)
            timer = self._timers.pop(key, None)
            if timer is not None:
                timer.cancel()
            if key in self._inflight:
                return generation
            timer = threading.Timer(delay, self._launch, args=(key,))
            timer.daemon = True
            self._timers[key] = timer
            timer.start()
        return generation

    def _launch(self, key: str) -> None:
        with self._lock:
            self._timers.pop(key, None)
            pending = self._latest.get(key)
            if pending is None or key in self._inflight:
                return
            self._inflight.add(key)
            _generation, snapshot, _callback = pending
        threading.Thread(
            target=self._refresh,
            args=(key, snapshot),
            daemon=True,
            name="jarv-model-catalog",
        ).start()

    def _refresh(self, key: str, snapshot: dict) -> None:
        from .model_catalog import get_cached_model_choices, refresh_model_choices

        try:
            choices = refresh_model_choices(snapshot)
        except Exception:
            choices = get_cached_model_choices(snapshot)
        with self._lock:
            self._inflight.discard(key)
            pending = self._latest.pop(key, None)
            closed = self._closed
        if pending is None or closed:
            return
        generation, _latest_snapshot, callback = pending
        callback(str(snapshot.get("provider", "openai")), choices, generation)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            timers = list(self._timers.values())
            self._timers.clear()
            self._latest.clear()
        for timer in timers:
            timer.cancel()

    def cancel_pending(self) -> None:
        with self._lock:
            timers = list(self._timers.values())
            self._timers.clear()
            for key in list(self._latest):
                if key not in self._inflight:
                    self._latest.pop(key, None)
        for timer in timers:
            timer.cancel()


_SETTINGS_SAFETY_CHOICES = (
    ("risky", "flag risky"),
    ("all", "confirm all"),
    ("none", "no prompts"),
)

_SETTINGS_REASONING_CHOICES = (
    ("", "off"),
    ("minimal", "minimal"),
    ("low", "low"),
    ("medium", "medium"),
    ("high", "high"),
    ("xhigh", "xhigh"),
)

_SETTINGS_READ_ONLY_DISPLAY_CHOICES = (
    ("fullscreen", "fullscreen"),
    ("print", "print"),
)


def _settings_service_tier_choices(config: dict) -> tuple[tuple[str, str], ...]:
    from .provider_catalog import service_tier_choices

    provider = str(config.get("provider", "openai"))
    return tuple((tier, tier) for tier in service_tier_choices(provider))


def _settings_service_tier(config: dict) -> str:
    from .provider_catalog import configured_service_tier

    return configured_service_tier(config)


def _settings_service_tier_description(config: dict) -> str:
    provider = str(config.get("provider", "openai"))
    if provider == "anthropic":
        return "priority uses committed capacity, then falls back to standard"
    if len(_settings_service_tier_choices(config)) == 1:
        return "this provider uses standard processing"
    return "standard cost, flex savings, or priority latency"


def _settings_set_service_tier(config: dict, tier: str) -> None:
    provider = str(config.get("provider", "openai"))
    existing = config.get("service_tiers")
    configured = dict(existing) if isinstance(existing, dict) else {}
    configured[provider] = tier
    config["service_tiers"] = configured


def _clip_text(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[: width - 1] + "\u2026"


def _settings_choice_label(value, choices: tuple[tuple[str, str], ...]) -> str:
    for key, label in choices:
        if value == key:
            return label
    return str(value)


def _settings_has_api_key(config: dict) -> tuple[bool, str]:
    from .provider import LOCAL_PROVIDERS, PROVIDERS, resolve_api_key

    provider = config.get("provider", "openai")
    if provider in LOCAL_PROVIDERS:
        return True, "not needed"
    if resolve_api_key(config):
        return True, "configured"
    label = PROVIDERS.get(provider, {}).get("label", provider)
    return False, f"missing for {label}"


def _settings_rows(config: dict) -> list[dict]:
    return [
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
            "label": "Model",
            "key": "model",
            "kind": "setup",
            "step": "model",
            "desc": "pick from the provider presets or enter a model",
        },
        {
            "section": "account",
            "label": "Processing tier",
            "key": "service_tier",
            "kind": "choice",
            "choices": _settings_service_tier_choices(config),
            "desc": _settings_service_tier_description(config),
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
            "choices": _SETTINGS_READ_ONLY_DISPLAY_CHOICES,
            "desc": "fullscreen temporary view or permanent print output",
        },
        {
            "section": "display",
            "label": "Print usage after agent",
            "key": "print_usage_after_agent",
            "kind": "bool",
            "desc": "print token totals after completed agent runs",
        },
        {
            "section": "command review",
            "label": "Command safety",
            "key": "command_safety",
            "kind": "choice",
            "choices": _SETTINGS_SAFETY_CHOICES,
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
            "empty": "auto",
            "desc": "empty uses the active model",
        },
        {
            "section": "runtime",
            "label": "Reasoning effort",
            "key": "reasoning_effort",
            "kind": "choice",
            "choices": _SETTINGS_REASONING_CHOICES,
            "desc": "optional effort hint",
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


def _settings_value_text(row: dict, config: dict, *, selected: bool = False) -> Text:
    from .provider import PROVIDERS

    key = row["key"]
    kind = row["kind"]
    value = (
        _settings_service_tier(config)
        if key == "service_tier"
        else config.get(key, DEFAULT_CONFIG.get(key, ""))
    )

    if key == "system_prompt":
        prompt = str(value)
        if not prompt:
            label = "empty"
        elif prompt == DEFAULT_CONFIG["system_prompt"]:
            label = "default"
        else:
            label = f"custom \u00b7 {len(prompt)} chars"
        return Text(label, style="bold green" if selected else "green")
    if key == "provider":
        label = PROVIDERS.get(value, {}).get("label", str(value))
        return Text(label, style="bold magenta" if selected else "magenta")
    if key == "api_key":
        ok, label = _settings_has_api_key(config)
        return Text(label, style=("bold green" if ok and selected else "green") if ok else "bold red")
    if kind == "bool":
        enabled = bool(value)
        label = "on" if enabled else "off"
        if enabled:
            return Text(label, style="bold green" if selected else "green")
        return Text(label, style="bold yellow" if selected else "yellow")
    if kind == "choice":
        label = _settings_choice_label(value, row["choices"])
        style = "bold cyan" if selected else "cyan"
        if key == "command_safety" and value == "none":
            style = "bold red" if selected else "red"
        elif key == "command_safety" and value == "all":
            style = "bold yellow" if selected else "yellow"
        return Text(label, style=style)
    if kind == "int":
        return Text(str(value), style="bold yellow" if selected else "yellow")
    if kind == "text":
        if value:
            return Text(str(value), style="bold green" if selected else "green")
        return Text(row.get("empty", "empty"), style="dim italic")
    return Text(str(value), style="bold" if selected else "")


def _settings_apply_quick(row: dict, config: dict) -> tuple[dict, str] | None:
    key = row["key"]
    kind = row["kind"]

    if kind == "bool":
        config[key] = not bool(config.get(key, DEFAULT_CONFIG.get(key, False)))
        save_config(config)
        state = "on" if config[key] else "off"
        return config, f"saved {row['label']}: {state}"

    if kind == "choice":
        choices = row["choices"]
        current = (
            _settings_service_tier(config)
            if key == "service_tier"
            else config.get(key, DEFAULT_CONFIG.get(key, choices[0][0]))
        )
        idx = next((i for i, (value, _) in enumerate(choices) if value == current), -1)
        value = choices[(idx + 1) % len(choices)][0]
        if key == "service_tier":
            _settings_set_service_tier(config, value)
        else:
            config[key] = value
        save_config(config)
        return config, f"saved {row['label']}: {_settings_choice_label(value, choices)}"

    return None


def _settings_reset_value(row: dict, config: dict):
    if row["key"] == "model":
        return _settings_default_model(config)
    if row["key"] == "service_tier":
        return "standard"
    return DEFAULT_CONFIG[row["key"]]


def _settings_reset_row(row: dict, config: dict) -> tuple[dict, str]:
    key = row["key"]
    if key == "api_key":
        provider = config.get("provider", "openai")
        changed = False
        api_keys = config.get("api_keys")
        if isinstance(api_keys, dict) and provider in api_keys:
            api_keys.pop(provider, None)
            changed = True
        if config.get("api_key"):
            config["api_key"] = ""
            changed = True
        if changed:
            save_config(config)
            return config, "cleared stored API key"
        return config, "no stored API key"
    if key == "service_tier":
        _settings_set_service_tier(config, "standard")
        save_config(config)
        return config, "reset Processing tier"
    if key not in DEFAULT_CONFIG:
        return config, f"{row['label']} has no default"
    config[key] = _settings_reset_value(row, config)
    save_config(config)
    return config, f"reset {row['label']}"


def _settings_reset_action_bar(
    row: dict,
    config: dict,
    inner_width: int,
) -> Text:
    key = row["key"]
    if key == "api_key":
        prompt = "Clear stored API key?"
        controls = "y clear   Esc cancel"
        current = default = ""
        left = prompt
    else:
        current = _settings_value_text(row, config).plain
        default_config = dict(config)
        if key == "service_tier":
            _settings_set_service_tier(default_config, "standard")
        else:
            default_config[key] = _settings_reset_value(row, config)
        default = _settings_value_text(row, default_config).plain
        prompt = f"Reset {row['label']}?"
        controls = "y reset   Esc cancel"
        left = f"{prompt}   {current} \u2192 {default}"

    gap = max(3, inner_width - len(left) - len(controls))
    clipped = False
    if len(left) + gap + len(controls) > inner_width:
        available = max(1, inner_width - len(controls) - 3)
        left = _clip_text(left, available)
        gap = max(1, inner_width - len(left) - len(controls))
        clipped = True

    line = Text(no_wrap=True, overflow="crop")
    if clipped or key == "api_key":
        line.append(left, style="bold yellow")
    else:
        line.append(prompt, style="bold yellow")
        line.append("   ")
        line.append(current, style="green")
        line.append(" \u2192 ", style="dim")
        line.append(default, style="cyan")
    line.append(" " * gap)
    line.append(controls.split("   ", 1)[0], style="bold red")
    line.append("   ")
    line.append(controls.split("   ", 1)[1], style="dim")
    return line


def _settings_finish_reset(
    row: dict,
    config: dict,
    key: str,
) -> tuple[dict, str, str]:
    if key in ("y", "Y"):
        config, message = _settings_reset_row(row, config)
        return config, message, "cyan"
    return config, f"{row['label']} reset cancelled", "dim"


def _settings_provider_choices() -> list[tuple[str, str, str]]:
    from .provider_catalog import PROVIDER_CHOICES

    return list(PROVIDER_CHOICES)


def _settings_model_choices(
    config: dict,
    *,
    refresh: bool = False,
) -> list[tuple[str, str]]:
    from .model_catalog import get_cached_model_choices, refresh_model_choices

    if refresh:
        return refresh_model_choices(config)
    return get_cached_model_choices(config)


def _settings_default_model(config: dict) -> str:
    return _settings_default_model_for_provider(
        config.get("provider", "openai"),
        config=config,
    )


def _settings_default_model_for_provider(
    provider: str,
    *,
    config: dict | None = None,
) -> str:
    from .model_catalog import get_default_model

    probe = dict(config or {})
    probe["provider"] = provider
    choices = _settings_model_choices(probe)
    if choices:
        return get_default_model(probe, choices=choices)
    for key, _label, model in _settings_provider_choices():
        if key == provider:
            return model
    return DEFAULT_CONFIG["model"]


def _settings_provider_keys() -> list[str]:
    return [key for key, _label, _model in _settings_provider_choices()]


def _settings_resolve_provider(choice: str) -> str | None:
    raw = choice.strip()
    if not raw:
        return None
    try:
        idx = int(raw)
        choices = _settings_provider_choices()
        if 1 <= idx <= len(choices):
            return choices[idx - 1][0]
        return None
    except ValueError:
        pass

    lowered = raw.lower()
    for key, label, _model in _settings_provider_choices():
        compact_label = label.split("(", 1)[0].strip().lower()
        if lowered in (key.lower(), label.lower(), compact_label):
            return key
    return None


def _settings_resolve_model(
    config: dict,
    choice: str,
    *,
    models: list[tuple[str, str]] | None = None,
) -> str:
    raw = choice.strip()
    if not raw:
        return str(config.get("model") or _settings_default_model(config))
    if models is None:
        models = _settings_model_choices(config)
    try:
        idx = int(raw)
        if models and 1 <= idx <= len(models):
            return models[idx - 1][0]
    except ValueError:
        pass
    for name, _desc in models:
        if raw.lower() == name.lower():
            return name
    return raw


def _settings_column_layout(inner_width: int) -> tuple[int, int, int, int]:
    """Return label/value widths and value/description start columns."""
    prefix_width = 3
    label_width = min(22, max(10, inner_width // 4))
    value_width = min(28, max(10, inner_width // 3))
    value_start = prefix_width + label_width + 2
    description_start = value_start + value_width + 2
    return label_width, value_width, value_start, description_start


def _settings_choice_grid_lines(
    items: list[tuple[int, str, str]],
    inner_width: int,
    *,
    max_lines: int | None = None,
    max_columns: int = 1,
    more_hint: str = "type a number/name",
    align_descriptions: bool = False,
) -> list[Text]:
    if not items:
        return []
    if max_lines is not None and max_lines <= 0:
        return []

    indent = "  "
    gap = "  "
    usable_width = max(1, inner_width - len(indent))
    if max_columns >= 3 and usable_width >= 96:
        columns = 3
    elif max_columns >= 2 and usable_width >= 58:
        columns = 2
    else:
        columns = 1
    columns = max(1, min(columns, max_columns, len(items)))
    row_count = (len(items) + columns - 1) // columns
    cell_width = max(8, (usable_width - (columns - 1) * len(gap)) // columns)
    single_column_primary_width = min(
        max(len(primary) for _idx, primary, _secondary in items),
        max(1, cell_width // 2),
    )
    _label_width, _value_width, _value_start, description_start = (
        _settings_column_layout(inner_width)
    )

    def _cell(idx: int, primary: str, secondary: str) -> str:
        marker = f"{idx:>2}. "
        body_width = max(1, cell_width - len(marker))
        if not secondary or body_width < 18:
            return marker + _clip_text(primary or secondary, body_width)
        if columns == 1:
            if align_descriptions:
                secondary_column = max(
                    len(indent) + len(marker) + 3,
                    description_start,
                )
                primary_width = max(
                    1,
                    secondary_column - len(indent) - len(marker) - 2,
                )
                remaining = max(0, inner_width - secondary_column)
                cell = marker + f"{_clip_text(primary, primary_width):<{primary_width}}"
                if remaining:
                    cell += "  " + _clip_text(secondary, remaining)
                return cell
            primary_width = min(single_column_primary_width, body_width)
            remaining = max(0, body_width - primary_width - 2)
            cell = marker + f"{_clip_text(primary, primary_width):<{primary_width}}"
            if remaining:
                cell += "  " + _clip_text(secondary, remaining)
            return cell

        secondary_width = min(18, max(8, body_width // 3))
        primary_width = max(1, body_width - secondary_width - 1)
        if primary_width < 10:
            return marker + _clip_text(primary, body_width)
        return (
            marker
            + f"{_clip_text(primary, primary_width):<{primary_width}}"
            + " "
            + _clip_text(secondary, secondary_width)
        )

    rows: list[list[str]] = []
    for row_idx in range(row_count):
        cells: list[str] = []
        for col_idx in range(columns):
            item_idx = row_idx + col_idx * row_count
            if item_idx < len(items):
                cells.append(_cell(*items[item_idx]))
        rows.append(cells)

    hidden = 0
    if max_lines is not None and len(rows) > max_lines:
        visible_rows = max(0, max_lines - 1)
        hidden = sum(len(row) for row in rows[visible_rows:])
        rows = rows[:visible_rows]

    lines = [
        Text(_clip_text(indent + gap.join(cells), inner_width), style="dim", no_wrap=True, overflow="crop")
        for cells in rows
    ]
    if hidden:
        label = "option" if hidden == 1 else "options"
        lines.append(Text(_clip_text(f"  ... {hidden} more {label}; {more_hint}", inner_width), style="dim"))
    return lines


def _settings_provider_display_label(label: str) -> str:
    return label.split("(", 1)[0].strip()


def _settings_provider_note(provider_key: str) -> str:
    from .provider import LOCAL_PROVIDERS, PROVIDERS

    notes = {
        "openai": "OpenAI-hosted models through the Responses API",
        "openrouter": "one API key for a marketplace of third-party models",
        "anthropic": "Anthropic-hosted Claude models through the Messages API",
        "gemini": "Google AI Studio Gemini models through the native API",
        "groq": "Groq-hosted OpenAI-compatible chat endpoint",
        "deepseek": "DeepSeek-hosted OpenAI-compatible chat endpoint",
        "together": "Together-hosted catalog of open and partner models",
        "fireworks": "Fireworks-hosted catalog of open and partner models",
        "ollama": "local Ollama server running models on this machine",
        "lm_studio": "local LM Studio server using an OpenAI-compatible API",
        "vllm": "self-hosted vLLM server using an OpenAI-compatible API",
    }
    if provider_key in notes:
        return notes[provider_key]
    if provider_key in LOCAL_PROVIDERS:
        base_url = PROVIDERS.get(provider_key, {}).get("base_url")
        return base_url or "no API key"
    return "hosted API provider"


def _settings_provider_choice_lines(
    config: dict,
    inner_width: int,
    *,
    selected_provider: str | None = None,
    max_lines: int | None = None,
) -> list[Text]:
    from .provider import LOCAL_PROVIDERS

    if max_lines is not None and max_lines <= 0:
        return []

    providers = [
        (idx, provider_key, label, default_model)
        for idx, (provider_key, label, default_model) in enumerate(_settings_provider_choices(), 1)
    ]
    sections = [
        ("Cloud providers", [item for item in providers if item[1] not in LOCAL_PROVIDERS]),
        ("Local runtimes", [item for item in providers if item[1] in LOCAL_PROVIDERS]),
    ]

    number_width = 2
    _settings_label_width, settings_value_width, value_start, description_start = (
        _settings_column_layout(inner_width)
    )
    prefix_width = 3
    number_gap = 2
    label_width = max(
        1,
        value_start - prefix_width - number_width - number_gap - 2,
    )
    key_width = settings_value_width
    note_width = max(0, inner_width - description_start)

    entries: list[tuple[str, tuple[int, str, str, str] | None]] = []
    for title, items in sections:
        if not items:
            continue
        if entries:
            entries.append(("blank", None))
        entries.append(("section", (0, title, "", "")))
        for idx, provider_key, label, default_model in items:
            entries.append(
                (
                    "provider",
                    (
                        idx,
                        provider_key,
                        _settings_provider_display_label(label),
                        _settings_provider_note(provider_key),
                    ),
                )
            )

    hidden = 0
    if max_lines is not None and len(entries) > max_lines:
        visible_count = max(0, max_lines - 1)
        selected_idx = next(
            (
                idx
                for idx, (kind, entry) in enumerate(entries)
                if kind == "provider" and entry is not None and entry[1] == selected_provider
            ),
            0,
        )
        start_idx = min(max(0, selected_idx - visible_count // 2), max(0, len(entries) - visible_count))
        visible_entries = entries[start_idx:start_idx + visible_count]
        hidden = sum(1 for kind, _entry in entries[:start_idx] + entries[start_idx + visible_count:] if kind == "provider")
        entries = visible_entries

    lines: list[Text] = []
    current_provider = config.get("provider", "openai")
    selected_provider = selected_provider or current_provider
    for kind, entry in entries:
        if kind == "blank":
            lines.append(Text(""))
            continue
        if kind == "section" and entry is not None:
            _idx, title, _label, _note = entry
            lines.append(Text(_clip_text(f"  {title}", inner_width), style="bold cyan"))
            continue
        if kind != "provider" or entry is None:
            continue

        idx, provider_key, label, note = entry
        is_current = provider_key == current_provider
        is_selected = provider_key == selected_provider
        marker = ">" if is_selected else " "
        selected_style = "bold bright_white"
        current_style = "bold green"
        label_style = selected_style if is_selected else current_style if is_current else "white"
        detail_style = selected_style if is_selected else "dim"
        line = Text(no_wrap=True, overflow="crop")
        line.append(f" {marker} ", style="bold cyan" if is_selected else "dim")
        line.append(f"{idx:>{number_width}}  ", style="bold cyan" if is_selected else "cyan")
        line.append(f"{_clip_text(label, label_width):<{label_width}}", style=label_style)
        line.append("  ", style="dim")
        line.append(f"{_clip_text(provider_key, key_width):<{key_width}}", style=detail_style)
        if note and note_width > 0:
            line.append("  ", style="dim")
            line.append(_clip_text(note, note_width), style=detail_style)
        lines.append(line)

    if hidden:
        label = "provider" if hidden == 1 else "providers"
        lines.append(Text(_clip_text(f"  ... {hidden} more {label}; use Up/Down", inner_width), style="dim"))
    return lines


def _settings_begin_edit(row: dict, config: dict) -> dict:
    key = row["key"]
    buffer = ""
    if row["kind"] in ("int", "text"):
        buffer = str(config.get(key, DEFAULT_CONFIG.get(key, "")))

    if key == "api_key":
        from .provider import LOCAL_PROVIDERS

        provider = config.get("provider", "openai")
        readonly = provider in LOCAL_PROVIDERS
        edit = {"row": row, "secret": True, "readonly": readonly, "error": ""}
        initialize_text_editor(edit, "")
        return edit

    edit = {
        "row": row,
        "buffer": buffer,
        "secret": False,
        "readonly": False,
        "error": "",
    }
    if row["kind"] in ("int", "text"):
        initialize_text_editor(edit, buffer, multiline=bool(row.get("multiline")))
        edit["discard_armed"] = False
    if key == "provider":
        edit["selected_provider"] = config.get("provider", "openai")
    elif key == "model":
        edit["model_choices"] = _settings_model_choices(config)
        edit["catalog_provider"] = config.get("provider", "openai")
        edit["catalog_status"] = "loading"
    return edit


def _settings_multiline_status(edit: dict) -> str:
    value = edit["buffer"]
    original = edit.get("original", value)
    if not value:
        state = "empty"
    elif value == DEFAULT_CONFIG["system_prompt"]:
        state = "default"
    else:
        state = "custom"
    modified = " \u00b7 modified" if value != original else ""
    line_count = value.count("\n") + 1
    line_label = "line" if line_count == 1 else "lines"
    return f"{state} \u00b7 {line_count} {line_label} \u00b7 {len(value)} chars{modified}"


def _settings_multiline_visual_lines(
    edit: dict,
    inner_width: int,
) -> tuple[list[Text], int]:
    return render_visual_lines(
        edit,
        max(1, inner_width - 4),
        indent="  ",
    )


def _settings_multiline_editor_lines(
    edit: dict,
    inner_width: int,
    *,
    max_lines: int | None = None,
) -> list[Text]:
    intro = [
        Text(_clip_text(f"  {_settings_multiline_status(edit)}", inner_width), style="dim"),
        Text(""),
    ]
    body, cursor_idx = _settings_multiline_visual_lines(edit, inner_width)
    tail: list[Text] = []
    if edit.get("discard_armed"):
        tail.append(
            Text(
                _clip_text("  Unsaved changes. Esc again to discard.", inner_width),
                style="bold yellow",
            )
        )
    tail.append(
        Text(
            _clip_text(
                "  Enter newline   Ctrl+S save   Esc cancel   Arrows move",
                inner_width,
            ),
            style="dim italic",
        )
    )

    if max_lines is None:
        return intro + body + tail

    body_budget = max(1, max_lines - len(intro) - len(tail))
    start = max(0, min(cursor_idx - body_budget + 1, len(body) - body_budget))
    visible_body = body[start : start + body_budget]
    lines = intro + visible_body
    while len(lines) < max_lines - len(tail):
        lines.append(Text(""))
    return (lines + tail)[:max_lines]


def _settings_multiline_apply_key(
    edit: dict,
    key: str,
    repeat_count: int = 1,
    *,
    inner_width: int = 80,
) -> bool:
    changed = apply_text_editor_key(
        edit,
        key,
        repeat_count,
        content_width=max(1, inner_width - 4),
        allow_newlines=True,
    )
    if changed:
        edit["discard_armed"] = False
        edit["error"] = ""
    return changed


def _settings_editor_lines(
    edit: dict | None,
    config: dict,
    inner_width: int,
    *,
    max_lines: int | None = None,
) -> list[Text]:
    if edit is None:
        return []

    from .provider import PROVIDERS

    row = edit["row"]
    key = row["key"]
    if row.get("multiline"):
        return _settings_multiline_editor_lines(
            edit,
            inner_width,
            max_lines=max_lines,
        )
    provider = config.get("provider", "openai")
    provider_label = PROVIDERS.get(provider, {}).get("label", provider)
    intro: list[Text] = []
    choices: list[Text] = []
    tail: list[Text] = []

    if key == "provider":
        intro.append(Text(_clip_text("  Up/Down choose", inner_width), style="dim"))
        choice_items = []
        prompt = ""
    elif key == "model":
        models = edit.get("model_choices")
        if not isinstance(models, list):
            models = _settings_model_choices(config)
        catalog_status = edit.get("catalog_status")
        if models:
            intro.append(Text(_clip_text(f"  provider: {provider_label}", inner_width), style="dim"))
            if catalog_status == "loading":
                intro.append(Text(_clip_text("  Checking for newer models...", inner_width), style="dim italic"))
            elif catalog_status == "updated":
                intro.append(Text(_clip_text("  Model list updated.", inner_width), style="green"))
            elif catalog_status == "current":
                intro.append(Text(_clip_text("  Model list is current.", inner_width), style="dim"))
            elif catalog_status == "deferred":
                intro.append(Text(
                    _clip_text("  New models will appear next time this editor opens.", inner_width),
                    style="dim",
                ))
            choice_items = [(idx, name, desc) for idx, (name, desc) in enumerate(models, 1)]
            prompt = "Model number/name/custom"
        else:
            intro.append(Text(_clip_text(f"  default for {provider_label}: {_settings_default_model(config)}", inner_width), style="dim"))
            choice_items = []
            prompt = "Model name"
    elif key == "api_key":
        if edit.get("readonly"):
            lines = [
                Text(_clip_text(f"  {provider_label} does not need an API key.", inner_width), style="green"),
                Text(_clip_text("  Esc closes this editor.", inner_width), style="dim italic"),
            ]
            return lines[:max_lines] if max_lines is not None else lines
        env_key = PROVIDERS.get(provider, {}).get("env_key") or "provider env var"
        intro.append(Text(_clip_text(f"  provider: {provider_label}   env: {env_key}", inner_width), style="dim"))
        intro.append(Text(_clip_text("  type a new key, or type clear to remove the stored key", inner_width), style="dim"))
        choice_items = []
        prompt = "API key"
    elif row["kind"] == "int":
        choice_items = []
        prompt = "Value"
    else:
        empty_label = row.get("empty", "empty")
        intro.append(Text(_clip_text(f"  type clear for {empty_label}", inner_width), style="dim"))
        choice_items = []
        prompt = row["label"]

    if key == "provider":
        pass
    else:
        line = Text(no_wrap=True, overflow="crop")
        line.append(_clip_text(f"  {prompt}: ", inner_width), style="bold")
        remaining = max(1, inner_width - len(line.plain))
        masked = bool(
            edit.get("secret")
            and str(edit["buffer"]).lower() != "clear"
        )
        line.append_text(
            render_single_line(
                edit,
                remaining,
                masked=masked,
            )
        )
        tail.append(line)

    if edit.get("error"):
        tail.append(Text(_clip_text(f"  {edit['error']}", inner_width), style="red"))

    fixed_count = len(intro) + len(tail)
    if max_lines is not None and fixed_count > max_lines:
        compact = intro[:1] + tail
        return compact[-max(1, max_lines):]

    choice_line_budget = None if max_lines is None else max(0, max_lines - fixed_count)
    if key == "provider":
        choices = _settings_provider_choice_lines(
            config,
            inner_width,
            selected_provider=edit.get("selected_provider"),
            max_lines=choice_line_budget,
        )
    elif key == "model" and choice_items:
        choices = _settings_choice_grid_lines(
            choice_items,
            inner_width,
            max_lines=choice_line_budget,
            max_columns=1,
            more_hint="type a number/name/custom",
            align_descriptions=True,
        )

    return intro + choices + tail


def _settings_desired_editor_height(
    edit: dict,
    config: dict,
    inner_width: int,
    terminal_height: int,
) -> int:
    content_height = len(_settings_editor_lines(edit, config, inner_width))
    minimum = 7 if edit["row"].get("multiline") else 3
    return max(minimum, min(content_height + 2, terminal_height))


def _settings_commit_edit(edit: dict, config: dict) -> tuple[dict, str, str, bool]:
    row = edit["row"]
    key = row["key"]
    raw_buffer = edit["buffer"]
    raw = raw_buffer.strip()

    if edit.get("readonly"):
        return config, f"{row['label']} unchanged", "dim", True

    if key == "provider":
        selected_provider = edit.get("selected_provider")
        if not selected_provider:
            return config, f"{row['label']} unchanged", "dim", True
        old_provider = config.get("provider", "openai")
        old_model = str(config.get("model") or "")
        old_default = _settings_default_model_for_provider(old_provider, config=config)
        provider = selected_provider
        if provider is None:
            edit["error"] = "Unknown provider. Enter a listed number or provider name."
            return config, edit["error"], "red", False
        config["provider"] = provider
        provider_models = [name for name, _desc in _settings_model_choices(config)]
        if (
            provider != old_provider
            and (
                not old_model
                or old_model == old_default
                or (provider_models and old_model not in provider_models)
            )
        ):
            config["model"] = _settings_default_model_for_provider(provider, config=config)
        save_config(config)
        message = f"saved Provider: {_settings_value_text(row, config).plain}"
        if config.get("model") != old_model:
            message += f" (model: {config.get('model')})"
        return config, message, "green", True

    if key == "api_key":
        provider = config.get("provider", "openai")
        if not raw:
            return config, "API key unchanged", "dim", True
        if raw.lower() == "clear":
            api_keys = config.get("api_keys")
            if isinstance(api_keys, dict):
                api_keys.pop(provider, None)
            config["api_key"] = ""
            save_config(config)
            return config, "cleared stored API key", "cyan", True
        config.setdefault("api_keys", {})[provider] = raw
        config["api_key"] = ""
        save_config(config)
        return config, "saved API key", "green", True

    if key == "model":
        models = edit.get("model_choices")
        model = _settings_resolve_model(
            config,
            raw,
            models=models if isinstance(models, list) else None,
        )
        if not model.strip():
            edit["error"] = "Model must not be empty."
            return config, edit["error"], "red", False
        config["model"] = model
        save_config(config)
        return config, f"saved Model: {model}", "green", True

    if key == "system_prompt":
        config[key] = raw_buffer
        save_config(config)
        return config, f"saved System prompt: {_settings_value_text(row, config).plain}", "green", True

    if row["kind"] == "int":
        try:
            value = int(raw)
            if value <= 0:
                raise ValueError
        except ValueError:
            edit["error"] = "Enter a positive integer."
            return config, edit["error"], "red", False
        config[key] = value
        save_config(config)
        return config, f"saved {row['label']}: {value}", "green", True

    value = "" if raw.lower() == "clear" else raw
    config[key] = value
    save_config(config)
    display = value if value else row.get("empty", "empty")
    return config, f"saved {row['label']}: {display}", "green", True


def _settings_plain(config: dict) -> None:
    rows = _settings_rows(config)
    parts: list = []
    current_section = None
    table = None

    for row in rows:
        if row["section"] != current_section:
            if table is not None:
                parts.extend([Text(""), table, Text("")])
            current_section = row["section"]
            parts.append(section_rule(current_section))
            table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
            table.add_column("Setting", style="bold cyan", no_wrap=True)
            table.add_column("Value", overflow="fold")
            table.add_column("Notes", style="dim", overflow="fold")
        table.add_row(row["label"], _settings_value_text(row, config), row["desc"])

    if table is not None:
        parts.extend([Text(""), table])

    console.print(jarv_panel(Group(*parts), title="settings", subtitle=str(CONFIG_FILE)))


def _settings_interactive(config: dict) -> None:
    rows = _settings_rows(config)
    selected = 0
    scroll_start = 0
    flash: tuple[str, str] | None = None
    edit: dict | None = None
    pending_reset: int | None = None
    catalog_refresher = _ModelCatalogRefresher()
    live_holder: list[Live] = []

    def _catalog_refreshed(
        provider: str,
        choices: list[tuple[str, str]],
        generation: int,
    ) -> None:
        current_edit = edit
        if (
            current_edit is not None
            and current_edit["row"]["key"] == "model"
            and current_edit.get("catalog_provider") == provider
            and current_edit.get("catalog_generation") == generation
        ):
            previous = current_edit.get("model_choices")
            if current_edit.get("catalog_input_dirty"):
                current_edit["catalog_status"] = "deferred"
            else:
                current_edit["model_choices"] = list(choices)
                current_edit["catalog_status"] = "updated" if choices != previous else "current"
        if live_holder:
            live_holder[0].refresh()

    def _request_catalog_refresh(
        provider: str,
        *,
        target_edit: dict | None = None,
        delay: float = 0,
    ) -> None:
        probe = dict(config)
        probe["provider"] = provider
        generation = catalog_refresher.request(
            probe,
            _catalog_refreshed,
            delay=delay,
        )
        if target_edit is not None:
            target_edit["catalog_generation"] = generation
            target_edit["catalog_input_dirty"] = False

    def _footer() -> str:
        return "\u2191\u2193 select   Enter edit/toggle   r reset   q exit"

    def _append_bottom_footer(parts: list, height: int, footer: Text) -> None:
        footer_rows = 2  # spacer + controls
        target_rows_before_footer = max(0, height - 2 - footer_rows)
        if len(parts) > target_rows_before_footer:
            del parts[target_rows_before_footer:]
        while len(parts) < target_rows_before_footer:
            parts.append(Text(""))
        parts.append(Text(""))
        parts.append(footer)

    def _settings_rendered_row_count(start: int, end: int) -> int:
        line_count = 0
        last_section = None
        for idx in range(start, end):
            row = rows[idx]
            if row["section"] != last_section:
                if idx != start:
                    line_count += 1
                line_count += 1
                last_section = row["section"]
            line_count += 1
        return line_count

    def _settings_window_end(start: int, max_lines: int) -> int:
        end = start
        while end < len(rows):
            candidate = end + 1
            if candidate > start + 1 and _settings_rendered_row_count(start, candidate) > max_lines:
                break
            end = candidate
            if _settings_rendered_row_count(start, end) >= max_lines:
                break
        return end

    def _settings_visible_window(max_lines: int) -> tuple[int, int]:
        nonlocal scroll_start
        if not rows:
            return 0, 0

        max_lines = max(1, max_lines)
        scroll_start = max(0, min(scroll_start, len(rows) - 1))
        if selected < scroll_start:
            scroll_start = selected

        while scroll_start < selected and _settings_rendered_row_count(scroll_start, selected + 1) > max_lines:
            scroll_start += 1

        end = _settings_window_end(scroll_start, max_lines)
        while selected >= end and scroll_start < selected:
            scroll_start += 1
            end = _settings_window_end(scroll_start, max_lines)

        if end == len(rows):
            while scroll_start > 0 and _settings_rendered_row_count(scroll_start - 1, end) <= max_lines:
                scroll_start -= 1

        return scroll_start, _settings_window_end(scroll_start, max_lines)

    def _render_settings_panel(height: int) -> Panel:
        nonlocal selected
        term_w, _ = terminal_size(console=console)
        panel_width = max(1, term_w)
        inner_width = max(1, panel_width - 4)
        height = max(3, height)
        show_footer = edit is None and height >= 8
        content_rows = max(1, height - 2)
        reserved = 1
        if flash is not None:
            reserved += 2
        if show_footer:
            reserved += 2
        body_rows = max(1, content_rows - reserved)
        start, end = _settings_visible_window(body_rows)

        parts: list = []
        parts.append(Text(_clip_text(f"  showing {start + 1}-{end} of {len(rows)}", inner_width), style="dim"))

        last_section = None
        for idx in range(start, end):
            row = rows[idx]
            if row["section"] != last_section:
                last_section = row["section"]
                if idx != start:
                    parts.append(Text(""))
                parts.append(Text(f"  {last_section}", style="bold cyan"))

            is_selected = idx == selected
            prefix = " \u203a " if is_selected else "   "
            label_width, value_width, _value_start, _description_start = (
                _settings_column_layout(inner_width)
            )
            desc_width = max(0, inner_width - len(prefix) - label_width - value_width - 4)

            line = Text(no_wrap=True, overflow="crop")
            line.append(prefix, style="bold cyan" if is_selected else "")
            label_style = "bold bright_white" if is_selected else "white"
            line.append(f"{_clip_text(row['label'], label_width):<{label_width}}", style=label_style)
            line.append("  ", style="dim")

            value = _settings_value_text(row, config, selected=is_selected)
            value_plain = _clip_text(value.plain, value_width)
            line.append(f"{value_plain:<{value_width}}", style=value.style)
            line.append("  ", style="dim")

            desc_style = "bold" if is_selected else "dim"
            line.append(_clip_text(row["desc"], desc_width), style=desc_style)
            parts.append(line)

        if flash is not None:
            msg, style = flash
            parts.append(Text(""))
            parts.append(Text(_clip_text(f"  {msg}", inner_width), style=style, no_wrap=True, overflow="crop"))

        if show_footer:
            if pending_reset is not None:
                footer = _settings_reset_action_bar(
                    rows[pending_reset],
                    config,
                    inner_width,
                )
            else:
                footer = Text(
                    _clip_text(_footer(), inner_width),
                    style="dim italic",
                    no_wrap=True,
                    overflow="crop",
                )
            _append_bottom_footer(
                parts,
                height,
                footer,
            )

        audit_state = "auditor on" if config.get("audit", True) else "auditor off"
        return Panel(
            Group(*parts),
            title="[bold bright_white]jarv \u25b8 settings[/bold bright_white]",
            title_align="left",
            subtitle=f"[dim]{audit_state}[/dim]",
            subtitle_align="right",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            width=panel_width,
            height=height,
        )

    def _render_editor_panel(height: int) -> Panel:
        term_w, _ = terminal_size(console=console)
        panel_width = max(1, term_w)
        inner_width = max(1, panel_width - 4)
        height = max(3, height)
        content_rows = max(1, height - 2)
        row = edit["row"] if edit is not None else {"label": "setting"}
        editor_parts = _settings_editor_lines(edit, config, inner_width, max_lines=content_rows)
        if not editor_parts:
            editor_parts = [Text("")]
        controls = "" if row.get("multiline") else "Enter save   Esc cancel"
        return Panel(
            Group(*editor_parts),
            title=f"[bold bright_white]jarv \u25b8 edit {row['label']}[/bold bright_white]",
            title_align="left",
            subtitle=f"[dim]{controls}[/dim]" if controls else None,
            subtitle_align="right",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            width=panel_width,
            height=height,
        )

    def _render():
        term_w, term_h = terminal_size(console=console)
        term_h = max(3, term_h)
        if edit is None:
            return _render_settings_panel(term_h)

        inner_width = max(1, max(1, term_w) - 4)
        desired_editor_height = _settings_desired_editor_height(
            edit,
            config,
            inner_width,
            term_h,
        )
        settings_min_height = 8
        editor_min_height = 7 if edit["row"].get("multiline") else 3

        if term_h >= desired_editor_height + settings_min_height:
            editor_height = desired_editor_height
            settings_height = term_h - editor_height
        elif term_h - settings_min_height >= editor_min_height:
            settings_height = settings_min_height
            editor_height = term_h - settings_height
        else:
            settings_height = max(3, min(settings_min_height, term_h // 2))
            editor_height = max(3, term_h - settings_height)

        return Group(
            _render_settings_panel(settings_height),
            _render_editor_panel(editor_height),
        )

    with Live(
        get_renderable=_render,
        console=console,
        screen=True,
        auto_refresh=False,
        transient=False,
        vertical_overflow="crop",
    ) as live, refresh_on_resize(live), mouse_capture():
        live_holder.append(live)
        while True:
            live.refresh()
            try:
                key, repeat_count = _read_key_with_repeats(text_mode=edit is not None)
            except KeyboardInterrupt:
                break

            if edit is not None:
                if edit["row"].get("multiline"):
                    if key == "ESC":
                        dirty = edit["buffer"] != edit.get("original", edit["buffer"])
                        if dirty and not edit.get("discard_armed"):
                            edit["discard_armed"] = True
                        else:
                            edit = None
                            flash = (f"{rows[selected]['label']} unchanged", "dim")
                    elif key == "CTRL_S":
                        config, message, style, done = _settings_commit_edit(edit, config)
                        if done:
                            rows = _settings_rows(config)
                            edit = None
                            flash = (message, style)
                    else:
                        edit["discard_armed"] = False
                        term_w, _term_h = terminal_size(console=console)
                        _settings_multiline_apply_key(
                            edit,
                            key,
                            repeat_count,
                            inner_width=max(1, term_w - 4),
                        )
                    continue
                if key == "ESC":
                    edit = None
                    flash = (f"{rows[selected]['label']} unchanged", "dim")
                elif edit["row"]["key"] == "provider" and key in ("UP", "DOWN", "HOME", "END"):
                    provider_keys = _settings_provider_keys()
                    current_provider = edit.get("selected_provider", config.get("provider", "openai"))
                    current_idx = provider_keys.index(current_provider) if current_provider in provider_keys else 0
                    if key == "UP":
                        current_idx = max(0, current_idx - repeat_count)
                    elif key == "DOWN":
                        current_idx = min(len(provider_keys) - 1, current_idx + repeat_count)
                    elif key == "HOME":
                        current_idx = 0
                    elif key == "END":
                        current_idx = len(provider_keys) - 1
                    edit["selected_provider"] = provider_keys[current_idx]
                    edit["buffer"] = ""
                    edit["error"] = ""
                    catalog_refresher.cancel_pending()
                    _request_catalog_refresh(
                        provider_keys[current_idx],
                        delay=0.2,
                    )
                elif key == "ENTER":
                    config, message, style, done = _settings_commit_edit(edit, config)
                    if done:
                        rows = _settings_rows(config)
                        edit = None
                        flash = (message, style)
                    else:
                        flash = None
                elif edit["row"]["key"] == "provider":
                    edit["error"] = ""
                else:
                    changed = apply_text_editor_key(
                        edit,
                        key,
                        repeat_count,
                        content_width=1,
                        allow_newlines=False,
                    )
                    if changed and edit["row"]["key"] == "model":
                        edit["catalog_input_dirty"] = True
                    edit["error"] = ""
                continue

            if pending_reset is not None:
                row = rows[pending_reset]
                config, message, style = _settings_finish_reset(row, config, key)
                rows = _settings_rows(config)
                pending_reset = None
                flash = (message, style) if key in ("y", "Y") else None
                continue

            if key == "ESC":
                break
            if key in ("UP", "k"):
                selected = max(0, selected - repeat_count)
                flash = None
            elif key in ("DOWN", "j"):
                selected = min(len(rows) - 1, selected + repeat_count)
                flash = None
            elif key == "HOME":
                selected = 0
                flash = None
            elif key == "END":
                selected = len(rows) - 1
                flash = None
            elif key == "ENTER":
                row = rows[selected]
                quick = _settings_apply_quick(row, config)
                if quick is None:
                    edit = _settings_begin_edit(row, config)
                    if row["key"] == "model":
                        _request_catalog_refresh(
                            str(config.get("provider", "openai")),
                            target_edit=edit,
                            delay=0.01,
                        )
                    flash = None
                    continue
                config, message = quick
                rows = _settings_rows(config)
                flash = (message, "green")
            elif key == "r":
                pending_reset = selected
                flash = None

    catalog_refresher.close()
    console.print("[dim]\u25cb Settings closed.[/dim]")


def cmd_settings() -> None:
    config = load_config()
    if not sys.stdin.isatty() or not console.is_terminal:
        _settings_plain(config)
        return
    _settings_interactive(config)


