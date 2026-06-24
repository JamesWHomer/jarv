"""Interactive settings command."""

import importlib
import re
import sys

from rich.console import Group
from rich.table import Table
from rich.text import Text
from rich import box

from .config import (
    CONFIG_FILE,
    DEFAULT_CONFIG,
    load_config,
    save_config,
    validate_config,
)
from .display import console, jarv_panel, section_rule
from .settings_schema import (
    settings_rows,
    settings_service_tier_choices,
    settings_service_tier_description,
)
from .tui_layout import clip_text
from .text_editor import (
    apply_text_editor_key,
    initialize_text_editor,
    render_single_line,
    render_visual_line_window,
    render_visual_lines,
)


from .settings_refresher import _ModelCatalogRefresher
from .settings_helpers import (
    AUDITOR_DEFAULT_MODEL_CHOICE,
    _clip_text,
    _settings_choice_label,
)
from .settings_model_picker import (
    _settings_choice_grid_lines,
    _settings_column_layout,
    _settings_default_model,
    _settings_default_model_for_provider,
    _settings_model_apply_key,
    _settings_model_choice_lines,
    _settings_model_choices,
    _settings_model_choices_for_key,
    _settings_model_choices_with_current,
    _settings_model_suggestion,
    _settings_model_update_notice,
    _settings_provider_choice_lines,
    _settings_provider_choices,
    _settings_provider_display_label,
    _settings_provider_keys,
    _settings_provider_note,
    _settings_resolve_auditor_model,
    _settings_resolve_model,
    _settings_resolve_provider,
)


def _settings_save_validated(config: dict) -> bool:
    trial = dict(DEFAULT_CONFIG)
    trial.update(config)
    if not validate_config(trial):
        return False
    config.clear()
    config.update(trial)
    save_config(config)
    return True


def _settings_is_model_picker_key(key: str) -> bool:
    return key in {"model", "auditor_model"}


def _settings_service_tier_choices(config: dict) -> tuple[tuple[str, str], ...]:
    return settings_service_tier_choices(config)


def _settings_service_tier(config: dict) -> str:
    from .provider_catalog import configured_service_tier

    return configured_service_tier(config)


def _settings_service_tier_description(config: dict) -> str:
    return settings_service_tier_description(config)


def _settings_set_service_tier(config: dict, tier: str) -> None:
    provider = str(config.get("provider", "openai"))
    existing = config.get("service_tiers")
    configured = dict(existing) if isinstance(existing, dict) else {}
    configured[provider] = tier
    config["service_tiers"] = configured




def _settings_has_api_key(config: dict) -> tuple[bool, str]:
    from .provider import LOCAL_PROVIDERS, PROVIDERS
    from .provider_auth import api_key_source

    provider = config.get("provider", "openai")
    if provider in LOCAL_PROVIDERS:
        return True, "not needed"
    source = api_key_source(config)
    if source == "config":
        return True, "configured"
    if source == "env":
        env_key = PROVIDERS.get(provider, {}).get("env_key") or "env"
        return True, f"from {env_key}"
    label = PROVIDERS.get(provider, {}).get("label", provider)
    return False, f"missing for {label}"


def _settings_rows(config: dict) -> list[dict]:
    return settings_rows(config)


def _settings_value_text(row: dict, config: dict, *, selected: bool = False) -> Text:
    from .provider import PROVIDERS

    key = row["key"]
    kind = row["kind"]
    if kind == "tool_bool":
        disabled = config.get("disabled_tools", [])
        value = not isinstance(disabled, list) or row["tool_name"] not in disabled
    else:
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
    if kind in {"bool", "tool_bool"}:
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
        if key == "auditor_model" and not value:
            return Text(
                row.get("empty", AUDITOR_DEFAULT_MODEL_CHOICE),
                style="dim italic",
            )
        if value:
            return Text(str(value), style="bold green" if selected else "green")
        return Text(row.get("empty", "empty"), style="dim italic")
    return Text(str(value), style="bold" if selected else "")


def _settings_apply_quick(row: dict, config: dict) -> tuple[dict, str] | None:
    key = row["key"]
    kind = row["kind"]

    if kind == "bool":
        config[key] = not bool(config.get(key, DEFAULT_CONFIG.get(key, False)))
        if not _settings_save_validated(config):
            return config, "config validation failed"
        state = "on" if config[key] else "off"
        return config, f"saved {row['label']}: {state}"

    if kind == "tool_bool":
        name = row["tool_name"]
        raw_disabled = config.get("disabled_tools", [])
        disabled = list(raw_disabled) if isinstance(raw_disabled, list) else []
        if name in disabled:
            disabled.remove(name)
            state = "on"
        else:
            disabled.append(name)
            state = "off"
        config["disabled_tools"] = disabled
        if not _settings_save_validated(config):
            return config, "config validation failed"
        return config, f"saved {row['label']}: {state}"

    if kind == "choice":
        choices = row["choices"]
        current = (
            _settings_service_tier(config)
            if key == "service_tier"
            else config.get(key, DEFAULT_CONFIG.get(key, choices[0][0]))
        )
        idx = next(
            (i for i, (value, _) in enumerate(choices) if value == current),
            None,
        )
        value = (
            choices[0][0]
            if idx is None
            else choices[(idx + 1) % len(choices)][0]
        )
        if key == "service_tier":
            _settings_set_service_tier(config, value)
        else:
            config[key] = value
        if not _settings_save_validated(config):
            return config, "config validation failed"
        return config, f"saved {row['label']}: {_settings_choice_label(value, choices)}"

    return None


def _settings_reset_value(row: dict, config: dict):
    if row["kind"] == "tool_bool":
        return True
    if row["key"] == "model":
        return _settings_default_model(config)
    if row["key"] == "service_tier":
        return "standard"
    return DEFAULT_CONFIG[row["key"]]


def _settings_reset_row(row: dict, config: dict) -> tuple[dict, str]:
    key = row["key"]
    if row["kind"] == "tool_bool":
        name = row["tool_name"]
        raw_disabled = config.get("disabled_tools", [])
        disabled = list(raw_disabled) if isinstance(raw_disabled, list) else []
        if name in disabled:
            disabled.remove(name)
            config["disabled_tools"] = disabled
            _settings_save_validated(config)
        return config, f"reset {row['label']}"
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
            _settings_save_validated(config)
            return config, "cleared stored API key"
        return config, "no stored API key"
    if key == "service_tier":
        _settings_set_service_tier(config, "standard")
        _settings_save_validated(config)
        return config, "reset Processing tier"
    if key not in DEFAULT_CONFIG:
        return config, f"{row['label']} has no default"
    config[key] = _settings_reset_value(row, config)
    _settings_save_validated(config)
    return config, f"reset {row['label']}"


def _settings_reset_action_bar(
    row: dict,
    config: dict,
    inner_width: int,
) -> Text:
    key = row["key"]
    if key == "api_key":
        prompt = "Clear stored API key?"
        controls = "y clear   Esc back"
        current = default = ""
        left = prompt
    else:
        current = _settings_value_text(row, config).plain
        default_config = dict(config)
        if row["kind"] == "tool_bool":
            raw_disabled = config.get("disabled_tools", [])
            disabled = list(raw_disabled) if isinstance(raw_disabled, list) else []
            if row["tool_name"] in disabled:
                disabled.remove(row["tool_name"])
            default_config["disabled_tools"] = disabled
        elif key == "service_tier":
            _settings_set_service_tier(default_config, "standard")
        else:
            default_config[key] = _settings_reset_value(row, config)
        default = _settings_value_text(row, default_config).plain
        prompt = f"Reset {row['label']}?"
        controls = "y reset   Esc back"
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


def _settings_begin_edit(row: dict, config: dict) -> dict:
    key = row["key"]
    buffer = ""
    if row["kind"] in ("int", "text"):
        buffer = str(config.get(key, DEFAULT_CONFIG.get(key, "")))

    if key == "api_key":
        from .provider import LOCAL_PROVIDERS
        from .provider_auth import api_key_source

        provider = config.get("provider", "openai")
        readonly = provider in LOCAL_PROVIDERS
        source = api_key_source(config)
        edit = {
            "row": row,
            "secret": True,
            "readonly": readonly,
            "error": "",
            "key_source": source,
        }
        initialize_text_editor(edit, "")
        # A key already stored in config is shown as a masked stand-in that a
        # single backspace clears; an env key is overridable but has nothing to
        # mask, so no placeholder is armed.
        edit["placeholder_active"] = (not readonly) and source == "config"
        edit["discard_armed"] = False
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
        edit["original_selected_provider"] = edit["selected_provider"]
        edit["discard_armed"] = False
    elif _settings_is_model_picker_key(key):
        edit["model_choices"] = _settings_model_choices_for_key(
            config,
            key,
            _settings_model_choices(config),
        )
        edit["catalog_provider"] = config.get("provider", "openai")
        if key == "auditor_model":
            current_model = str(
                config.get("auditor_model") or AUDITOR_DEFAULT_MODEL_CHOICE
            ).lower()
        else:
            current_model = str(config.get("model") or "").lower()
        edit["selected_model_index"] = next(
            (
                idx
                for idx, (name, _description) in enumerate(edit["model_choices"])
                if name.lower() == current_model
            ),
            0,
        )
        edit["model_input_active"] = False
        initialize_text_editor(edit, "")
        edit["original_selected_model_index"] = edit["selected_model_index"]
        edit["original_model_input_active"] = False
        edit["discard_armed"] = False
    return edit


def _settings_edit_is_dirty(edit: dict | None, config: dict) -> bool:
    if edit is None or edit.get("readonly"):
        return False

    row = edit["row"]
    key = row["key"]
    if key == "provider":
        return edit.get("selected_provider") != edit.get(
            "original_selected_provider",
            config.get("provider", "openai"),
        )
    if _settings_is_model_picker_key(key):
        return (
            bool(edit.get("model_input_active"))
            != bool(edit.get("original_model_input_active"))
            or int(edit.get("selected_model_index", 0))
            != int(edit.get("original_selected_model_index", 0))
            or str(edit.get("buffer", "")) != str(edit.get("original", ""))
            or bool(edit.get("model_validation_warning"))
        )
    return str(edit.get("buffer", "")) != str(edit.get("original", ""))


def _settings_discard_warning(inner_width: int) -> Text:
    return Text(
        _clip_text("  Unsaved changes. Esc again to discard.", inner_width),
        style="bold yellow",
    )


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
    body, _cursor_idx = _settings_multiline_visual_lines(edit, inner_width)
    tail: list[Text] = []
    if edit.get("discard_armed"):
        tail.append(_settings_discard_warning(inner_width))
    tail.append(
        Text(
            _clip_text(
                "  Enter newline   Ctrl+S save   Esc back   Arrows move",
                inner_width,
            ),
            style="dim italic",
        )
    )

    if max_lines is None:
        return intro + body + tail

    body_budget = max(1, max_lines - len(intro) - len(tail))
    visible_body, _visible_cursor_idx, _start = render_visual_line_window(
        edit,
        max(1, inner_width - 4),
        max_lines=body_budget,
        indent="  ",
    )
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
    price_models: bool = True,
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
    elif _settings_is_model_picker_key(key):
        models = edit.get("model_choices")
        if not isinstance(models, list):
            models = _settings_model_choices_for_key(
                config,
                key,
                _settings_model_choices(config),
            )
        if models:
            notice = str(edit.get("catalog_notice") or "")
            if notice:
                intro.append(Text(_clip_text(f"  {notice}", inner_width), style="green"))
            choice_items = []
            prompt = (
                "Auditor model number, name, or custom model"
                if key == "auditor_model"
                else "Model number, name, or custom model"
            )
        else:
            intro.append(Text(_clip_text(f"  default for {provider_label}: {_settings_default_model(config)}", inner_width), style="dim"))
            choice_items = []
            prompt = "Model name"
    elif key == "api_key":
        if edit.get("readonly"):
            lines = [
                Text(_clip_text(f"  {provider_label} does not need an API key.", inner_width), style="green"),
                Text(_clip_text("  Esc returns to settings.", inner_width), style="dim italic"),
            ]
            return lines[:max_lines] if max_lines is not None else lines
        key_url = PROVIDERS.get(provider, {}).get("key_url")
        if key_url:
            intro.append(Text(_clip_text(f"  Get a key at {key_url}", inner_width), style="cyan"))
        env_key = PROVIDERS.get(provider, {}).get("env_key") or "provider env var"
        source = edit.get("key_source")
        if source == "config" and edit.get("placeholder_active"):
            intro.append(Text(_clip_text("  A key is already saved for this provider (shown as *****).", inner_width), style="green"))
            intro.append(Text(_clip_text("  Backspace clears it, or type a new key to replace it.", inner_width), style="dim"))
        elif source == "config" and edit.get("cleared") and not edit.get("buffer"):
            intro.append(Text(_clip_text("  The saved key will be removed when you save.", inner_width), style="yellow"))
            intro.append(Text(_clip_text("  Type a new key to keep one instead.", inner_width), style="dim"))
        elif source == "env":
            intro.append(Text(_clip_text(f"  A key is set via {env_key} in your environment.", inner_width), style="green"))
            intro.append(Text(_clip_text("  Type a key here to override it for this provider.", inner_width), style="dim"))
        else:
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
        warning_active = bool(edit.get("model_validation_warning"))
        input_active = (
            (not _settings_is_model_picker_key(key) or bool(edit.get("model_input_active")))
            and not warning_active
        )
        prompt_style = "bold bright_white" if input_active else "dim"
        line.append(
            _clip_text(f"  {prompt}: ", inner_width),
            style=prompt_style,
        )
        remaining = max(1, inner_width - len(line.plain))
        if edit.get("placeholder_active") and not edit.get("buffer"):
            # Masked stand-in for a key already stored in config; one backspace
            # clears the whole thing.
            stars = "*" * min(5, max(1, remaining - 1))
            line.append(stars, style="green" if input_active else "dim")
            line.append(" ", style="reverse" if input_active else "dim")
        else:
            masked = bool(
                edit.get("secret")
                and str(edit["buffer"]).lower() != "clear"
            )
            line.append_text(
                render_single_line(
                    edit,
                    remaining,
                    masked=masked,
                    text_style="green" if input_active else "dim",
                    cursor_style="reverse" if input_active else "dim",
                    cursor_visible=not bool(edit.get("model_validation_warning")),
                )
            )
        tail.append(line)

    if edit.get("error"):
        tail.append(Text(_clip_text(f"  {edit['error']}", inner_width), style="red"))
    if edit.get("discard_armed"):
        tail.append(_settings_discard_warning(inner_width))
    if _settings_is_model_picker_key(key) and edit.get("model_validation_warning"):
        suggestion = str(edit.get("model_validation_suggestion") or "")
        warning_selection = int(edit.get("model_warning_selection", 0))
        warning = (
            f"  Not found. Did you mean {suggestion}?"
            if suggestion
            else f"  Not found in {provider_label}'s cached models."
        )
        tail.append(Text(
            _clip_text(warning, inner_width),
            style="yellow",
        ))
        action_line = Text("  ", no_wrap=True, overflow="crop")
        for index, action in enumerate(edit.get("model_warning_actions") or []):
            selected = index == warning_selection
            marker = "\u203a" if selected else " "
            if index:
                action_line.append("   ")
            action_line.append(
                f"{marker} {action['label']}",
                style="bold bright_white" if selected else "dim",
            )
        action_line.truncate(inner_width, overflow="ellipsis")
        tail.append(action_line)

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
    elif _settings_is_model_picker_key(key) and models:
        choices = _settings_model_choice_lines(
            models,
            provider,
            (
                -1
                if edit.get("model_validation_warning")
                else int(edit.get("selected_model_index", 0))
            ),
            bool(edit.get("model_input_active"))
            or bool(edit.get("model_validation_warning")),
            inner_width,
            max_lines=choice_line_budget,
            active_model=(
                str(config.get("model") or _settings_default_model(config))
                if key == "auditor_model"
                else ""
            ),
            include_prices=price_models,
        )
        if choice_line_budget is None or len(choices) < choice_line_budget:
            choices.append(Text(""))

    return intro + choices + tail


def _settings_desired_editor_height(
    edit: dict,
    config: dict,
    inner_width: int,
    terminal_height: int,
) -> int:
    max_content_lines = max(1, terminal_height - 2)
    row = edit["row"]
    max_lines = None if row.get("multiline") else max_content_lines
    content_height = len(
        _settings_editor_lines(
            edit,
            config,
            inner_width,
            max_lines=max_lines,
            price_models=False,
        )
    )
    minimum = 7 if row.get("multiline") else 3
    return min(terminal_height, max(minimum, content_height + 2))


def _settings_commit_edit(edit: dict, config: dict) -> tuple[dict, str, str, bool]:
    row = edit["row"]
    key = row["key"]
    raw_buffer = edit["buffer"]
    raw = raw_buffer.strip()

    if edit.get("readonly"):
        return config, f"{row['label']} unchanged", "dim", True

    if key == "provider":
        from .reasoning import reconcile_reasoning_effort

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
        reset_effort = reconcile_reasoning_effort(config)
        if not _settings_save_validated(config):
            edit["error"] = "Provider change failed validation."
            return config, edit["error"], "red", False
        message = f"saved Provider: {_settings_value_text(row, config).plain}"
        if config.get("model") != old_model:
            message += f" (model: {config.get('model')})"
        if reset_effort is not None:
            message += " (reasoning effort reset to default)"
        return config, message, "green", True

    if key == "api_key":
        provider = config.get("provider", "openai")
        if not raw:
            if edit.get("cleared"):
                # User backspaced over the masked stand-in for a stored key.
                api_keys = config.get("api_keys")
                if isinstance(api_keys, dict):
                    api_keys.pop(provider, None)
                config["api_key"] = ""
                if not _settings_save_validated(config):
                    edit["error"] = "Could not clear API key."
                    return config, edit["error"], "red", False
                return config, "cleared stored API key", "cyan", True
            return config, "API key unchanged", "dim", True
        if raw.lower() == "clear":
            api_keys = config.get("api_keys")
            if isinstance(api_keys, dict):
                api_keys.pop(provider, None)
            config["api_key"] = ""
            if not _settings_save_validated(config):
                edit["error"] = "Could not clear API key."
                return config, edit["error"], "red", False
            return config, "cleared stored API key", "cyan", True
        from .provider import KEY_PATTERNS

        pattern = KEY_PATTERNS.get(provider)
        if pattern and not edit.get("format_warned") and not re.match(pattern, raw):
            edit["format_warned"] = True
            edit["error"] = (
                "Key format looks off for this provider. Enter again to save anyway."
            )
            return config, edit["error"], "yellow", False
        config.setdefault("api_keys", {})[provider] = raw
        config["api_key"] = ""
        if not _settings_save_validated(config):
            edit["error"] = "Could not save API key."
            return config, edit["error"], "red", False
        return config, "saved API key", "green", True

    if _settings_is_model_picker_key(key):
        warning_model = str(edit.get("model_validation_warning") or "")
        if warning_model:
            actions = edit.get("model_warning_actions") or []
            selected = max(
                0,
                min(
                    int(edit.get("model_warning_selection", 0)),
                    len(actions) - 1,
                ),
            )
            action = actions[selected]["value"] if actions else "edit"
            if action == "edit":
                edit.pop("model_validation_warning", None)
                edit.pop("model_validation_suggestion", None)
                edit.pop("model_warning_actions", None)
                edit.pop("model_warning_selection", None)
                return config, "Continue editing the model name.", "dim", False
            model = warning_model if action == "continue" else str(action)
            edit.pop("model_validation_warning", None)
            edit.pop("model_validation_suggestion", None)
            edit.pop("model_warning_actions", None)
            edit.pop("model_warning_selection", None)
            if key == "model":
                from .reasoning import reconcile_reasoning_effort

                config["model"] = model
                reset_effort = reconcile_reasoning_effort(config)
            else:
                config["auditor_model"] = model
                reset_effort = None
            if not _settings_save_validated(config):
                edit["error"] = "Model change failed validation."
                return config, edit["error"], "red", False
            display = model if model else AUDITOR_DEFAULT_MODEL_CHOICE
            message = f"saved {row['label']}: {display}"
            if reset_effort is not None:
                message += " (reasoning effort reset to default)"
            return config, message, "yellow", True

        models = edit.get("model_choices")
        input_active = bool(edit.get("model_input_active"))
        if (
            isinstance(models, list)
            and models
            and not input_active
        ):
            selected = max(
                0,
                min(
                    int(edit.get("selected_model_index", 0)),
                    len(models) - 1,
                ),
            )
            raw = models[selected][0]
        if key == "auditor_model":
            model = _settings_resolve_auditor_model(
                config,
                raw,
                models=models if isinstance(models, list) else None,
            )
        else:
            model = _settings_resolve_model(
                config,
                raw,
                models=models if isinstance(models, list) else None,
            )
        if key == "model" and not model.strip():
            edit["error"] = "Model must not be empty."
            return config, edit["error"], "red", False
        if input_active and model:
            from .model_catalog import cached_provider_has_model

            listed_model = (
                isinstance(models, list)
                and any(
                    name.lower() == model.lower()
                    for name, _description in models
                )
            )
            if not listed_model and not cached_provider_has_model(config, model):
                suggestion = _settings_model_suggestion(config, model)
                edit["model_validation_warning"] = model
                edit["model_validation_suggestion"] = suggestion
                if suggestion:
                    edit["model_warning_actions"] = [
                        {"label": f"Use {suggestion}", "value": suggestion},
                        {"label": "Keep editing", "value": "edit"},
                        {"label": f"Use {model} anyway", "value": "continue"},
                    ]
                else:
                    edit["model_warning_actions"] = [
                        {"label": "Keep editing", "value": "edit"},
                        {"label": "Use anyway", "value": "continue"},
                    ]
                edit["model_warning_selection"] = 0
                edit["error"] = ""
                return (
                    config,
                    f"Model not found in cached provider list: {model}",
                    "yellow",
                    False,
                )
        if key == "model":
            from .reasoning import reconcile_reasoning_effort

            config["model"] = model
            reset_effort = reconcile_reasoning_effort(config)
        else:
            config["auditor_model"] = model
            reset_effort = None
        if not _settings_save_validated(config):
            edit["error"] = "Model change failed validation."
            return config, edit["error"], "red", False
        display = model if model else AUDITOR_DEFAULT_MODEL_CHOICE
        message = f"saved {row['label']}: {display}"
        if reset_effort is not None:
            message += " (reasoning effort reset to default)"
        return config, message, "green", True

    if key == "system_prompt":
        config[key] = raw_buffer
        if not _settings_save_validated(config):
            edit["error"] = "System prompt failed validation."
            return config, edit["error"], "red", False
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
        if not _settings_save_validated(config):
            edit["error"] = "Invalid value for this setting."
            return config, edit["error"], "red", False
        return config, f"saved {row['label']}: {value}", "green", True

    value = "" if raw.lower() == "clear" else raw
    config[key] = value
    if not _settings_save_validated(config):
        edit["error"] = "Invalid value for this setting."
        return config, edit["error"], "red", False
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


def cmd_settings() -> None:
    config = load_config()
    from .reasoning import reconcile_reasoning_effort

    if reconcile_reasoning_effort(config) is not None:
        _settings_save_validated(config)
    if not sys.stdin.isatty() or not console.is_terminal:
        _settings_plain(config)
        return
    run_settings_interactive = importlib.import_module(
        "jarv.settings_interactive"
    ).run_settings_interactive
    run_settings_interactive(config)


