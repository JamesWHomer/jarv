"""Interactive settings command."""

import difflib
import importlib
import sys

from rich.console import Group
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich import box

from .command_input import TextInput
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
    render_visual_lines,
)


from .settings_refresher import _ModelCatalogRefresher


AUDITOR_DEFAULT_MODEL_CHOICE = "default"


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


def _clip_text(value: str, width: int) -> str:
    return clip_text(value, width, ellipsis="\u2026")


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


def _settings_model_choices_with_current(
    config: dict,
    choices: list[tuple[str, str]],
    *,
    current_model: str | None = None,
) -> list[tuple[str, str]]:
    """Append the current model when it belongs to the active provider catalog."""
    from .model_catalog import cached_provider_has_model

    result = list(choices)
    current = str(
        (config.get("model") if current_model is None else current_model) or ""
    ).strip()
    if (
        current
        and all(name.lower() != current.lower() for name, _description in result)
        and cached_provider_has_model(config, current)
    ):
        result.append((current, ""))
    return result


def _settings_model_choices_for_key(
    config: dict,
    key: str,
    choices: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    current_model = (
        str(config.get("auditor_model") or "")
        if key == "auditor_model"
        else str(config.get("model") or "")
    )
    result = _settings_model_choices_with_current(
        config,
        choices,
        current_model=current_model,
    )
    if key == "auditor_model":
        result = [
            (name, description)
            for name, description in result
            if name.lower() != AUDITOR_DEFAULT_MODEL_CHOICE
        ]
        result.append((AUDITOR_DEFAULT_MODEL_CHOICE, "Default"))
    return result


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


def _settings_resolve_auditor_model(
    config: dict,
    choice: str,
    *,
    models: list[tuple[str, str]] | None = None,
) -> str:
    raw = choice.strip()
    if raw.lower() in {"", "clear", AUDITOR_DEFAULT_MODEL_CHOICE}:
        return ""
    return _settings_resolve_model(config, raw, models=models)


def _settings_model_apply_key(
    edit: dict,
    key: str,
    repeat_count: int = 1,
) -> bool:
    """Handle model picker navigation and activation of custom input."""
    if edit.get("model_validation_warning"):
        actions = edit.get("model_warning_actions") or []
        selected = max(
            0,
            min(int(edit.get("model_warning_selection", 0)), len(actions) - 1),
        )
        if key in ("LEFT", "UP", "HOME"):
            edit["model_warning_selection"] = (
                0 if key == "HOME" else max(0, selected - repeat_count)
            )
            return True
        if key in ("RIGHT", "DOWN", "END"):
            edit["model_warning_selection"] = (
                max(0, len(actions) - 1)
                if key == "END"
                else min(max(0, len(actions) - 1), selected + repeat_count)
            )
            return True
        return key not in ("ENTER", "ESC")

    models = edit.get("model_choices")
    if not isinstance(models, list) or not models:
        return False

    selected = max(
        0,
        min(int(edit.get("selected_model_index", 0)), len(models) - 1),
    )
    if key in ("UP", "DOWN", "HOME", "END"):
        input_active = bool(edit.get("model_input_active"))
        if key == "UP":
            if input_active:
                edit["model_input_active"] = False
                selected = max(0, len(models) - repeat_count)
            else:
                selected = max(0, selected - repeat_count)
        elif key == "DOWN":
            if input_active:
                edit["error"] = ""
                return True
            target = selected + repeat_count
            if target >= len(models):
                selected = len(models) - 1
                edit["model_input_active"] = True
            else:
                selected = target
        elif key == "HOME":
            selected = 0
            edit["model_input_active"] = False
        else:
            selected = len(models) - 1
            edit["model_input_active"] = False
        edit["selected_model_index"] = selected
        edit["error"] = ""
        return True

    if (
        isinstance(key, str)
        and key
        and (len(key) == 1 or isinstance(key, TextInput))
        and all(char.isprintable() for char in key)
    ):
        if not edit.get("model_input_active"):
            edit["model_input_active"] = True
        apply_text_editor_key(
            edit,
            key,
            repeat_count,
            content_width=1,
            allow_newlines=False,
        )
        edit["error"] = ""
        return True

    if key in ("ENTER", "ESC"):
        return False

    if edit.get("model_input_active"):
        apply_text_editor_key(
            edit,
            key,
            repeat_count,
            content_width=1,
            allow_newlines=False,
        )
        edit["error"] = ""
        return True
    return False


def _settings_model_suggestion(config: dict, model: str) -> str:
    """Return a cached model only when one fuzzy match is clearly strongest."""
    from .model_catalog import cached_provider_model_ids

    target = model.strip().lower()
    candidates = [
        candidate
        for candidate in cached_provider_model_ids(config)
        if target and candidate.lower() != target
    ]
    prefix_matches = [
        candidate
        for candidate in candidates
        if candidate.lower().startswith(target)
    ]
    if prefix_matches:
        shortest_length = min(len(candidate) for candidate in prefix_matches)
        shortest = [
            candidate
            for candidate in prefix_matches
            if len(candidate) == shortest_length
        ]
        if len(shortest) == 1:
            canonical = shortest[0]
            canonical_lower = canonical.lower()
            if all(
                candidate.lower() == canonical_lower
                or candidate.lower().startswith(f"{canonical_lower}-")
                for candidate in prefix_matches
            ):
                return canonical

    scored = sorted(
        (
            (
                difflib.SequenceMatcher(None, target, candidate.lower()).ratio(),
                candidate,
            )
            for candidate in candidates
        ),
        reverse=True,
    )
    if not scored:
        return ""
    best_score, best_model = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= 0.84 and best_score - second_score >= 0.06:
        return best_model
    return ""


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


def _settings_model_update_notice(
    previous: list[tuple[str, str]],
    current: list[tuple[str, str]],
) -> str:
    """Describe model replacements, additions, and removals."""
    previous_names = [name for name, _description in previous]
    current_names = [name for name, _description in current]
    if set(previous_names) == set(current_names):
        return ""

    previous_by_description: dict[str, list[str]] = {}
    current_by_description: dict[str, list[str]] = {}
    for name, description in previous:
        previous_by_description.setdefault(description, []).append(name)
    for name, description in current:
        current_by_description.setdefault(description, []).append(name)

    replaced_previous: set[str] = set()
    replaced_current: set[str] = set()
    changes: list[str] = []
    for name, description in current:
        old_matches = previous_by_description.get(description, [])
        new_matches = current_by_description.get(description, [])
        if (
            len(old_matches) == 1
            and len(new_matches) == 1
            and old_matches[0] != name
        ):
            old_name = old_matches[0]
            replaced_previous.add(old_name)
            replaced_current.add(name)
            changes.append(f"{old_name} \u2192 {name}")

    previous_set = set(previous_names)
    current_set = set(current_names)
    added = [
        name for name in current_names
        if name not in previous_set and name not in replaced_current
    ]
    removed = [
        name for name in previous_names
        if name not in current_set and name not in replaced_previous
    ]
    changes.extend(f"added {name}" for name in added)
    changes.extend(f"removed {name}" for name in removed)
    return f"Model list updated: {'; '.join(changes)}"


def _settings_model_choice_lines(
    models: list[tuple[str, str]],
    provider: str,
    selected_index: int,
    input_active: bool,
    inner_width: int,
    *,
    max_lines: int | None = None,
    active_model: str = "",
    include_prices: bool = True,
) -> list[Text]:
    if not models or (max_lines is not None and max_lines <= 0):
        return []

    model_pricing_values = None
    if include_prices:
        from .model_catalog import model_pricing_values as _model_pricing_values

        model_pricing_values = _model_pricing_values

    model_entries = []
    same_model_entries = []
    for idx, (name, description) in enumerate(models, 1):
        is_same_model = (
            name.lower() == AUDITOR_DEFAULT_MODEL_CHOICE
            and description == "Default"
        )
        if is_same_model:
            same_model_entries.append((idx, name, description))
            continue
        model_entries.append((idx, name, description))

    content_entries: list[tuple] = [
        ("model", idx, name, description)
        for idx, name, description in model_entries
    ]
    if same_model_entries:
        if content_entries:
            content_entries.append(("blank",))
        content_entries.extend(
            ("same", idx, name, description)
            for idx, name, description in same_model_entries
        )

    visible_entries = content_entries
    hidden_count = 0
    if max_lines is not None:
        full_line_count = (1 if model_entries else 0) + len(content_entries)
        if full_line_count > max_lines:
            visible_count = max(0, max_lines - 2)
            selected_entry_index = next(
                (
                    index
                    for index, entry in enumerate(content_entries)
                    if entry[0] in {"model", "same"}
                    and entry[1] - 1 == selected_index
                ),
                0,
            )
            start = min(
                max(0, selected_entry_index - visible_count // 2),
                max(0, len(content_entries) - visible_count),
            )
            visible_entries = content_entries[start:start + visible_count]
            hidden_entries = (
                content_entries[:start]
                + content_entries[start + visible_count:]
            )
            hidden_count = sum(
                1 for entry in hidden_entries if entry[0] in {"model", "same"}
            )

    display_rows = []
    for entry in visible_entries:
        if entry[0] != "model":
            continue
        _kind, idx, name, description = entry
        if model_pricing_values is None:
            input_price, cached_price, output_price = "n/a", "n/a", "n/a"
        else:
            input_price, cached_price, output_price = model_pricing_values(
                provider,
                name,
            )
        display_rows.append((
            idx,
            name,
            input_price,
            cached_price,
            output_price,
            description.split(" - ", 1)[0] if description else "",
        ))

    prefix_width = 7
    gap = "   "
    input_width = max([len("Input")] + [len(row[2]) for row in display_rows])
    cached_width = max([len("Cached")] + [len(row[3]) for row in display_rows])
    output_width = max([len("Output")] + [len(row[4]) for row in display_rows])
    tier_width = max([len("Tier")] + [len(row[5]) for row in display_rows])
    fixed_width = (
        prefix_width
        + len(gap) * 4
        + input_width
        + cached_width
        + output_width
        + tier_width
    )
    max_model_width = max(
        [len(row[1]) for row in display_rows]
        + [
            len(entry[2])
            for entry in visible_entries
            if entry[0] == "same"
        ]
        + [len("Model")]
    )
    model_width = min(
        max(36, max_model_width + 2),
        max(1, inner_width - fixed_width),
    )

    lines: list[Text] = []
    if display_rows:
        header = (
            " " * prefix_width
            + f"{'Model':<{model_width}}{gap}"
            + f"{'Input':>{input_width}}{gap}"
            + f"{'Cached':>{cached_width}}{gap}"
            + f"{'Output':>{output_width}}{gap}"
            + f"{'Tier':<{tier_width}}"
        )
        lines.append(
            Text(
                _clip_text(header, inner_width),
                style="dim bold",
                no_wrap=True,
                overflow="crop",
            )
        )

    display_by_index = {row[0]: row for row in display_rows}
    rows: list[Text] = []
    for entry in visible_entries:
        kind = entry[0]
        if kind == "blank":
            rows.append(Text(""))
            continue
        if kind == "same":
            _kind, idx, name, _description = entry
            is_selected = idx - 1 == selected_index
            is_bright = is_selected and not input_active
            marker = "\u203a" if is_selected else " "
            line = Text(no_wrap=True, overflow="crop")
            line.append(
                f"  {marker}{idx:>2}. ",
                style="bold cyan" if is_bright else "dim",
            )
            line.append(
                f"{_clip_text(name, model_width):<{model_width}}",
                style="bold bright_white" if is_bright else "white",
            )
            line.append(gap, style="dim")
            description = "uses active model"
            if active_model:
                description = f"{description} ({active_model})"
            line.append(
                description,
                style="white" if is_bright else "dim",
            )
            line.truncate(inner_width, overflow="ellipsis")
            rows.append(line)
            continue

        _kind, idx, _name, _description = entry
        (
            _idx,
            name,
            input_price,
            cached_price,
            output_price,
            tier,
        ) = display_by_index[idx]
        is_selected = idx - 1 == selected_index
        is_bright = is_selected and not input_active
        marker = "\u203a" if is_selected else " "
        line = Text(no_wrap=True, overflow="crop")
        line.append(
            f"  {marker}{idx:>2}. ",
            style="bold cyan" if is_bright else "dim",
        )
        line.append(
            f"{_clip_text(name, model_width):<{model_width}}",
            style="bold bright_white" if is_bright else "dim",
        )
        line.append(gap, style="dim")
        line.append(
            f"{input_price:>{input_width}}",
            style="white" if is_bright else "dim",
        )
        line.append(gap, style="dim")
        line.append(
            f"{cached_price:>{cached_width}}",
            style="white" if is_bright else "dim",
        )
        line.append(gap, style="dim")
        line.append(
            f"{output_price:>{output_width}}",
            style="white" if is_bright else "dim",
        )
        line.append(gap, style="dim")
        line.append(
            f"{_clip_text(tier, tier_width):<{tier_width}}",
            style="white" if is_bright else "dim",
        )
        rows.append(line)

    if hidden_count:
        label = "choice" if hidden_count == 1 else "choices"
        rows.append(Text(
            _clip_text(
                f"  ... {hidden_count} more {label}; type a number/name/custom",
                inner_width,
            ),
            style="dim",
        ))
    return lines + rows


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
    body, cursor_idx = _settings_multiline_visual_lines(edit, inner_width)
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
    content_height = len(
        _settings_editor_lines(
            edit,
            config,
            inner_width,
            max_lines=max_content_lines,
            price_models=False,
        )
    )
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


