"""Model and provider picker for the settings command.

The choice grids, model/provider resolution, and apply-key handling for the
`model`/`auditor_model` settings rows. Extracted from settings_command.py, which
now re-exports these as a facade. Provider/model-catalog imports stay lazy inside
the functions (as before) to keep import cost down.
"""

import difflib

from rich.text import Text

from .command_input import TextInput
from .config import DEFAULT_CONFIG
from .settings_helpers import AUDITOR_DEFAULT_MODEL_CHOICE, _clip_text
from .text_editor import apply_text_editor_key


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
