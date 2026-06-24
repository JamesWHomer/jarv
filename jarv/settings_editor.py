"""Shared fullscreen editor for the /settings and /setup screens.

The edit-mode rendering and key dispatch used to live inside
``run_settings_interactive``. They are extracted here as closure-free functions so
both the settings screen and the setup wizard drive the *same* per-setting
submenus without duplicating edit/render code.

- :func:`render_editor_panel` renders one fullscreen editor panel.
- :func:`apply_editor_key` is the full edit-mode key dispatch, returning an
  :class:`EditOutcome` (continue / committed / cancelled).
- :func:`apply_catalog_refresh` applies a background catalog refresh to an active
  model edit (the edit-mutating half of the settings screen's refresh callback).

A ``catalog`` adapter (exposing ``.request(provider, *, target_edit, delay)`` and
``.cancel_pending()``) is injected so this module stays free of the per-screen
``_ModelCatalogRefresher`` wiring.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from .text_editor import apply_text_editor_key
from .settings_command import (
    _settings_commit_edit,
    _settings_edit_is_dirty,
    _settings_editor_lines,
    _settings_model_apply_key,
    _settings_model_choices_for_key,
    _settings_model_update_notice,
    _settings_multiline_apply_key,
    _settings_provider_keys,
)


@dataclass(frozen=True)
class EditOutcome:
    """Result of dispatching one key to the editor.

    ``kind`` is ``"continue"`` (still editing), ``"committed"`` (the value was
    saved), or ``"cancelled"`` (the edit was abandoned). ``message``/``style``
    carry the flash text for committed/cancelled outcomes; ``clear_flash`` mirrors
    the settings screen's behaviour of clearing the flash on certain continue keys.
    """

    kind: str
    message: str = ""
    style: str = ""
    clear_flash: bool = False


def _continue(*, clear_flash: bool = False) -> EditOutcome:
    return EditOutcome("continue", clear_flash=clear_flash)


def _committed(message: str, style: str) -> EditOutcome:
    return EditOutcome("committed", message, style)


def _cancelled(message: str) -> EditOutcome:
    return EditOutcome("cancelled", message, "dim")


def render_editor_panel(
    edit: dict | None,
    config: dict,
    *,
    panel_width: int,
    height: int,
    title: str,
    controls: str | None = None,
) -> Panel:
    """Render a single fullscreen editor panel for *edit*.

    *title* is the suffix after ``jarv ▸`` (settings passes ``edit {label}``;
    setup passes ``setup · Step N/M · {label}``). *controls* overrides the
    default subtitle help in the normal (non-warning, non-discard) case.
    """
    inner_width = max(1, panel_width - 4)
    height = max(3, height)
    content_rows = max(1, height - 2)
    row = edit["row"] if edit is not None else {"label": "setting"}
    editor_parts = _settings_editor_lines(edit, config, inner_width, max_lines=content_rows)
    if not editor_parts:
        editor_parts = [Text("")]

    if edit is not None and edit.get("model_validation_warning"):
        footer = "←→ select   Enter confirm   Esc keep editing"
    elif edit is not None and edit.get("discard_armed"):
        footer = "Esc discard"
    elif controls is not None:
        footer = controls
    else:
        footer = "" if row.get("multiline") else "Enter save   Esc back"

    return Panel(
        Group(*editor_parts),
        title=f"[bold bright_white]jarv ▸ {title}[/bold bright_white]",
        title_align="left",
        subtitle=f"[dim]{footer}[/dim]" if footer else None,
        subtitle_align="right",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 1),
        width=panel_width,
        height=height,
    )


def apply_editor_key(
    edit: dict,
    config: dict,
    key: str,
    repeat: int,
    *,
    catalog,
    inner_width: int,
) -> tuple[dict, EditOutcome]:
    """Dispatch one key to the active editor; returns ``(config, outcome)``.

    ``catalog`` must expose ``request(provider, *, target_edit=None, delay=0)`` and
    ``cancel_pending()`` for provider-driven model catalog refreshes.
    """
    row = edit["row"]
    key_name = row["key"]

    if row.get("multiline"):
        if key == "ESC":
            dirty = _settings_edit_is_dirty(edit, config)
            if dirty and not edit.get("discard_armed"):
                edit["discard_armed"] = True
                return config, _continue()
            return config, _cancelled(f"{row['label']} unchanged")
        if key == "CTRL_S":
            config, message, style, done = _settings_commit_edit(edit, config)
            if done:
                return config, _committed(message, style)
            return config, _continue()
        edit["discard_armed"] = False
        _settings_multiline_apply_key(edit, key, repeat, inner_width=inner_width)
        return config, _continue()

    if key == "ESC" and edit.get("model_validation_warning"):
        edit.pop("model_validation_warning", None)
        edit.pop("model_validation_suggestion", None)
        edit.pop("model_warning_actions", None)
        edit.pop("model_warning_selection", None)
        return config, _continue(clear_flash=True)

    if key == "ESC":
        if _settings_edit_is_dirty(edit, config) and not edit.get("discard_armed"):
            edit["discard_armed"] = True
            return config, _continue(clear_flash=True)
        return config, _cancelled(f"{row['label']} unchanged")

    if key_name == "provider" and key in ("UP", "DOWN", "HOME", "END"):
        edit["discard_armed"] = False
        provider_keys = _settings_provider_keys()
        current_provider = edit.get("selected_provider", config.get("provider", "openai"))
        current_idx = (
            provider_keys.index(current_provider)
            if current_provider in provider_keys
            else 0
        )
        if key == "UP":
            current_idx = max(0, current_idx - repeat)
        elif key == "DOWN":
            current_idx = min(len(provider_keys) - 1, current_idx + repeat)
        elif key == "HOME":
            current_idx = 0
        elif key == "END":
            current_idx = len(provider_keys) - 1
        edit["selected_provider"] = provider_keys[current_idx]
        edit["buffer"] = ""
        edit["error"] = ""
        catalog.cancel_pending()
        catalog.request(provider_keys[current_idx], delay=0.2)
        return config, _continue()

    if key_name in {"model", "auditor_model"} and _settings_model_apply_key(
        edit, key, repeat
    ):
        edit["discard_armed"] = False
        return config, _continue(clear_flash=True)

    if key == "ENTER":
        config, message, style, done = _settings_commit_edit(edit, config)
        if done:
            return config, _committed(message, style)
        return config, _continue(clear_flash=True)

    if key_name == "provider":
        edit["discard_armed"] = False
        edit["error"] = ""
        return config, _continue()

    # API key: a single backspace clears the masked stand-in for a stored key.
    if (
        key_name == "api_key"
        and edit.get("placeholder_active")
        and key in ("BACKSPACE", "DELETE")
    ):
        edit["placeholder_active"] = False
        edit["cleared"] = True
        edit["discard_armed"] = False
        edit["error"] = ""
        return config, _continue(clear_flash=True)

    edit["discard_armed"] = False
    before = str(edit.get("buffer", ""))
    apply_text_editor_key(
        edit,
        key,
        repeat,
        content_width=1,
        allow_newlines=False,
    )
    # Typing over the masked stand-in replaces the stored key.
    if key_name == "api_key" and str(edit.get("buffer", "")) != before:
        edit["placeholder_active"] = False
    edit["error"] = ""
    return config, _continue()


def apply_catalog_refresh(
    edit: dict | None,
    config: dict,
    provider: str,
    choices: list[tuple[str, str]],
    generation: int,
) -> bool:
    """Apply a background catalog refresh to an active model edit.

    Returns ``True`` when *edit* matched the refresh and was updated.
    """
    if (
        edit is None
        or edit["row"]["key"] not in {"model", "auditor_model"}
        or edit.get("catalog_provider") != provider
        or edit.get("catalog_generation") != generation
    ):
        return False

    previous = list(edit.get("model_choices") or [])
    displayed_choices = _settings_model_choices_for_key(
        config,
        edit["row"]["key"],
        choices,
    )
    selected_name = ""
    previous_selected = int(edit.get("selected_model_index", 0))
    if 0 <= previous_selected < len(previous):
        selected_name = previous[previous_selected][0]
    edit["model_choices"] = displayed_choices
    edit["selected_model_index"] = next(
        (
            idx
            for idx, (name, _description) in enumerate(displayed_choices)
            if name == selected_name
        ),
        min(previous_selected, max(0, len(displayed_choices) - 1)),
    )
    edit["catalog_notice"] = _settings_model_update_notice(previous, displayed_choices)
    return True
