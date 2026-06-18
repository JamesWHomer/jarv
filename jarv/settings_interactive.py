"""Interactive /settings screen loop."""

from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .command_input import _read_key_with_repeats, mouse_capture
from .config import DEFAULT_CONFIG, save_config
from .display import console, refresh_on_resize, terminal_size
from .settings_refresher import _ModelCatalogRefresher
from .tui_layout import append_bottom_footer
from .text_editor import apply_text_editor_key
from .settings_command import (
    _clip_text,
    _settings_apply_quick,
    _settings_begin_edit,
    _settings_column_layout,
    _settings_commit_edit,
    _settings_desired_editor_height,
    _settings_editor_lines,
    _settings_finish_reset,
    _settings_model_apply_key,
    _settings_model_choices_with_current,
    _settings_model_update_notice,
    _settings_multiline_apply_key,
    _settings_provider_keys,
    _settings_reset_action_bar,
    _settings_rows,
    _settings_value_text,
)

def run_settings_interactive(config: dict) -> None:
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
        nonlocal rows, flash

        current_edit = edit
        if (
            current_edit is not None
            and current_edit["row"]["key"] == "model"
            and current_edit.get("catalog_provider") == provider
            and current_edit.get("catalog_generation") == generation
        ):
            previous = list(current_edit.get("model_choices") or [])
            displayed_choices = _settings_model_choices_with_current(
                config,
                choices,
            )
            selected_name = ""
            previous_selected = int(
                current_edit.get("selected_model_index", 0)
            )
            if 0 <= previous_selected < len(previous):
                selected_name = previous[previous_selected][0]
            current_edit["model_choices"] = displayed_choices
            current_edit["selected_model_index"] = next(
                (
                    idx
                    for idx, (name, _description) in enumerate(displayed_choices)
                    if name == selected_name
                ),
                min(previous_selected, max(0, len(displayed_choices) - 1)),
            )
            current_edit["catalog_notice"] = _settings_model_update_notice(
                previous,
                displayed_choices,
            )
        if provider == config.get("provider"):
            from .reasoning import reconcile_reasoning_effort

            if reconcile_reasoning_effort(config) is not None:
                save_config(config)
                rows = _settings_rows(config)
                flash = (
                    "Reasoning effort reset to default for this model.",
                    "yellow",
                )
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

    def _footer() -> str:
        return "\u2191\u2193 select   Enter edit/toggle   r reset   q exit"

    def _append_bottom_footer(parts: list, height: int, footer: Text) -> None:
        append_bottom_footer(parts, height, footer, crop=True)

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
        if edit is not None and edit.get("model_validation_warning"):
            controls = "\u2190\u2192 select   Enter confirm   Esc keep editing"
        else:
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
                key, repeat_count = _read_key_with_repeats(
                    text_mode=edit is not None,
                    batch_text=edit is not None,
                )
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
                if key == "ESC" and edit.get("model_validation_warning"):
                    edit.pop("model_validation_warning", None)
                    edit.pop("model_validation_suggestion", None)
                    edit.pop("model_warning_actions", None)
                    edit.pop("model_warning_selection", None)
                    flash = None
                elif key == "ESC":
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
                elif edit["row"]["key"] == "model" and _settings_model_apply_key(
                    edit,
                    key,
                    repeat_count,
                ):
                    flash = None
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
