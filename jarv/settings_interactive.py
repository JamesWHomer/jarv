"""Interactive /settings screen loop."""

import threading
from types import SimpleNamespace

from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .command_input import _read_key_with_repeats, mouse_capture
from .config import CONFIG_FILE, DEFAULT_CONFIG, save_config
from .display import console, mark_first_paint, refresh_on_resize, terminal_size
from .settings_editor import apply_catalog_refresh, apply_editor_key, render_editor_panel
from .settings_refresher import _ModelCatalogRefresher
from .tui_frame import panel_width
from .tui_layout import append_bottom_footer
from .settings_command import (
    _clip_text,
    _settings_apply_quick,
    _settings_begin_edit,
    _settings_column_layout,
    _settings_desired_editor_height,
    _settings_finish_reset,
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
    refresh_lock = threading.RLock()

    def _refresh_live() -> None:
        if not live_holder:
            return
        with refresh_lock:
            live_holder[0].refresh()

    def _catalog_refreshed(
        provider: str,
        choices: list[tuple[str, str]],
        generation: int,
    ) -> None:
        nonlocal rows, flash

        with refresh_lock:
            apply_catalog_refresh(edit, config, provider, choices, generation)
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

    catalog = SimpleNamespace(
        request=_request_catalog_refresh,
        cancel_pending=catalog_refresher.cancel_pending,
    )

    def _footer() -> str:
        return "\u2191\u2193 select   Enter edit/toggle   r reset   Esc/q exit"

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
        width = panel_width(term_w)
        inner_width = max(1, width - 4)
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

        return Panel(
            Group(*parts),
            title="[bold bright_white]jarv \u25b8 settings[/bold bright_white]",
            title_align="left",
            subtitle=f"[dim]{CONFIG_FILE}[/dim]",
            subtitle_align="right",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            width=width,
            height=height,
        )

    def _render():
        term_w, term_h = terminal_size(console=console)
        term_h = max(3, term_h)
        if edit is None:
            return _render_settings_panel(term_h)

        width = panel_width(term_w)
        inner_width = max(1, width - 4)
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
            render_editor_panel(
                edit,
                config,
                panel_width=width,
                height=editor_height,
                title=f"edit {edit['row']['label']}",
            ),
        )

    with Live(
        get_renderable=_render,
        console=console,
        screen=True,
        auto_refresh=False,
        transient=False,
        vertical_overflow="crop",
    ) as live, refresh_on_resize(live, on_change=_refresh_live), mouse_capture():
        mark_first_paint("settings")
        live_holder.append(live)
        while True:
            _refresh_live()
            try:
                key, repeat_count = _read_key_with_repeats(
                    text_mode=edit is not None,
                    batch_text=edit is not None,
                )
            except KeyboardInterrupt:
                break

            if edit is not None:
                term_w, _term_h = terminal_size(console=console)
                config, outcome = apply_editor_key(
                    edit,
                    config,
                    key,
                    repeat_count,
                    catalog=catalog,
                    inner_width=max(1, term_w - 4),
                )
                if outcome.kind == "committed":
                    rows = _settings_rows(config)
                    edit = None
                    flash = (outcome.message, outcome.style)
                elif outcome.kind == "cancelled":
                    edit = None
                    flash = (outcome.message, outcome.style)
                elif outcome.clear_flash:
                    flash = None
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
                    if row["key"] in {"model", "auditor_model"}:
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
