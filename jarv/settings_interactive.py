"""Interactive /settings screen on the shared AltScreenApp loop.

Until recently this was the one full-screen view still driven by the historical
threaded model -- a Rich ``Live`` plus the ``refresh_on_resize`` daemon thread
(``SIGWINCH`` + ``live._lock``) that the rest of the TUI had already left behind
for :class:`~jarv.tui_app.AltScreenApp`. It stayed behind because it has a
genuinely concurrent producer: a background model-catalog refresher repaints the
screen from its own thread when a provider's model list arrives.

That producer now follows the same discipline as every other worker thread on the
loop -- it mutates state under ``self.lock`` and calls
:meth:`~jarv.tui_app.AltScreenApp.invalidate`; only the loop thread ever paints.
The screen mirrors the :class:`~jarv.setup_interactive.SetupApp` structure (it
reuses the very same per-setting editors via :mod:`jarv.settings_editor`), adding
the scrollable settings list and the reset-confirmation flow.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from rich import box
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from .config import CONFIG_FILE, save_config
from .display import console, terminal_size
from .settings_editor import apply_catalog_refresh, apply_editor_key, render_editor_panel
from .settings_refresher import _ModelCatalogRefresher
from .tui_app import AltScreenApp
from .tui_frame import panel_width, wrap_frame
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


class SettingsApp(AltScreenApp):
    """The /settings screen on the single-threaded alt-screen loop."""

    use_mouse_capture = True
    use_bracketed_paste = False
    clear_on_resize = False
    first_paint_label = "settings"

    def __init__(self, config: dict, *, render_console=console):
        super().__init__(console=render_console)
        self.config = config
        self.lock = threading.RLock()

        self.rows = _settings_rows(config)
        self.selected = 0
        self.scroll_start = 0
        self.flash: tuple[str, str] | None = None
        self.edit: dict | None = None
        self.pending_reset: int | None = None

        self.catalog_refresher = _ModelCatalogRefresher()
        self.catalog = SimpleNamespace(
            request=self._request_catalog_refresh,
            cancel_pending=self.catalog_refresher.cancel_pending,
        )

    # The reader batches typed text only while a field is being edited; the
    # settings list is single-key navigation. (Properties so they track ``edit``
    # without re-reading on every key dispatch.)
    @property
    def text_mode(self) -> bool:  # type: ignore[override]
        return self.edit is not None

    @property
    def batch_text(self) -> bool:  # type: ignore[override]
        return self.edit is not None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def on_stop(self) -> None:
        self.catalog_refresher.close()

    # ------------------------------------------------------------------ #
    # Background model-catalog refresher (the one concurrent producer)
    # ------------------------------------------------------------------ #
    def _request_catalog_refresh(
        self,
        provider: str,
        *,
        target_edit: dict | None = None,
        delay: float = 0,
    ) -> None:
        probe = dict(self.config)
        probe["provider"] = provider
        generation = self.catalog_refresher.request(
            probe, self._catalog_refreshed, delay=delay
        )
        if target_edit is not None:
            target_edit["catalog_generation"] = generation

    def _catalog_refreshed(
        self,
        provider: str,
        choices: list[tuple[str, str]],
        generation: int,
    ) -> None:
        with self.lock:
            apply_catalog_refresh(self.edit, self.config, provider, choices, generation)
            if provider == self.config.get("provider"):
                from .reasoning import reconcile_reasoning_effort

                if reconcile_reasoning_effort(self.config) is not None:
                    save_config(self.config)
                    self.rows = _settings_rows(self.config)
                    self.flash = (
                        "Reasoning effort reset to default for this model.",
                        "yellow",
                    )
        self.invalidate()

    # ------------------------------------------------------------------ #
    # Key handling
    # ------------------------------------------------------------------ #
    def on_key(self, key: str, repeat: int) -> None:
        # Serialize every state mutation against the catalog refresher thread
        # (which also takes ``self.lock``) so a repaint never observes a
        # half-applied change.
        with self.lock:
            if self.edit is not None:
                self._on_edit_key(key, repeat)
            elif self.pending_reset is not None:
                self._on_reset_key(key)
            else:
                self._on_list_key(key, repeat)

    def _on_edit_key(self, key: str, repeat: int) -> None:
        term_w, _term_h = terminal_size(console=self.console)
        self.config, outcome = apply_editor_key(
            self.edit,
            self.config,
            key,
            repeat,
            catalog=self.catalog,
            inner_width=max(1, term_w - 4),
        )
        if outcome.kind == "committed":
            self.rows = _settings_rows(self.config)
            self.edit = None
            self.flash = (outcome.message, outcome.style)
        elif outcome.kind == "cancelled":
            self.edit = None
            self.flash = (outcome.message, outcome.style)
        elif outcome.clear_flash:
            self.flash = None

    def _on_reset_key(self, key: str) -> None:
        row = self.rows[self.pending_reset]
        self.config, message, style = _settings_finish_reset(row, self.config, key)
        self.rows = _settings_rows(self.config)
        self.pending_reset = None
        self.flash = (message, style) if key in ("y", "Y") else None

    def _on_list_key(self, key: str, repeat: int) -> None:
        if key == "ESC":
            self.stop()
        elif key in ("UP", "k"):
            self.selected = max(0, self.selected - repeat)
            self.flash = None
        elif key in ("DOWN", "j"):
            self.selected = min(len(self.rows) - 1, self.selected + repeat)
            self.flash = None
        elif key == "HOME":
            self.selected = 0
            self.flash = None
        elif key == "END":
            self.selected = len(self.rows) - 1
            self.flash = None
        elif key == "ENTER":
            row = self.rows[self.selected]
            quick = _settings_apply_quick(row, self.config)
            if quick is None:
                self.edit = _settings_begin_edit(row, self.config)
                if row["key"] in {"model", "auditor_model"}:
                    self._request_catalog_refresh(
                        str(self.config.get("provider", "openai")),
                        target_edit=self.edit,
                        delay=0.01,
                    )
                self.flash = None
                return
            self.config, message = quick
            self.rows = _settings_rows(self.config)
            self.flash = (message, "green")
        elif key == "r":
            self.pending_reset = self.selected
            self.flash = None

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self) -> RenderableType:
        # Hold the lock so the frame is composed from a consistent snapshot of
        # state the refresher thread may be mutating. Erase a previous, wider
        # frame's stale right border on WSL/ConPTY via wrap_frame.
        with self.lock:
            return wrap_frame(self._render())

    def _footer(self) -> str:
        return "↑↓ select   Enter edit/toggle   r reset   Esc/q exit"

    def _settings_rendered_row_count(self, start: int, end: int) -> int:
        line_count = 0
        last_section = None
        for idx in range(start, end):
            row = self.rows[idx]
            if row["section"] != last_section:
                if idx != start:
                    line_count += 1
                line_count += 1
                last_section = row["section"]
            line_count += 1
        return line_count

    def _settings_window_end(self, start: int, max_lines: int) -> int:
        end = start
        while end < len(self.rows):
            candidate = end + 1
            if candidate > start + 1 and self._settings_rendered_row_count(start, candidate) > max_lines:
                break
            end = candidate
            if self._settings_rendered_row_count(start, end) >= max_lines:
                break
        return end

    def _settings_visible_window(self, max_lines: int) -> tuple[int, int]:
        if not self.rows:
            return 0, 0

        max_lines = max(1, max_lines)
        self.scroll_start = max(0, min(self.scroll_start, len(self.rows) - 1))
        if self.selected < self.scroll_start:
            self.scroll_start = self.selected

        while self.scroll_start < self.selected and self._settings_rendered_row_count(self.scroll_start, self.selected + 1) > max_lines:
            self.scroll_start += 1

        end = self._settings_window_end(self.scroll_start, max_lines)
        while self.selected >= end and self.scroll_start < self.selected:
            self.scroll_start += 1
            end = self._settings_window_end(self.scroll_start, max_lines)

        if end == len(self.rows):
            while self.scroll_start > 0 and self._settings_rendered_row_count(self.scroll_start - 1, end) <= max_lines:
                self.scroll_start -= 1

        return self.scroll_start, self._settings_window_end(self.scroll_start, max_lines)

    def _render_settings_panel(self, height: int) -> Panel:
        term_w, _ = terminal_size(console=self.console)
        width = panel_width(term_w)
        inner_width = max(1, width - 4)
        height = max(3, height)
        show_footer = self.edit is None and height >= 8
        content_rows = max(1, height - 2)
        reserved = 1
        if self.flash is not None:
            reserved += 2
        if show_footer:
            reserved += 2
        body_rows = max(1, content_rows - reserved)
        start, end = self._settings_visible_window(body_rows)

        parts: list = []
        parts.append(Text(_clip_text(f"  showing {start + 1}-{end} of {len(self.rows)}", inner_width), style="dim"))

        last_section = None
        for idx in range(start, end):
            row = self.rows[idx]
            if row["section"] != last_section:
                last_section = row["section"]
                if idx != start:
                    parts.append(Text(""))
                parts.append(Text(f"  {last_section}", style="bold cyan"))

            is_selected = idx == self.selected
            prefix = " › " if is_selected else "   "
            label_width, value_width, _value_start, _description_start = (
                _settings_column_layout(inner_width)
            )
            desc_width = max(0, inner_width - len(prefix) - label_width - value_width - 4)

            line = Text(no_wrap=True, overflow="crop")
            line.append(prefix, style="bold cyan" if is_selected else "")
            label_style = "bold bright_white" if is_selected else "white"
            line.append(f"{_clip_text(row['label'], label_width):<{label_width}}", style=label_style)
            line.append("  ", style="dim")

            value = _settings_value_text(row, self.config, selected=is_selected)
            value_plain = _clip_text(value.plain, value_width)
            line.append(f"{value_plain:<{value_width}}", style=value.style)
            line.append("  ", style="dim")

            desc_style = "bold" if is_selected else "dim"
            line.append(_clip_text(row["desc"], desc_width), style=desc_style)
            parts.append(line)

        if self.flash is not None:
            msg, style = self.flash
            parts.append(Text(""))
            parts.append(Text(_clip_text(f"  {msg}", inner_width), style=style, no_wrap=True, overflow="crop"))

        if show_footer:
            if self.pending_reset is not None:
                footer = _settings_reset_action_bar(
                    self.rows[self.pending_reset],
                    self.config,
                    inner_width,
                )
            else:
                footer = Text(
                    _clip_text(self._footer(), inner_width),
                    style="dim italic",
                    no_wrap=True,
                    overflow="crop",
                )
            append_bottom_footer(parts, height, footer, crop=True)

        return Panel(
            Group(*parts),
            title="[bold bright_white]jarv ▸ settings[/bold bright_white]",
            title_align="left",
            subtitle=f"[dim]{CONFIG_FILE}[/dim]",
            subtitle_align="right",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            width=width,
            height=height,
        )

    def _render(self) -> RenderableType:
        term_w, term_h = terminal_size(console=self.console)
        term_h = max(3, term_h)
        if self.edit is None:
            return self._render_settings_panel(term_h)

        width = panel_width(term_w)
        inner_width = max(1, width - 4)
        desired_editor_height = _settings_desired_editor_height(
            self.edit,
            self.config,
            inner_width,
            term_h,
        )
        settings_min_height = 8
        editor_min_height = 7 if self.edit["row"].get("multiline") else 3

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
            self._render_settings_panel(settings_height),
            render_editor_panel(
                self.edit,
                self.config,
                panel_width=width,
                height=editor_height,
                title=f"edit {self.edit['row']['label']}",
            ),
        )


def run_settings_interactive(config: dict) -> None:
    SettingsApp(config).run()
    console.print("[dim]○ Settings closed.[/dim]")
