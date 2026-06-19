"""Session, archive, and history command screens."""

import sys

from rich import box
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .command_input import _read_key_with_repeats, mouse_capture
from .display import (
    console,
    flatten_headings,
    jarv_panel,
    refresh_on_resize,
    section_rule,
    terminal_size,
)
from .history import (
    SESSIONS_DIR,
    forget_current_session,
    load_history,
    prepare_session_context,
)
from .config import CONFIG_DIR
from . import session_store as _session_store
from .session_browser import cmd_sessions as _browser_cmd_sessions
from .session_render import (
    _history_visual_lines,
    _history_visual_lines_and_anchors,
    _session_row_widths,
    _status_renderable,
    _tool_call_output,
    _tool_call_renderable,
)
ARCHIVE_DIR = CONFIG_DIR / "archive"


def _sync_session_store_paths() -> None:
    _session_store.ARCHIVE_DIR = ARCHIVE_DIR
    _session_store.SESSIONS_DIR = SESSIONS_DIR


def archive_session_files(history_path):
    _sync_session_store_paths()
    return _session_store.archive_session_files(history_path)


def unarchive_session_files(archived_history_path, session_id: str):
    _sync_session_store_paths()
    return _session_store.unarchive_session_files(archived_history_path, session_id)


def delete_session_files(history_path) -> None:
    _sync_session_store_paths()
    _session_store.delete_session_files(history_path)


def cmd_sessions(args: list | None = None) -> None:
    return _browser_cmd_sessions(args)

def cmd_archive() -> None:
    session_context = prepare_session_context()
    history_path = session_context.history_file

    archived_history = archive_session_files(history_path)
    if archived_history is not None:
        console.print(f"[bold cyan]▸[/bold cyan] [dim]Session archived to[/dim] [cyan]{archived_history}[/cyan]")
    else:
        console.print("[dim]○ No history to archive.[/dim]")

    forget_current_session()
    if archived_history is not None:
        console.print("[bold green]✓[/bold green] [green]New session starts on your next message.[/green]")


def cmd_history() -> None:
    session_context = prepare_session_context()
    history = load_history(session_context.history_file)
    if not history:
        console.print("[dim]○ No history yet.[/dim]")
        return

    exchanges = sum(1 for m in history if isinstance(m, dict) and m.get("role") == "user")

    if not sys.stdin.isatty() or not console.is_terminal:
        parts: list = [section_rule("conversation"), Text("")]
        pending_tool_parts: list = []

        def _append_jarv_heading() -> None:
            line = Text()
            line.append("▌ ", style="bold green")
            line.append("Jarv", style="bold green")
            parts.append(line)

        for item_index, m in enumerate(history):
            role = m.get("role")
            if role == "user":
                if pending_tool_parts:
                    _append_jarv_heading()
                    parts.extend(pending_tool_parts)
                    pending_tool_parts.clear()
                line = Text()
                line.append("▌ ", style="bold cyan")
                line.append("You", style="bold cyan")
                parts.append(line)
                parts.append(Text(f"  {m.get('content', '')}"))
                parts.append(Text(""))
            elif role == "assistant":
                content = m.get("content", "")
                if content:
                    _append_jarv_heading()
                    parts.extend(pending_tool_parts)
                    pending_tool_parts.clear()
                    parts.append(Markdown(flatten_headings(content)))
                    parts.append(Text(""))
            elif m.get("type") == "status":
                content = str(m.get("content") or "").strip()
                if content:
                    _append_jarv_heading()
                    parts.extend(pending_tool_parts)
                    pending_tool_parts.clear()
                    parts.append(_status_renderable(m))
            elif m.get("type") == "function_call":
                pending_tool_parts.append(
                    _tool_call_renderable(
                        m,
                        _tool_call_output(
                            history,
                            item_index,
                            m.get("call_id"),
                        ),
                    )
                )
        if pending_tool_parts:
            _append_jarv_heading()
            parts.extend(pending_tool_parts)
        console.print(jarv_panel(Group(*parts), title="history", subtitle=f"{exchanges} exchange(s)"))
        return

    offset = 0
    visual_cache: dict[int, tuple[list[Text], list[int]]] = {}

    def _visual(width: int) -> tuple[list[Text], list[int]]:
        width = max(1, width)
        cached = visual_cache.get(width)
        if cached is None:
            cached = _history_visual_lines_and_anchors(history, width)
            visual_cache[width] = cached
        return cached

    def _lines(width: int) -> list[Text]:
        lines, _ = _visual(width)
        return lines

    def _anchors(width: int) -> list[int]:
        _, anchors = _visual(width)
        return anchors

    def _body_rows() -> int:
        _, term_h = terminal_size(console=console)
        show_footer = term_h >= 6
        return max(1, term_h - 2 - (2 if show_footer else 0))  # panel border + footer

    def _max_offset(width: int) -> int:
        return max(0, len(_lines(width)) - _body_rows())

    def _jump_to_message(delta: int) -> None:
        nonlocal offset
        term_w, _ = terminal_size(console=console)
        width = max(1, term_w - 4)
        anchors = _anchors(width)
        if not anchors:
            return
        if delta < 0:
            candidates = [anchor for anchor in anchors if anchor < offset]
            target = candidates[-1] if candidates else anchors[0]
        else:
            candidates = [anchor for anchor in anchors if anchor > offset]
            target = candidates[0] if candidates else anchors[-1]
        offset = min(_max_offset(width), max(0, target))

    def _render() -> Panel:
        nonlocal offset
        term_w, term_h = terminal_size(console=console)
        panel_width = max(1, term_w)
        show_footer = term_h >= 6
        body = _body_rows()
        inner_width = max(1, panel_width - 4)
        lines = _lines(inner_width)
        total = len(lines)
        max_off = max(0, total - body)
        offset = max(0, min(offset, max_off))
        start = offset
        end = min(total, start + body)

        parts: list = []
        for i in range(start, end):
            parts.append(lines[i])
        if not lines:
            parts.append(Text("  (empty)", style="dim"))

        if show_footer:
            target_rows_before_footer = max(0, term_h - 2 - 2)
            while len(parts) < target_rows_before_footer:
                parts.append(Text(""))
            position = f"{start + 1}–{end} of {total}" if total else "0"
            parts.append(Text(""))
            parts.append(
                Text(
                    f"←→ chat/reply   ↑↓ scroll   PgUp/PgDn   Home/End   q exit   ·   {position}",
                    style="dim italic",
                    no_wrap=True,
                    overflow="crop",
                )
            )

        return Panel(
            Group(*parts),
            title="[bold bright_white]jarv ▸ history[/bold bright_white]",
            title_align="left",
            subtitle=f"[dim]{exchanges} exchange(s)[/dim]",
            subtitle_align="right",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            width=panel_width,
            height=term_h,
        )

    with Live(
        get_renderable=_render,
        console=console,
        screen=True,
        auto_refresh=False,
        transient=False,
        vertical_overflow="crop",
    ) as live, refresh_on_resize(live), mouse_capture():
        while True:
            live.refresh()
            try:
                key, repeat_count = _read_key_with_repeats()
            except KeyboardInterrupt:
                break
            term_w, _ = terminal_size(console=console)
            width = max(1, term_w - 4)
            total = len(_lines(width))
            page = max(1, _body_rows() - 1)
            max_off = max(0, total - _body_rows())
            if key == "ESC":
                break
            elif key == "UP":
                offset = max(0, offset - repeat_count)
            elif key == "DOWN":
                offset = min(max_off, offset + repeat_count)
            elif key == "LEFT":
                for _ in range(repeat_count):
                    _jump_to_message(-1)
            elif key == "RIGHT":
                for _ in range(repeat_count):
                    _jump_to_message(1)
            elif key == "PAGEUP":
                offset = max(0, offset - (page * repeat_count))
            elif key == "PAGEDOWN":
                offset = min(max_off, offset + (page * repeat_count))
            elif key == "HOME":
                offset = 0
            elif key == "END":
                offset = max_off
