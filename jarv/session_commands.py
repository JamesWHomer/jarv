"""Session, archive, and history command screens."""

import sys

from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .display import (
    console,
    flatten_headings,
    jarv_panel,
    section_rule,
    terminal_size,
)
from .tui_layout import append_bottom_footer
from .tui_overlay import (
    ScrollOverlayState,
    apply_scroll_keys,
    body_content_rows,
    clamp_scroll_offset,
    run_scroll_live,
    scroll_position_hint,
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

    visual_cache: dict[int, tuple[list[Text], list[int]]] = {}
    state = ScrollOverlayState()

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

    def _jump_to_message(delta: int) -> None:
        term_w, term_h = terminal_size(console=console)
        width = max(1, term_w - 4)
        anchors = _anchors(width)
        if not anchors:
            return
        if delta < 0:
            candidates = [anchor for anchor in anchors if anchor < state.offset]
            target = candidates[-1] if candidates else anchors[0]
        else:
            candidates = [anchor for anchor in anchors if anchor > state.offset]
            target = candidates[0] if candidates else anchors[-1]
        body_rows, _ = body_content_rows(term_h)
        state.offset = clamp_scroll_offset(target, len(_lines(width)), body_rows)

    def _render() -> Panel:
        term_w, term_h = terminal_size(console=console)
        panel_width = max(1, term_w)
        body_rows, show_footer = body_content_rows(term_h)
        inner_width = max(1, panel_width - 4)
        lines = _lines(inner_width)
        total = len(lines)
        state.offset = clamp_scroll_offset(state.offset, total, body_rows)
        start = state.offset
        end = min(total, start + body_rows)

        parts: list = []
        for index in range(start, end):
            parts.append(lines[index])
        if not lines:
            parts.append(Text("  (empty)", style="dim"))

        if show_footer:
            position = scroll_position_hint(start, end, total)
            append_bottom_footer(
                parts,
                term_h,
                Text(
                    f"←→ chat/reply   ↑↓ scroll   PgUp/PgDn   Home/End   q exit   ·   {position}",
                    style="dim italic",
                    no_wrap=True,
                    overflow="crop",
                ),
            )

        return jarv_panel(
            Group(*parts),
            "history",
            subtitle=f"{exchanges} exchange(s)",
            padding=(0, 1),
            width=panel_width,
            height=term_h,
        )

    def _on_key(key: str, repeat_count: int, scroll_state: ScrollOverlayState) -> bool:
        if key == "LEFT":
            for _ in range(repeat_count):
                _jump_to_message(-1)
            return False
        if key == "RIGHT":
            for _ in range(repeat_count):
                _jump_to_message(1)
            return False
        term_w, term_h = terminal_size(console=console)
        width = max(1, term_w - 4)
        total = len(_lines(width))
        body_rows, _ = body_content_rows(term_h)
        scroll_state.offset = apply_scroll_keys(
            key,
            repeat_count,
            offset=scroll_state.offset,
            total=total,
            body_rows=body_rows,
        )
        return False

    run_scroll_live(_render, _on_key, state=state, close_keys=frozenset({"ESC"}))
