"""Session, archive, and history command screens."""

import sys
import threading
import time
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .command_input import _read_key_with_repeats, mouse_capture
from .config import CONFIG_DIR
from .display import (
    console,
    flatten_headings,
    jarv_panel,
    refresh_on_resize,
    rendered_text_lines,
    section_rule,
    terminal_size,
)
from .history import (
    SESSIONS_DIR,
    artifact_file_for,
    detect_terminal,
    forget_current_session,
    history_file_for_session,
    isoformat_utc,
    load_history,
    load_sessions,
    parse_timestamp,
    prepare_session_context,
    reads_file_for,
    redo_file_for,
    save_sessions,
    set_terminal_session,
    utc_now,
)
from .usage import usage_file_for

ARCHIVE_DIR = CONFIG_DIR / "archive"


def archive_session_files(history_path: Path) -> Path | None:
    """Move history and sidecars for a session into ARCHIVE_DIR.

    Returns the new archived history path, or None if nothing was archived.
    """
    if not history_path.exists() or not load_history(history_path):
        return None
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    cleared_at = utc_now().strftime("%Y%m%dT%H%M%SZ")
    stem_suffix = history_path.stem[len("history"):]
    archived_history = ARCHIVE_DIR / f"history-{cleared_at}{stem_suffix}.json"
    history_path.rename(archived_history)

    artifact_path = artifact_file_for(history_path)
    if artifact_path.exists():
        artifact_path.rename(ARCHIVE_DIR / f"artifacts-{cleared_at}{stem_suffix}.json")

    reads_path = reads_file_for(history_path)
    if reads_path.exists():
        reads_path.rename(ARCHIVE_DIR / f"reads-{cleared_at}{stem_suffix}.json")

    usage_path = usage_file_for(history_path)
    if usage_path.exists():
        usage_path.rename(ARCHIVE_DIR / f"usage-{cleared_at}{stem_suffix}.json")

    redo_path = redo_file_for(history_path)
    if redo_path.exists():
        redo_path.unlink()

    return archived_history


def unarchive_session_files(archived_history_path: Path, session_id: str) -> Path | None:
    """Reverse archive_session_files for the given session id."""
    if not archived_history_path.exists():
        return None
    restored_history = history_file_for_session(session_id)
    archived_history_path.rename(restored_history)

    archived_dir = archived_history_path.parent
    archived_tail = archived_history_path.stem[len("history"):]  # "-{ts}-{hash}"
    restored_suffix = restored_history.stem[len("history"):]  # "-{hash}"
    for kind in ("artifacts", "reads", "usage"):
        sib = archived_dir / f"{kind}{archived_tail}.json"
        if sib.exists():
            sib.rename(SESSIONS_DIR / f"{kind}{restored_suffix}.json")
    return restored_history


def _history_content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text" and isinstance(c.get("text"), str):
                    chunks.append(c["text"])
                elif "content" in c and isinstance(c["content"], str):
                    chunks.append(c["content"])
                else:
                    chunks.append(f"[{c.get('type', 'item')}]")
            else:
                chunks.append(str(c))
        return "\n".join(chunks)
    return str(content)


def _markdown_to_text_lines(content: str, width: int) -> list[Text]:
    return rendered_text_lines(Markdown(flatten_headings(content)), width)


def _history_visual_lines_and_anchors(history: list, width: int) -> tuple[list[Text], list[int]]:
    lines: list[Text] = []
    anchors: list[int] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).lower()
        if role == "system":
            continue
        body = _history_content_to_str(item.get("content", "")).strip()
        if not body:
            continue
        start = len(lines)
        if role == "user":
            for j, raw in enumerate(body.splitlines() or [""]):
                t = Text(no_wrap=False, overflow="fold")
                if j == 0:
                    t.append("user: ", style="bold cyan")
                else:
                    t.append("  ")
                t.append(raw, style="bold")
                lines.extend(rendered_text_lines(t, width))
        elif role == "assistant":
            lines.append(Text("jarv:", style="bold green", no_wrap=True, overflow="crop"))
            lines.extend(_markdown_to_text_lines(body, width))
        else:
            label = role or "?"
            for j, raw in enumerate(body.splitlines() or [""]):
                t = Text(no_wrap=False, overflow="fold")
                if j == 0:
                    t.append(f"{label}: ", style="dim")
                else:
                    t.append("  ")
                t.append(raw, style="dim")
                lines.extend(rendered_text_lines(t, width))
        if len(lines) > start:
            anchors.append(start)
        lines.append(Text(""))
    if lines and lines[-1].plain == "":
        lines.pop()
    return lines, anchors


def _history_visual_lines(history: list, width: int) -> list[Text]:
    lines, _ = _history_visual_lines_and_anchors(history, width)
    return lines


def delete_session_files(history_path: Path) -> None:
    """Permanently remove history and sidecars for a session."""
    for path in (
        history_path,
        artifact_file_for(history_path),
        reads_file_for(history_path),
        usage_file_for(history_path),
        redo_file_for(history_path),
    ):
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


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



def _short_session_id(sid: str) -> str:
    """Return the shortest unambiguous prefix hint for display (type prefix + 6 hash chars)."""
    # IDs look like: parent-5d44fec1a0fe  or  windows-terminal-3dece1d0fac8
    # Keep the descriptive prefix and show only 6 chars of the trailing hash.
    parts = sid.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) >= 6:
        return f"{parts[0]}-{parts[1][:6]}"
    return sid[:16]


def _session_row_widths(width: int) -> tuple[int, int, int]:
    """Allocate session, date, and message columns within a row."""
    date_width = min(7, max(0, width))
    if width <= date_width:
        return (0, date_width, 0)

    gutter_width = 2
    message_min_width = 16
    fixed_width = date_width + (2 * gutter_width)
    session_width = min(28, max(0, width - fixed_width - message_min_width))
    message_width = max(0, width - session_width - fixed_width)
    return (session_width, date_width, message_width)


def _sessions_plain(sessions: dict, terminals: dict) -> None:
    """Non-interactive fallback session list (used when stdout is not a tty)."""
    terminal_id, _ = detect_terminal()
    current_session_id = terminals.get(terminal_id)
    now = utc_now()

    def sort_key(sid: str) -> str:
        meta = sessions[sid]
        return meta.get("last_message_at") or meta.get("last_used_at") or ""

    sorted_sessions = sorted(sessions.keys(), key=sort_key, reverse=True)[:5]

    table = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 2), header_style="bold cyan", pad_edge=False)
    table.add_column("", no_wrap=True, width=1)
    table.add_column("ID prefix", style="bold cyan", no_wrap=True)
    table.add_column("Last active", style="dim", no_wrap=True)
    table.add_column("First message")

    for sid in sorted_sessions:
        meta = sessions[sid]
        ts_str = meta.get("last_message_at") or meta.get("last_used_at")
        ts = parse_timestamp(ts_str)
        if ts:
            delta = now - ts
            secs = int(delta.total_seconds())
            if secs < 60:
                time_str = "just now"
            elif secs < 3600:
                time_str = f"{secs // 60}m ago"
            elif secs < 86400:
                time_str = f"{secs // 3600}h ago"
            elif secs < 7 * 86400:
                time_str = f"{secs // 86400}d ago"
            else:
                time_str = ts.strftime("%b %d")
        else:
            time_str = "—"

        snippet = ""
        history_path_str = meta.get("history_file")
        if history_path_str:
            history_path = Path(history_path_str)
            if history_path.exists():
                history = load_history(history_path)
                for item in history:
                    if isinstance(item, dict) and item.get("role") == "user":
                        content = str(item.get("content", "")).replace("\n", " ").strip()
                        if content:
                            snippet = content[:72] + ("..." if len(content) > 72 else "")
                            break

        marker = "[green]●[/green]" if sid == current_session_id else ""
        table.add_row(marker, _short_session_id(sid), time_str, snippet or "[dim]no messages[/dim]")

    total = len(sessions)
    shown = len(sorted_sessions)
    footer_parts: list = [table]
    if total > shown:
        footer_parts += [Text(""), Text(f"Showing {shown} most recent of {total} sessions.", style="dim")]
    footer_parts += [Text("Run jarv /sessions <id> to switch to a session.", style="dim italic")]
    console.print(jarv_panel(Group(*footer_parts), title="sessions", subtitle=f"{shown}/{total}"))


def _cmd_sessions_load(prefix: str) -> None:
    data = load_sessions()
    sessions = data["sessions"]
    if not sessions:
        console.print("[yellow]No sessions exist yet.[/yellow]")
        return
    if prefix in sessions:
        session_id = prefix
    else:
        matches = [sid for sid in sessions if sid.startswith(prefix)]
        if not matches:
            console.print(f"[bold red]✗[/bold red] [red]No session matches:[/red] [bold]{prefix}[/bold]")
            console.print("[dim]  Run [bold]jarv /sessions[/bold] to see available sessions.[/dim]")
            return
        if len(matches) > 1:
            console.print(f"[bold yellow]?[/bold yellow] [yellow]Ambiguous prefix[/yellow] [bold]{prefix}[/bold] [dim]matches {len(matches)} sessions:[/dim]")
            for m in matches:
                console.print(f"  [dim]•[/dim] [cyan]{m}[/cyan]")
            return
        session_id = matches[0]
    set_terminal_session(session_id)
    label = sessions[session_id].get("label", session_id)
    console.print(f"[bold green]✓[/bold green] [green]Loaded[/green] [bold cyan]{_short_session_id(session_id)}[/bold cyan] [dim]({label})[/dim]")


def cmd_sessions(args: list | None = None) -> None:
    if args:
        _cmd_sessions_load(args[0])
        return
    data = load_sessions()
    sessions = data["sessions"]
    terminals = data["terminals"]

    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        console.print("[dim]Sessions are created automatically when you start chatting.[/dim]")
        return

    if not sys.stdin.isatty() or not console.is_terminal:
        _sessions_plain(sessions, terminals)
        return

    terminal_id, _ = detect_terminal()
    current_session_id = terminals.get(terminal_id)
    now = utc_now()

    def sort_key(sid: str) -> str:
        meta = sessions[sid]
        return meta.get("last_message_at") or meta.get("last_used_at") or ""

    sorted_sessions = sorted(sessions.keys(), key=sort_key, reverse=True)

    # Precompute cheap display data only. History and usage sidecars are loaded
    # lazily for visible rows so first paint is bounded by viewport size.
    rows: list[dict] = []
    for sid in sorted_sessions:
        meta = sessions[sid]
        ts_str = meta.get("last_message_at") or meta.get("last_used_at")
        ts = parse_timestamp(ts_str)
        if ts:
            delta = now - ts
            secs = int(delta.total_seconds())
            if secs < 60:
                time_str = "just now"
            elif secs < 3600:
                time_str = f"{secs // 60}m ago"
            elif secs < 86400:
                time_str = f"{secs // 3600}h ago"
            elif secs < 7 * 86400:
                time_str = f"{secs // 86400}d ago"
            else:
                time_str = ts.strftime("%b %d")
        else:
            time_str = "—"

        cached_snippet = meta.get("first_user_snippet")
        if not isinstance(cached_snippet, str):
            cached_snippet = meta.get("first_message")
        if not isinstance(cached_snippet, str):
            cached_snippet = ""
        snippet = cached_snippet[:60] + ("..." if len(cached_snippet) > 60 else "")

        rows.append({
            "sid": sid,
            "short_id": _short_session_id(sid),
            "time_str": time_str,
            "snippet": snippet,
            "snippet_loaded": bool(cached_snippet),
            "is_current": sid == current_session_id,
            "archived": bool(meta.get("archived")),
        })

    view_mode = "active"  # "active" | "all" | "archived"
    arm_delete_sid: str | None = None
    flash: tuple[str, str] | None = None  # (message, style) shown above the footer
    search_query: str = ""
    search_active: bool = False  # input bar focused for typing
    search_text_cache: dict[str, str] = {}  # sid -> lowercased transcript text
    last_action: dict | None = None  # most recent undoable action (5s window)
    undo_lock = threading.Lock()
    UNDO_WINDOW = 5.0
    # When a row is archived/unarchived from a filtered view, keep it visible in
    # place (with its new aesthetic) until the cursor moves.
    ghost_sid: str | None = None
    selected_sid: str | None = next(
        (r["sid"] for r in rows if r["is_current"] and not r["archived"]),
        next((r["sid"] for r in rows if not r["archived"]), rows[0]["sid"] if rows else None),
    )
    offset = 0
    preview_sid: str | None = None
    preview_offset = 0
    preview_cache: dict[tuple[str, int], list[Text]] = {}  # (sid, width) -> pre-built visual lines

    def _truncate(value: str, width: int) -> str:
        if width <= 0:
            return ""
        if len(value) <= width:
            return value
        if width <= 3:
            return value[:width]
        return value[:width - 3] + "..."

    def _content_rows(term_h: int, has_status: bool, show_footer: bool, has_search: bool = False) -> int:
        # Panel border = 2 rows. Header consumes 1 row. Footer = 2 rows (blank + controls).
        content = max(1, term_h - 2 - 1)
        if show_footer:
            content -= 2
        if has_status:
            content -= 1
        if has_search:
            content -= 1
        return max(1, content)

    def _max_vis(has_status: bool = False) -> int:
        _, term_h = terminal_size(console=console)
        has_search = bool(search_active or search_query)
        return _content_rows(term_h, has_status, show_footer=term_h >= 6, has_search=has_search)

    def _fast_search_text(r: dict) -> str:
        # Cheap fields available without disk I/O — exact short id, full sid,
        # the user's first-message snippet, and the session label.
        meta = sessions.get(r["sid"], {})
        label = meta.get("label", "") if isinstance(meta.get("label"), str) else ""
        return f"{r['short_id']} {r['sid']} {r.get('snippet', '')} {label}".lower()

    def _first_user_snippet(meta: dict, width: int = 60) -> str:
        hp_str = meta.get("history_file")
        if not hp_str:
            return ""
        hp = Path(hp_str)
        if not hp.exists():
            return ""
        try:
            history = load_history(hp)
        except Exception:
            return ""
        for item in history:
            if isinstance(item, dict) and item.get("role") == "user":
                content = str(item.get("content", "")).replace("\n", " ").strip()
                if content:
                    return content[:width] + ("..." if len(content) > width else "")
        return ""

    def _ensure_row_metadata(r: dict) -> None:
        meta = sessions.get(r["sid"], {})
        if not r.get("snippet_loaded"):
            r["snippet"] = _first_user_snippet(meta)
            r["snippet_loaded"] = True

    def _build_search_text(sid: str) -> str:
        meta = sessions.get(sid, {})
        hp_str = meta.get("history_file")
        chunks: list[str] = []
        label = meta.get("label")
        if isinstance(label, str):
            chunks.append(label)
        if hp_str:
            hp = Path(hp_str)
            if hp.exists():
                try:
                    history = load_history(hp)
                except Exception:
                    history = []
                for item in history:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content", "")
                    if isinstance(content, str):
                        chunks.append(content)
                    elif isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict):
                                t = c.get("text") if c.get("type") == "text" else c.get("content")
                                if isinstance(t, str):
                                    chunks.append(t)
        return "\n".join(chunks).lower()

    def _search_text(sid: str) -> str:
        cached = search_text_cache.get(sid)
        if cached is not None:
            return cached
        text = _build_search_text(sid)
        search_text_cache[sid] = text
        return text

    def _prefetch_worker() -> None:
        for r in rows:
            if prefetch_stop.is_set():
                return
            sid = r["sid"]
            if sid in search_text_cache:
                continue
            try:
                text = _build_search_text(sid)
            except Exception:
                text = ""
            search_text_cache[sid] = text

    prefetch_stop = threading.Event()
    prefetch_thread: threading.Thread | None = None

    def _start_prefetch() -> None:
        nonlocal prefetch_thread
        if prefetch_thread is not None:
            return
        prefetch_thread = threading.Thread(target=_prefetch_worker, daemon=True)
        prefetch_thread.start()

    def _visible_rows_list() -> list[dict]:
        q = search_query.lower().strip()
        def keep(r: dict) -> bool:
            if r["sid"] == ghost_sid:
                return True
            if view_mode == "active" and r["archived"]:
                return False
            if view_mode == "archived" and not r["archived"]:
                return False
            if q:
                if q in _fast_search_text(r):
                    return True
                if q not in _search_text(r["sid"]):
                    return False
            return True
        return [r for r in rows if keep(r)]

    def _selected_pos(visible: list[dict]) -> int:
        for i, r in enumerate(visible):
            if r["sid"] == selected_sid:
                return i
        return 0

    def _clamp_offset(sel: int, off: int, mv: int, n: int) -> int:
        if n == 0:
            return 0
        if sel < off:
            return sel
        if sel >= off + mv:
            return sel - mv + 1
        return max(0, min(off, max(0, n - mv)))

    def _subtitle() -> str:
        n_active = sum(1 for r in rows if not r["archived"])
        n_archived = len(rows) - n_active
        if view_mode == "active":
            return f"[dim]{n_active} active[/dim]"
        if view_mode == "archived":
            return f"[dim]{n_archived} archived[/dim]"
        return f"[dim]{n_active} active · {n_archived} archived[/dim]"

    def _footer_text() -> str:
        if search_active:
            return "type to filter   Enter apply   Esc cancel   Backspace delete"
        cur_visible = _visible_rows_list()
        cur = cur_visible[_selected_pos(cur_visible)] if cur_visible else None
        a_hint = "a unarchive" if (cur and cur["archived"]) else "a archive"
        find_hint = "^F edit search" if search_query else "^F find"
        parts = [
            "←→/↑↓ navigate", "Enter load", "p preview", "d delete",
            a_hint, f"Tab view: {view_mode}", find_hint,
        ]
        action = last_action
        if action is not None:
            remaining = action["deadline"] - time.time()
            if remaining > 0:
                parts.append(f"u undo ({int(remaining) + 1}s)")
        parts.append("q cancel")
        return "   ".join(parts)

    def _build_preview_lines(sid: str, width: int) -> list[Text]:
        meta = sessions.get(sid, {})
        hp_str = meta.get("history_file")
        if not hp_str:
            return [Text("(no history file)", style="dim")]
        hp = Path(hp_str)
        if not hp.exists():
            return [Text("(history file missing)", style="dim")]
        history = load_history(hp)
        if not history:
            return [Text("(empty conversation)", style="dim")]
        return _history_visual_lines(history, width) or [Text("(empty conversation)", style="dim")]

    def _preview_lines(sid: str, width: int) -> list[Text]:
        cache_key = (sid, width)
        if cache_key not in preview_cache:
            preview_cache[cache_key] = _build_preview_lines(sid, width)
        return preview_cache[cache_key]

    def _append_bottom_footer(parts: list, term_h: int, footer: Text) -> None:
        footer_rows = 2  # spacer + controls
        target_rows_before_footer = max(0, term_h - 2 - footer_rows)
        while len(parts) < target_rows_before_footer:
            parts.append(Text(""))
        parts.append(Text(""))
        parts.append(footer)

    def _render_preview() -> Panel:
        nonlocal preview_offset
        term_w, term_h = terminal_size(console=console)
        panel_width = max(1, term_w)
        inner_width = max(1, panel_width - 4)
        show_footer = term_h >= 6
        # Header (1 row) + footer (2 rows: blank + controls) inside the panel border (2 rows).
        body_rows = max(1, term_h - 2 - 1 - (2 if show_footer else 0))

        sid = preview_sid or ""
        all_lines = _preview_lines(sid, inner_width)
        total = len(all_lines)
        max_off = max(0, total - body_rows)
        if preview_offset > max_off:
            preview_offset = max_off
        if preview_offset < 0:
            preview_offset = 0
        start = preview_offset
        end = min(total, start + body_rows)

        meta = sessions.get(sid, {})
        short_id = _short_session_id(sid) if sid else ""
        label = meta.get("label", "")

        parts: list = []
        parts.append(
            Text(
                _truncate(f"  {short_id}  {label}", inner_width),
                style="dim",
                no_wrap=True,
                overflow="crop",
            )
        )
        for i in range(start, end):
            parts.append(all_lines[i])
        if total == 0:
            parts.append(Text(_truncate("  (empty)", inner_width), style="dim"))

        if show_footer:
            position = f"{start + 1}–{end} of {total}" if total else "0"
            _append_bottom_footer(
                parts,
                term_h,
                Text(
                    _truncate(
                        f"↑↓ scroll   ←→ session   Enter load   p/Esc back   ·   {position}",
                        inner_width,
                    ),
                    style="dim italic",
                    no_wrap=True,
                    overflow="crop",
                ),
            )

        return Panel(
            Group(*parts),
            title="[bold bright_white]jarv ▸ preview[/bold bright_white]",
            title_align="left",
            subtitle=f"[dim]{short_id}[/dim]" if short_id else None,
            subtitle_align="right",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            width=panel_width,
            height=term_h,
        )

    def _render() -> Panel:
        if preview_sid is not None:
            return _render_preview()
        nonlocal offset
        term_w, term_h = terminal_size(console=console)
        panel_width = max(1, term_w)
        inner_width = max(1, panel_width - 4)
        show_footer = term_h >= 6

        visible = _visible_rows_list()
        n = len(visible)
        sel = _selected_pos(visible)

        status: Text | None = None
        cur = visible[sel] if visible else None
        if arm_delete_sid and cur and cur["sid"] == arm_delete_sid:
            prompt = (
                f"Delete {cur['short_id']} permanently? "
                "Press d again to confirm · any other key cancels"
            )
            status = Text(_truncate(prompt, inner_width), style="bold red", no_wrap=True, overflow="crop")
        elif flash is not None:
            msg, style = flash
            status = Text(_truncate(msg, inner_width), style=style, no_wrap=True, overflow="crop")

        has_search = bool(search_active or search_query)
        mv = _content_rows(
            term_h,
            has_status=status is not None,
            show_footer=show_footer,
            has_search=has_search,
        )
        offset = _clamp_offset(sel, offset, mv, n)
        start = offset
        end = min(n, offset + mv)

        def _search_bar() -> Text:
            cursor = "▌" if search_active else ""
            shown = search_query + cursor
            prefix = " › " if search_active else "   "
            label_style = "bold cyan" if search_active else "cyan"
            value_style = "bold cyan" if search_active else "bold"
            placeholder_style = "bold cyan" if search_active else "dim italic"
            line = Text(no_wrap=True, overflow="crop")
            line.append(prefix, style="bold cyan" if search_active else "")
            line.append("find: ", style=label_style)
            avail = max(0, inner_width - len(prefix) - 6)
            if shown:
                line.append(_truncate(shown, avail), style=value_style)
            else:
                line.append(_truncate("(type to filter transcripts)", avail), style=placeholder_style)
            return line

        parts: list = []
        if n == 0:
            if has_search:
                parts.append(_search_bar())
            empty_msg = (
                f"  (no sessions match \"{search_query}\")"
                if search_query
                else "  (no sessions in this view)"
            )
            parts.append(Text(_truncate(empty_msg, inner_width), style="dim"))
            if status is not None:
                parts.append(status)
            if show_footer:
                _append_bottom_footer(
                    parts,
                    term_h,
                    Text(
                        _truncate(_footer_text(), inner_width),
                        style="dim italic",
                        no_wrap=True,
                        overflow="crop",
                    ),
                )
            return Panel(
                Group(*parts),
                title="[bold bright_white]jarv ▸ sessions[/bold bright_white]",
                title_align="left",
                subtitle=_subtitle(),
                subtitle_align="right",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 1),
                width=panel_width,
                height=term_h,
            )

        parts.append(
            Text(
                _truncate(f"  showing {start + 1}–{end} of {n}", inner_width),
                style="dim",
                no_wrap=True,
                overflow="crop",
            )
        )
        if has_search:
            parts.append(_search_bar())

        for i in range(start, end):
            r = visible[i]
            _ensure_row_metadata(r)
            is_sel = (i == sel) and not search_active
            is_armed = is_sel and arm_delete_sid == r["sid"]
            t = Text(no_wrap=True, overflow="ellipsis")
            prefix = " › " if is_sel else "   "
            if r["is_current"]:
                marker = "●  "
            elif r["archived"]:
                marker = "⌫  "
            else:
                marker = "   "
            remaining = max(0, inner_width - len(prefix) - len(marker))
            id_width, time_width, snippet_width = _session_row_widths(remaining)

            if is_armed:
                prefix_style = "bold red"
            elif is_sel:
                prefix_style = "bold cyan"
            else:
                prefix_style = ""
            t.append(_truncate(prefix, inner_width), style=prefix_style)

            if inner_width > len(prefix):
                if is_armed:
                    marker_style = "bold red"
                elif r["is_current"]:
                    marker_style = "green"
                elif r["archived"]:
                    marker_style = "dim"
                else:
                    marker_style = ""
                t.append(_truncate(marker, inner_width - len(prefix)), style=marker_style)

            if id_width:
                short_id = _truncate(r["short_id"], id_width)
                if is_armed:
                    id_style = "bold red"
                elif is_sel:
                    id_style = "bold cyan"
                elif r["archived"]:
                    id_style = "dim cyan"
                else:
                    id_style = "cyan"
                t.append(f"{short_id:<{id_width}}", style=id_style)
                t.append("  ")

            if time_width:
                time_str = _truncate(r["time_str"], time_width)
                if is_armed:
                    time_style = "bold red"
                elif is_sel:
                    time_style = "bold"
                else:
                    time_style = "dim"
                t.append(f"{time_str:<{time_width}}", style=time_style)
                t.append("  ")

            snip = r["snippet"] or "no messages"
            if snippet_width:
                if is_armed:
                    snip_style = "bold red"
                elif is_sel:
                    snip_style = "bold" if not r["archived"] else "dim strike"
                elif r["archived"]:
                    snip_style = "dim strike"
                else:
                    snip_style = "dim"
                t.append(_truncate(snip, snippet_width), style=snip_style)
            parts.append(t)

        if status is not None:
            parts.append(status)

        if show_footer:
            _append_bottom_footer(
                parts,
                term_h,
                Text(
                    _truncate(_footer_text(), inner_width),
                    style="dim italic",
                    no_wrap=True,
                    overflow="crop",
                ),
            )

        return Panel(
            Group(*parts),
            title="[bold bright_white]jarv \u25b8 sessions[/bold bright_white]",
            title_align="left",
            subtitle=_subtitle(),
            subtitle_align="right",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            width=panel_width,
            height=term_h,
        )

    def _finalize_action(action: dict) -> None:
        """Apply the irreversible part of an action that's leaving its window."""
        if action["kind"] == "did_delete":
            hp_str = action.get("history_path")
            if hp_str:
                delete_session_files(Path(hp_str))

    def _expire_action(action: dict) -> None:
        nonlocal last_action
        with undo_lock:
            if last_action is not action:
                return
            last_action = None
        _finalize_action(action)

    def _commit_pending() -> None:
        nonlocal last_action
        with undo_lock:
            action = last_action
            last_action = None
        if action is None:
            return
        t = action.get("timer")
        if t is not None:
            t.cancel()
        _finalize_action(action)

    def _start_undo(action: dict) -> None:
        nonlocal last_action
        _commit_pending()
        action["deadline"] = time.time() + UNDO_WINDOW
        timer = threading.Timer(UNDO_WINDOW, _expire_action, args=(action,))
        timer.daemon = True
        action["timer"] = timer
        with undo_lock:
            last_action = action
        timer.start()

    def _take_last_action() -> dict | None:
        """Atomically grab and clear the current undoable action if still valid."""
        nonlocal last_action
        with undo_lock:
            action = last_action
            if action is None:
                return None
            if time.time() >= action["deadline"]:
                # Already expired — let the timer handle finalization.
                return None
            last_action = None
        t = action.get("timer")
        if t is not None:
            t.cancel()
        return action

    def _do_undo() -> tuple[tuple[str, str], str | None] | None:
        """Returns ((flash_msg, flash_style), restored_sid) or None."""
        action = _take_last_action()
        if action is None:
            return None
        kind = action["kind"]
        if kind == "did_archive":
            sid = action["sid"]
            row = next((r for r in rows if r["sid"] == sid), None)
            if row is None:
                return (("○ session no longer exists", "dim"), None)
            meta = sessions.get(sid, {})
            hp_str = meta.get("history_file")
            hp = Path(hp_str) if hp_str else None
            restored = unarchive_session_files(hp, sid) if hp else None
            if restored is not None:
                meta["history_file"] = str(restored)
            meta.pop("archived", None)
            meta.pop("archived_at", None)
            row["archived"] = False
            save_sessions(data)
            return ((f"↺ restored {row['short_id']}", "green"), sid)
        if kind == "did_unarchive":
            sid = action["sid"]
            row = next((r for r in rows if r["sid"] == sid), None)
            if row is None:
                return (("○ session no longer exists", "dim"), None)
            meta = sessions.get(sid, {})
            hp_str = meta.get("history_file")
            hp = Path(hp_str) if hp_str else None
            archived_path = archive_session_files(hp) if hp else None
            if archived_path is None:
                return (("○ couldn't re-archive", "dim"), None)
            meta["history_file"] = str(archived_path)
            meta["archived"] = True
            meta["archived_at"] = isoformat_utc(utc_now())
            for term_id, mapped_sid in list(terminals.items()):
                if mapped_sid == sid:
                    terminals.pop(term_id)
            row["archived"] = True
            row["is_current"] = False
            save_sessions(data)
            return ((f"↺ archived {row['short_id']}", "cyan"), sid)
        if kind == "did_delete":
            sid = action["sid"]
            snapshot_row = action["row"]
            snapshot_meta = action["meta"]
            row_index = action.get("row_index", len(rows))
            removed_terminals = action.get("removed_terminals", [])
            sessions[sid] = snapshot_meta
            for term_id in removed_terminals:
                terminals[term_id] = sid
            if 0 <= row_index <= len(rows):
                rows.insert(row_index, snapshot_row)
            else:
                rows.append(snapshot_row)
            save_sessions(data)
            return ((f"↺ restored {snapshot_row['short_id']}", "green"), sid)
        return None

    loaded_row: dict | None = None
    auto_restored = False
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
                text_mode = search_active and preview_sid is None
                key, repeat_count = _read_key_with_repeats(
                    text_mode=text_mode,
                    repeatable=() if text_mode else ("UP", "DOWN", "LEFT", "RIGHT", "PAGEUP", "PAGEDOWN"),
                )
            except KeyboardInterrupt:
                break

            # Search-input mode intercepts most keys (only outside preview).
            if search_active and preview_sid is None:
                if key == "ESC":
                    search_active = False
                    search_query = ""
                    offset = 0
                elif key in ("ENTER", "DOWN"):
                    # Exit search mode, drop focus into the filtered list.
                    search_active = False
                    visible_now = _visible_rows_list()
                    if visible_now and not any(r["sid"] == selected_sid for r in visible_now):
                        selected_sid = visible_now[0]["sid"]
                    offset = 0
                elif key == "BACKSPACE":
                    if search_query:
                        search_query = search_query[:-1]
                        offset = 0
                elif key == "CTRL_F":
                    _start_prefetch()
                    search_active = False
                elif isinstance(key, str) and len(key) == 1 and key.isprintable():
                    search_query += key
                    offset = 0
                    visible_now = _visible_rows_list()
                    if visible_now and not any(r["sid"] == selected_sid for r in visible_now):
                        selected_sid = visible_now[0]["sid"]
                # All other keys (UP, PAGEUP/DOWN, HOME/END, TAB, a, d, p, …)
                # are swallowed while typing the query.
                continue

            # Preview mode intercepts most keys.
            if preview_sid is not None:
                if key in ("p", "ESC"):
                    preview_sid = None
                    preview_offset = 0
                elif key in ("LEFT", "RIGHT"):
                    visible_now = _visible_rows_list()
                    if visible_now:
                        pos = next(
                            (i for i, r in enumerate(visible_now) if r["sid"] == preview_sid),
                            _selected_pos(visible_now),
                        )
                        delta = -repeat_count if key == "LEFT" else repeat_count
                        pos = max(0, min(len(visible_now) - 1, pos + delta))
                        preview_sid = visible_now[pos]["sid"]
                        selected_sid = preview_sid
                        preview_offset = 0
                elif key == "UP":
                    preview_offset = max(0, preview_offset - repeat_count)
                elif key == "DOWN":
                    preview_offset += repeat_count
                elif key == "PAGEUP":
                    preview_offset = max(0, preview_offset - (_max_vis() * repeat_count))
                elif key == "PAGEDOWN":
                    preview_offset += _max_vis() * repeat_count
                elif key == "HOME":
                    preview_offset = 0
                elif key == "END":
                    term_w, _ = terminal_size(console=console)
                    preview_offset = max(0, len(_preview_lines(preview_sid, max(1, term_w - 4))) - 1)
                elif key == "ENTER":
                    row = next((r for r in rows if r["sid"] == preview_sid), None)
                    if row is not None:
                        if row["archived"]:
                            meta = sessions.get(row["sid"], {})
                            hp_str = meta.get("history_file")
                            if hp_str:
                                restored = unarchive_session_files(Path(hp_str), row["sid"])
                                if restored is not None:
                                    meta["history_file"] = str(restored)
                                    meta.pop("archived", None)
                                    meta.pop("archived_at", None)
                                    row["archived"] = False
                                    save_sessions(data)
                                    auto_restored = True
                        set_terminal_session(row["sid"])
                        loaded_row = row
                        break
                continue

            if key != "d":
                arm_delete_sid = None
            flash = None

            visible = _visible_rows_list()
            n_vis = len(visible)
            sel = _selected_pos(visible) if visible else 0
            cur = visible[sel] if visible else None

            if key in ("UP", "LEFT"):
                if visible:
                    selected_sid = visible[max(0, sel - repeat_count)]["sid"]
                ghost_sid = None
            elif key in ("DOWN", "RIGHT"):
                if visible:
                    selected_sid = visible[min(n_vis - 1, sel + repeat_count)]["sid"]
                ghost_sid = None
            elif key == "HOME":
                if visible:
                    selected_sid = visible[0]["sid"]
                ghost_sid = None
            elif key == "END":
                if visible:
                    selected_sid = visible[n_vis - 1]["sid"]
                ghost_sid = None
            elif key == "PAGEUP":
                if visible:
                    selected_sid = visible[max(0, sel - (_max_vis() * repeat_count))]["sid"]
                ghost_sid = None
            elif key == "PAGEDOWN":
                if visible:
                    selected_sid = visible[min(n_vis - 1, sel + (_max_vis() * repeat_count))]["sid"]
                ghost_sid = None
            elif key == "ENTER":
                if cur is None:
                    continue
                if cur["archived"]:
                    meta = sessions.get(cur["sid"], {})
                    hp_str = meta.get("history_file")
                    if hp_str:
                        restored = unarchive_session_files(Path(hp_str), cur["sid"])
                        if restored is not None:
                            meta["history_file"] = str(restored)
                            meta.pop("archived", None)
                            meta.pop("archived_at", None)
                            cur["archived"] = False
                            save_sessions(data)
                            auto_restored = True
                set_terminal_session(cur["sid"])
                loaded_row = cur
                break
            elif key == "ESC":
                if search_query:
                    search_query = ""
                    offset = 0
                    continue
                break
            elif key == "CTRL_F":
                _start_prefetch()
                search_active = True
                continue
            elif key == "TAB":
                view_mode = {"active": "all", "all": "archived", "archived": "active"}[view_mode]
                offset = 0
                ghost_sid = None
            elif key == "p":
                if cur is not None:
                    preview_sid = cur["sid"]
                    preview_offset = 0
            elif key == "a":
                if cur is None:
                    continue
                sid = cur["sid"]
                meta = sessions.get(sid, {})
                hp_str = meta.get("history_file")
                hp = Path(hp_str) if hp_str else None
                if cur["archived"]:
                    restored = unarchive_session_files(hp, sid) if hp else None
                    if restored is not None:
                        meta["history_file"] = str(restored)
                        meta.pop("archived", None)
                        meta.pop("archived_at", None)
                        cur["archived"] = False
                        save_sessions(data)
                        flash = (f"✓ restored {cur['short_id']}", "green")
                        ghost_sid = sid if view_mode == "archived" else None
                        _start_undo({"kind": "did_unarchive", "sid": sid})
                    else:
                        meta.pop("archived", None)
                        meta.pop("archived_at", None)
                        cur["archived"] = False
                        save_sessions(data)
                        flash = (f"○ archive missing for {cur['short_id']} — marked active", "dim")
                        ghost_sid = sid if view_mode == "archived" else None
                else:
                    archived_path = archive_session_files(hp) if hp else None
                    if archived_path is not None:
                        meta["history_file"] = str(archived_path)
                        meta["archived"] = True
                        meta["archived_at"] = isoformat_utc(utc_now())
                        for term_id, mapped_sid in list(terminals.items()):
                            if mapped_sid == sid:
                                terminals.pop(term_id)
                        cur["archived"] = True
                        cur["is_current"] = False
                        save_sessions(data)
                        flash = (f"✓ archived {cur['short_id']}", "cyan")
                        ghost_sid = sid if view_mode == "active" else None
                        _start_undo({"kind": "did_archive", "sid": sid})
                    else:
                        flash = (f"○ nothing to archive for {cur['short_id']}", "dim")
            elif key == "d":
                if cur is None:
                    continue
                sid = cur["sid"]
                if arm_delete_sid == sid:
                    meta = sessions.get(sid, {})
                    hp_str = meta.get("history_file")
                    snapshot_meta = dict(meta)
                    snapshot_row = dict(cur)
                    row_index = next(
                        (i for i, r in enumerate(rows) if r["sid"] == sid), len(rows)
                    )
                    removed_terminals: list[str] = []
                    for term_id, mapped_sid in list(terminals.items()):
                        if mapped_sid == sid:
                            removed_terminals.append(term_id)
                            terminals.pop(term_id)
                    sessions.pop(sid, None)
                    rows[:] = [r for r in rows if r["sid"] != sid]
                    save_sessions(data)
                    new_visible = _visible_rows_list()
                    if new_visible:
                        new_sel = min(sel, len(new_visible) - 1)
                        selected_sid = new_visible[new_sel]["sid"]
                    else:
                        selected_sid = None
                    flash = (f"✓ deleted {cur['short_id']}", "red")
                    arm_delete_sid = None
                    _start_undo({
                        "kind": "did_delete",
                        "sid": sid,
                        "row": snapshot_row,
                        "meta": snapshot_meta,
                        "row_index": row_index,
                        "removed_terminals": removed_terminals,
                        "history_path": hp_str,
                    })
                else:
                    arm_delete_sid = sid
            elif key == "u":
                result = _do_undo()
                if result is not None:
                    flash, restored_sid = result
                    if restored_sid is not None:
                        selected_sid = restored_sid
                        ghost_sid = restored_sid

    prefetch_stop.set()
    _commit_pending()

    if loaded_row is not None:
        label = sessions.get(loaded_row["sid"], {}).get("label", loaded_row["sid"])
        prefix = "Restored & loaded" if auto_restored else "Loaded"
        console.print(
            f"[bold green]✓[/bold green] [green]{prefix}[/green] "
            f"[bold cyan]{loaded_row['short_id']}[/bold cyan] [dim]({label})[/dim]"
        )
        return
    console.print("[dim]○ Cancelled.[/dim]")



def cmd_history() -> None:
    session_context = prepare_session_context()
    history = load_history(session_context.history_file)
    if not history:
        console.print("[dim]○ No history yet.[/dim]")
        return

    exchanges = sum(1 for m in history if isinstance(m, dict) and m.get("role") == "user")

    if not sys.stdin.isatty() or not console.is_terminal:
        parts: list = [section_rule("conversation"), Text("")]
        for m in history:
            role = m.get("role")
            if role == "user":
                line = Text()
                line.append("▌ ", style="bold cyan")
                line.append("You", style="bold cyan")
                parts.append(line)
                parts.append(Text(f"  {m.get('content', '')}"))
                parts.append(Text(""))
            elif role == "assistant":
                content = m.get("content", "")
                if content:
                    line = Text()
                    line.append("▌ ", style="bold green")
                    line.append("Jarv", style="bold green")
                    parts.append(line)
                    parts.append(Markdown(flatten_headings(content)))
                    parts.append(Text(""))
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


