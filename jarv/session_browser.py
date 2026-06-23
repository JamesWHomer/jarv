"""Interactive and plain /sessions browser."""

import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .command_input import _key_available, _read_key_with_repeats, mouse_capture
from .display import console, jarv_panel, refresh_on_resize, terminal_size
from .tui_app import AltScreenApp
from .history import (
    detect_terminal,
    isoformat_utc,
    load_history,
    load_sessions,
    parse_timestamp,
    save_sessions,
    set_terminal_session,
    utc_now,
)
from .session_render import _history_visual_lines, _session_row_widths
from .session_store import archive_session_files, delete_session_files, unarchive_session_files
from .tui_layout import append_bottom_footer, clip_text
from .tui_overlay import (
    apply_scroll_keys,
    body_content_rows,
    clamp_scroll_offset,
    scroll_position_hint,
)

def _short_session_id(sid: str) -> str:
    """Return the shortest unambiguous prefix hint for display (type prefix + 6 hash chars)."""
    # IDs look like: parent-5d44fec1a0fe  or  windows-terminal-3dece1d0fac8
    # Keep the descriptive prefix and show only 6 chars of the trailing hash.
    parts = sid.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) >= 6:
        return f"{parts[0]}-{parts[1][:6]}"
    return sid[:16]


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

    screen = SessionBrowserScreen(
        data=data,
        sessions=sessions,
        terminals=terminals,
        rows=rows,
        current_session_id=current_session_id,
    )
    screen.run()

    loaded_row = screen.loaded_row
    if loaded_row is not None:
        label = sessions.get(loaded_row["sid"], {}).get("label", loaded_row["sid"])
        prefix = "Restored & loaded" if screen.auto_restored else "Loaded"
        console.print(
            f"[bold green]✓[/bold green] [green]{prefix}[/green] "
            f"[bold cyan]{loaded_row['short_id']}[/bold cyan] [dim]({label})[/dim]"
        )
        return
    console.print("[dim]○ Cancelled.[/dim]")


class SessionBrowserScreen(AltScreenApp):
    """The interactive /sessions browser on the single-threaded alt-screen loop.

    Was a ~900-line closure in ``cmd_sessions``; the closures are now methods and
    the bespoke ``Live`` loop is the shared :class:`AltScreenApp` loop. A daemon
    prefetch thread warms the transcript-search cache and a ``threading.Timer``
    backs the 5-second undo window; neither paints -- they only mutate state and
    request a repaint, the loop is the sole renderer.
    """

    use_mouse_capture = True
    use_bracketed_paste = False
    clear_on_resize = False
    first_paint_label = "sessions"
    UNDO_WINDOW = 5.0

    def __init__(self, *, data, sessions, terminals, rows, current_session_id):
        super().__init__(
            console=console,
            read_key_fn=self._read_browser_key,
            key_available_fn=self._browser_key_available,
            terminal_size_fn=self._browser_terminal_size,
        )
        self.data = data
        self.sessions = sessions
        self.terminals = terminals
        self.rows = rows
        self.current_session_id = current_session_id

        self.view_mode = "active"  # "active" | "all" | "archived"
        self.arm_delete_sid: str | None = None
        self.flash: tuple[str, str] | None = None  # (message, style) shown above the footer
        self.search_query = ""
        self.search_active = False  # input bar focused for typing
        self.search_text_cache: dict[str, str] = {}  # sid -> lowercased transcript text
        self.last_action: dict | None = None  # most recent undoable action (5s window)
        self.undo_lock = threading.Lock()
        # When a row is archived/unarchived from a filtered view, keep it visible
        # in place (with its new aesthetic) until the cursor moves.
        self.ghost_sid: str | None = None
        self.selected_sid: str | None = next(
            (r["sid"] for r in rows if r["is_current"] and not r["archived"]),
            next((r["sid"] for r in rows if not r["archived"]), rows[0]["sid"] if rows else None),
        )
        self.offset = 0
        self.preview_sid: str | None = None
        self.preview_offset = 0
        self.preview_cache: dict[tuple[str, int], list[Text]] = {}
        self.prefetch_stop = threading.Event()
        self.prefetch_thread: threading.Thread | None = None
        self.loaded_row: dict | None = None
        self.auto_restored = False
        self._last_undo_tick = 0.0

    # ------------------------------------------------------------------ #
    # AltScreenApp wiring (module symbols resolved at call time so tests
    # patching ``jarv.session_browser.*`` keep driving the loop).
    # ------------------------------------------------------------------ #
    def _read_browser_key(self) -> tuple[str, int]:
        text_mode = self.search_active and self.preview_sid is None
        return _read_key_with_repeats(
            text_mode=text_mode,
            repeatable=() if text_mode else ("UP", "DOWN", "LEFT", "RIGHT", "PAGEUP", "PAGEDOWN"),
        )

    def _browser_key_available(self) -> bool:
        return _key_available()

    def _browser_terminal_size(self, *, console=None):
        return terminal_size(console=console)

    def _screen_context(self):
        @contextmanager
        def _ctx():
            live = Live(
                get_renderable=self.render,
                console=console,
                screen=True,
                auto_refresh=False,
                transient=False,
                vertical_overflow="crop",
            )
            with live, refresh_on_resize(live, on_change=self._on_resize_signal), mouse_capture():
                self.live = live
                yield live

        return _ctx()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def on_interrupt(self) -> None:
        # Ctrl-C cancels the browser (like the old loop's KeyboardInterrupt break).
        self.stop()

    def on_stop(self) -> None:
        self.prefetch_stop.set()
        self._commit_pending()

    def on_tick(self) -> None:
        # Animate the undo-window countdown and clear it once it lapses.
        if self.last_action is not None:
            now = time.monotonic()
            if now - self._last_undo_tick >= 0.25:
                self._last_undo_tick = now
                self.invalidate()

    # ------------------------------------------------------------------ #
    # Display helpers
    # ------------------------------------------------------------------ #
    def _truncate(self, value: str, width: int) -> str:
        return clip_text(value, width)

    def _content_rows(self, term_h: int, has_status: bool, show_footer: bool, has_search: bool = False) -> int:
        # Panel border = 2 rows. Header consumes 1 row. Footer = 2 rows (blank + controls).
        content = max(1, term_h - 2 - 1)
        if show_footer:
            content -= 2
        if has_status:
            content -= 1
        if has_search:
            content -= 1
        return max(1, content)

    def _max_vis(self, has_status: bool = False) -> int:
        _, term_h = terminal_size(console=console)
        has_search = bool(self.search_active or self.search_query)
        return self._content_rows(term_h, has_status, show_footer=term_h >= 6, has_search=has_search)

    def _fast_search_text(self, r: dict) -> str:
        # Cheap fields available without disk I/O — exact short id, full sid,
        # the user's first-message snippet, and the session label.
        meta = self.sessions.get(r["sid"], {})
        label = meta.get("label", "") if isinstance(meta.get("label"), str) else ""
        return f"{r['short_id']} {r['sid']} {r.get('snippet', '')} {label}".lower()

    def _first_user_snippet(self, meta: dict, width: int = 60) -> str:
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

    def _ensure_row_metadata(self, r: dict) -> None:
        meta = self.sessions.get(r["sid"], {})
        if not r.get("snippet_loaded"):
            r["snippet"] = self._first_user_snippet(meta)
            r["snippet_loaded"] = True

    def _build_search_text(self, sid: str) -> str:
        meta = self.sessions.get(sid, {})
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

    def _search_text(self, sid: str) -> str:
        cached = self.search_text_cache.get(sid)
        if cached is not None:
            return cached
        text = self._build_search_text(sid)
        self.search_text_cache[sid] = text
        return text

    def _prefetch_worker(self) -> None:
        for r in self.rows:
            if self.prefetch_stop.is_set():
                return
            sid = r["sid"]
            if sid in self.search_text_cache:
                continue
            try:
                text = self._build_search_text(sid)
            except Exception:
                text = ""
            self.search_text_cache[sid] = text

    def _start_prefetch(self) -> None:
        if self.prefetch_thread is not None:
            return
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()

    def _visible_rows_list(self) -> list[dict]:
        q = self.search_query.lower().strip()

        def keep(r: dict) -> bool:
            if r["sid"] == self.ghost_sid:
                return True
            if self.view_mode == "active" and r["archived"]:
                return False
            if self.view_mode == "archived" and not r["archived"]:
                return False
            if q:
                if q in self._fast_search_text(r):
                    return True
                if q not in self._search_text(r["sid"]):
                    return False
            return True

        return [r for r in self.rows if keep(r)]

    def _selected_pos(self, visible: list[dict]) -> int:
        for i, r in enumerate(visible):
            if r["sid"] == self.selected_sid:
                return i
        return 0

    def _clamp_offset(self, sel: int, off: int, mv: int, n: int) -> int:
        if n == 0:
            return 0
        if sel < off:
            return sel
        if sel >= off + mv:
            return sel - mv + 1
        return max(0, min(off, max(0, n - mv)))

    def _subtitle(self) -> str:
        n_active = sum(1 for r in self.rows if not r["archived"])
        n_archived = len(self.rows) - n_active
        if self.view_mode == "active":
            return f"[dim]{n_active} active[/dim]"
        if self.view_mode == "archived":
            return f"[dim]{n_archived} archived[/dim]"
        return f"[dim]{n_active} active · {n_archived} archived[/dim]"

    def _footer_text(self) -> str:
        if self.search_active:
            return "type to filter   Enter apply   Esc back   Backspace delete"
        cur_visible = self._visible_rows_list()
        cur = cur_visible[self._selected_pos(cur_visible)] if cur_visible else None
        a_hint = "a unarchive" if (cur and cur["archived"]) else "a archive"
        find_hint = "^F edit search" if self.search_query else "^F find"
        parts = [
            "←→/↑↓ navigate", "Enter load", "p preview", "d delete",
            a_hint, f"Tab view: {self.view_mode}", find_hint,
        ]
        action = self.last_action
        if action is not None:
            remaining = action["deadline"] - time.time()
            if remaining > 0:
                parts.append(f"u undo ({int(remaining) + 1}s)")
        parts.append("q cancel")
        return "   ".join(parts)

    def _build_preview_lines(self, sid: str, width: int) -> list[Text]:
        meta = self.sessions.get(sid, {})
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

    def _preview_lines(self, sid: str, width: int) -> list[Text]:
        cache_key = (sid, width)
        if cache_key not in self.preview_cache:
            self.preview_cache[cache_key] = self._build_preview_lines(sid, width)
        return self.preview_cache[cache_key]

    def _render_preview(self) -> Panel:
        term_w, term_h = terminal_size(console=console)
        panel_width = max(1, term_w)
        inner_width = max(1, panel_width - 4)
        body_rows, show_footer = body_content_rows(term_h)
        body_rows = max(1, body_rows - 1)

        sid = self.preview_sid or ""
        all_lines = self._preview_lines(sid, inner_width)
        total = len(all_lines)
        self.preview_offset = clamp_scroll_offset(self.preview_offset, total, body_rows)
        start = self.preview_offset
        end = min(total, start + body_rows)

        meta = self.sessions.get(sid, {})
        short_id = _short_session_id(sid) if sid else ""
        label = meta.get("label", "")

        parts: list = []
        parts.append(
            Text(
                self._truncate(f"  {short_id}  {label}", inner_width),
                style="dim",
                no_wrap=True,
                overflow="crop",
            )
        )
        for index in range(start, end):
            parts.append(all_lines[index])
        if total == 0:
            parts.append(Text(self._truncate("  (empty)", inner_width), style="dim"))

        if show_footer:
            position = scroll_position_hint(start, end, total)
            append_bottom_footer(
                parts,
                term_h,
                Text(
                    self._truncate(
                        f"↑↓ scroll   ←→ session   Enter load   p/Esc back   ·   {position}",
                        inner_width,
                    ),
                    style="dim italic",
                    no_wrap=True,
                    overflow="crop",
                ),
            )

        return jarv_panel(
            Group(*parts),
            "preview",
            subtitle=short_id or None,
            padding=(0, 1),
            width=panel_width,
            height=term_h,
        )

    def _search_bar(self, inner_width: int) -> Text:
        cursor = "▌" if self.search_active else ""
        shown = self.search_query + cursor
        prefix = " › " if self.search_active else "   "
        label_style = "bold cyan" if self.search_active else "cyan"
        value_style = "bold cyan" if self.search_active else "bold"
        placeholder_style = "bold cyan" if self.search_active else "dim italic"
        line = Text(no_wrap=True, overflow="crop")
        line.append(prefix, style="bold cyan" if self.search_active else "")
        line.append("find: ", style=label_style)
        avail = max(0, inner_width - len(prefix) - 6)
        if shown:
            line.append(self._truncate(shown, avail), style=value_style)
        else:
            line.append(self._truncate("(type to filter transcripts)", avail), style=placeholder_style)
        return line

    def render(self) -> Panel:
        if self.preview_sid is not None:
            return self._render_preview()
        term_w, term_h = terminal_size(console=console)
        panel_width = max(1, term_w)
        inner_width = max(1, panel_width - 4)
        show_footer = term_h >= 6

        visible = self._visible_rows_list()
        n = len(visible)
        sel = self._selected_pos(visible)

        status: Text | None = None
        cur = visible[sel] if visible else None
        if self.arm_delete_sid and cur and cur["sid"] == self.arm_delete_sid:
            prompt = (
                f"Delete {cur['short_id']} permanently? "
                "Press d again to confirm · any other key cancels"
            )
            status = Text(self._truncate(prompt, inner_width), style="bold red", no_wrap=True, overflow="crop")
        elif self.flash is not None:
            msg, style = self.flash
            status = Text(self._truncate(msg, inner_width), style=style, no_wrap=True, overflow="crop")

        has_search = bool(self.search_active or self.search_query)
        mv = self._content_rows(
            term_h,
            has_status=status is not None,
            show_footer=show_footer,
            has_search=has_search,
        )
        self.offset = self._clamp_offset(sel, self.offset, mv, n)
        start = self.offset
        end = min(n, self.offset + mv)

        parts: list = []
        if n == 0:
            if has_search:
                parts.append(self._search_bar(inner_width))
            empty_msg = (
                f"  (no sessions match \"{self.search_query}\")"
                if self.search_query
                else "  (no sessions in this view)"
            )
            parts.append(Text(self._truncate(empty_msg, inner_width), style="dim"))
            if status is not None:
                parts.append(status)
            if show_footer:
                append_bottom_footer(
                    parts,
                    term_h,
                    Text(
                        self._truncate(self._footer_text(), inner_width),
                        style="dim italic",
                        no_wrap=True,
                        overflow="crop",
                    ),
                )
            return Panel(
                Group(*parts),
                title="[bold bright_white]jarv ▸ sessions[/bold bright_white]",
                title_align="left",
                subtitle=self._subtitle(),
                subtitle_align="right",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 1),
                width=panel_width,
                height=term_h,
            )

        parts.append(
            Text(
                self._truncate(f"  showing {start + 1}–{end} of {n}", inner_width),
                style="dim",
                no_wrap=True,
                overflow="crop",
            )
        )
        if has_search:
            parts.append(self._search_bar(inner_width))

        for i in range(start, end):
            r = visible[i]
            self._ensure_row_metadata(r)
            is_sel = (i == sel) and not self.search_active
            is_armed = is_sel and self.arm_delete_sid == r["sid"]
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
            t.append(self._truncate(prefix, inner_width), style=prefix_style)

            if inner_width > len(prefix):
                if is_armed:
                    marker_style = "bold red"
                elif r["is_current"]:
                    marker_style = "green"
                elif r["archived"]:
                    marker_style = "dim"
                else:
                    marker_style = ""
                t.append(self._truncate(marker, inner_width - len(prefix)), style=marker_style)

            if id_width:
                short_id = self._truncate(r["short_id"], id_width)
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
                time_str = self._truncate(r["time_str"], time_width)
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
                t.append(self._truncate(snip, snippet_width), style=snip_style)
            parts.append(t)

        if status is not None:
            parts.append(status)

        if show_footer:
            append_bottom_footer(
                parts,
                term_h,
                Text(
                    self._truncate(self._footer_text(), inner_width),
                    style="dim italic",
                    no_wrap=True,
                    overflow="crop",
                ),
            )

        return Panel(
            Group(*parts),
            title="[bold bright_white]jarv ▸ sessions[/bold bright_white]",
            title_align="left",
            subtitle=self._subtitle(),
            subtitle_align="right",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            width=panel_width,
            height=term_h,
        )

    # ------------------------------------------------------------------ #
    # Undo window (backed by a daemon Timer that finalizes on expiry)
    # ------------------------------------------------------------------ #
    def _finalize_action(self, action: dict) -> None:
        """Apply the irreversible part of an action that's leaving its window."""
        if action["kind"] == "did_delete":
            hp_str = action.get("history_path")
            if hp_str:
                delete_session_files(Path(hp_str))

    def _expire_action(self, action: dict) -> None:
        with self.undo_lock:
            if self.last_action is not action:
                return
            self.last_action = None
        self._finalize_action(action)
        # Drop the stale "u undo" footer hint now the window has lapsed.
        self.invalidate()

    def _commit_pending(self) -> None:
        with self.undo_lock:
            action = self.last_action
            self.last_action = None
        if action is None:
            return
        t = action.get("timer")
        if t is not None:
            t.cancel()
        self._finalize_action(action)

    def _start_undo(self, action: dict) -> None:
        self._commit_pending()
        action["deadline"] = time.time() + self.UNDO_WINDOW
        timer = threading.Timer(self.UNDO_WINDOW, self._expire_action, args=(action,))
        timer.daemon = True
        action["timer"] = timer
        with self.undo_lock:
            self.last_action = action
        timer.start()

    def _take_last_action(self) -> dict | None:
        """Atomically grab and clear the current undoable action if still valid."""
        with self.undo_lock:
            action = self.last_action
            if action is None:
                return None
            if time.time() >= action["deadline"]:
                # Already expired — let the timer handle finalization.
                return None
            self.last_action = None
        t = action.get("timer")
        if t is not None:
            t.cancel()
        return action

    def _do_undo(self) -> tuple[tuple[str, str], str | None] | None:
        """Returns ((flash_msg, flash_style), restored_sid) or None."""
        action = self._take_last_action()
        if action is None:
            return None
        kind = action["kind"]
        if kind == "did_archive":
            sid = action["sid"]
            row = next((r for r in self.rows if r["sid"] == sid), None)
            if row is None:
                return (("○ session no longer exists", "dim"), None)
            meta = self.sessions.get(sid, {})
            hp_str = meta.get("history_file")
            hp = Path(hp_str) if hp_str else None
            restored = unarchive_session_files(hp, sid) if hp else None
            if restored is not None:
                meta["history_file"] = str(restored)
            meta.pop("archived", None)
            meta.pop("archived_at", None)
            row["archived"] = False
            save_sessions(self.data)
            return ((f"↺ restored {row['short_id']}", "green"), sid)
        if kind == "did_unarchive":
            sid = action["sid"]
            row = next((r for r in self.rows if r["sid"] == sid), None)
            if row is None:
                return (("○ session no longer exists", "dim"), None)
            meta = self.sessions.get(sid, {})
            hp_str = meta.get("history_file")
            hp = Path(hp_str) if hp_str else None
            archived_path = archive_session_files(hp) if hp else None
            if archived_path is None:
                return (("○ couldn't re-archive", "dim"), None)
            meta["history_file"] = str(archived_path)
            meta["archived"] = True
            meta["archived_at"] = isoformat_utc(utc_now())
            for term_id, mapped_sid in list(self.terminals.items()):
                if mapped_sid == sid:
                    self.terminals.pop(term_id)
            row["archived"] = True
            row["is_current"] = False
            save_sessions(self.data)
            return ((f"↺ archived {row['short_id']}", "cyan"), sid)
        if kind == "did_delete":
            sid = action["sid"]
            snapshot_row = action["row"]
            snapshot_meta = action["meta"]
            row_index = action.get("row_index", len(self.rows))
            removed_terminals = action.get("removed_terminals", [])
            self.sessions[sid] = snapshot_meta
            for term_id in removed_terminals:
                self.terminals[term_id] = sid
            if 0 <= row_index <= len(self.rows):
                self.rows.insert(row_index, snapshot_row)
            else:
                self.rows.append(snapshot_row)
            save_sessions(self.data)
            return ((f"↺ restored {snapshot_row['short_id']}", "green"), sid)
        return None

    # ------------------------------------------------------------------ #
    # Row actions
    # ------------------------------------------------------------------ #
    def _activate_row(self, row: dict) -> None:
        if row["archived"]:
            meta = self.sessions.get(row["sid"], {})
            hp_str = meta.get("history_file")
            if hp_str:
                restored = unarchive_session_files(Path(hp_str), row["sid"])
                if restored is not None:
                    meta["history_file"] = str(restored)
                    meta.pop("archived", None)
                    meta.pop("archived_at", None)
                    row["archived"] = False
                    save_sessions(self.data)
                    self.auto_restored = True
        set_terminal_session(row["sid"])
        self.loaded_row = row
        self.stop()

    def _toggle_archive(self, cur: dict | None) -> None:
        if cur is None:
            return
        sid = cur["sid"]
        meta = self.sessions.get(sid, {})
        hp_str = meta.get("history_file")
        hp = Path(hp_str) if hp_str else None
        if cur["archived"]:
            restored = unarchive_session_files(hp, sid) if hp else None
            if restored is not None:
                meta["history_file"] = str(restored)
                meta.pop("archived", None)
                meta.pop("archived_at", None)
                cur["archived"] = False
                save_sessions(self.data)
                self.flash = (f"✓ restored {cur['short_id']}", "green")
                self.ghost_sid = sid if self.view_mode == "archived" else None
                self._start_undo({"kind": "did_unarchive", "sid": sid})
            else:
                meta.pop("archived", None)
                meta.pop("archived_at", None)
                cur["archived"] = False
                save_sessions(self.data)
                self.flash = (f"○ archive missing for {cur['short_id']} — marked active", "dim")
                self.ghost_sid = sid if self.view_mode == "archived" else None
        else:
            archived_path = archive_session_files(hp) if hp else None
            if archived_path is not None:
                meta["history_file"] = str(archived_path)
                meta["archived"] = True
                meta["archived_at"] = isoformat_utc(utc_now())
                for term_id, mapped_sid in list(self.terminals.items()):
                    if mapped_sid == sid:
                        self.terminals.pop(term_id)
                cur["archived"] = True
                cur["is_current"] = False
                save_sessions(self.data)
                self.flash = (f"✓ archived {cur['short_id']}", "cyan")
                self.ghost_sid = sid if self.view_mode == "active" else None
                self._start_undo({"kind": "did_archive", "sid": sid})
            else:
                self.flash = (f"○ nothing to archive for {cur['short_id']}", "dim")

    def _handle_delete_key(self, cur: dict | None, sel: int) -> None:
        if cur is None:
            return
        sid = cur["sid"]
        if self.arm_delete_sid != sid:
            self.arm_delete_sid = sid
            return
        meta = self.sessions.get(sid, {})
        hp_str = meta.get("history_file")
        snapshot_meta = dict(meta)
        snapshot_row = dict(cur)
        row_index = next((i for i, r in enumerate(self.rows) if r["sid"] == sid), len(self.rows))
        removed_terminals: list[str] = []
        for term_id, mapped_sid in list(self.terminals.items()):
            if mapped_sid == sid:
                removed_terminals.append(term_id)
                self.terminals.pop(term_id)
        self.sessions.pop(sid, None)
        self.rows[:] = [r for r in self.rows if r["sid"] != sid]
        save_sessions(self.data)
        new_visible = self._visible_rows_list()
        if new_visible:
            new_sel = min(sel, len(new_visible) - 1)
            self.selected_sid = new_visible[new_sel]["sid"]
        else:
            self.selected_sid = None
        self.flash = (f"✓ deleted {cur['short_id']}", "red")
        self.arm_delete_sid = None
        self._start_undo({
            "kind": "did_delete",
            "sid": sid,
            "row": snapshot_row,
            "meta": snapshot_meta,
            "row_index": row_index,
            "removed_terminals": removed_terminals,
            "history_path": hp_str,
        })

    # ------------------------------------------------------------------ #
    # Key handling
    # ------------------------------------------------------------------ #
    def on_key(self, key: str, repeat: int) -> None:
        repeat_count = repeat

        # Search-input mode intercepts most keys (only outside preview).
        if self.search_active and self.preview_sid is None:
            self._on_key_search(key)
            return

        # Preview mode intercepts most keys.
        if self.preview_sid is not None:
            self._on_key_preview(key, repeat_count)
            return

        if key == "ESC" and self.arm_delete_sid is not None:
            self.arm_delete_sid = None
            self.flash = None
            return

        if key != "d":
            self.arm_delete_sid = None
        self.flash = None

        visible = self._visible_rows_list()
        n_vis = len(visible)
        sel = self._selected_pos(visible) if visible else 0
        cur = visible[sel] if visible else None

        if key in ("UP", "LEFT"):
            if visible:
                self.selected_sid = visible[max(0, sel - repeat_count)]["sid"]
            self.ghost_sid = None
        elif key in ("DOWN", "RIGHT"):
            if visible:
                self.selected_sid = visible[min(n_vis - 1, sel + repeat_count)]["sid"]
            self.ghost_sid = None
        elif key == "HOME":
            if visible:
                self.selected_sid = visible[0]["sid"]
            self.ghost_sid = None
        elif key == "END":
            if visible:
                self.selected_sid = visible[n_vis - 1]["sid"]
            self.ghost_sid = None
        elif key == "PAGEUP":
            if visible:
                self.selected_sid = visible[max(0, sel - (self._max_vis() * repeat_count))]["sid"]
            self.ghost_sid = None
        elif key == "PAGEDOWN":
            if visible:
                self.selected_sid = visible[min(n_vis - 1, sel + (self._max_vis() * repeat_count))]["sid"]
            self.ghost_sid = None
        elif key == "ENTER":
            if cur is not None:
                self._activate_row(cur)
        elif key == "ESC":
            self.stop()
        elif key == "CTRL_F":
            self._start_prefetch()
            self.search_active = True
        elif key == "TAB":
            self.view_mode = {"active": "all", "all": "archived", "archived": "active"}[self.view_mode]
            self.offset = 0
            self.ghost_sid = None
        elif key == "p":
            if cur is not None:
                self.preview_sid = cur["sid"]
                self.preview_offset = 0
        elif key == "a":
            self._toggle_archive(cur)
        elif key == "d":
            self._handle_delete_key(cur, sel)
        elif key == "u":
            result = self._do_undo()
            if result is not None:
                self.flash, restored_sid = result
                if restored_sid is not None:
                    self.selected_sid = restored_sid
                    self.ghost_sid = restored_sid

    def _on_key_search(self, key: str) -> None:
        if key == "ESC":
            self.search_active = False
            self.search_query = ""
            self.offset = 0
        elif key in ("ENTER", "DOWN"):
            # Exit search mode, drop focus into the filtered list.
            self.search_active = False
            visible_now = self._visible_rows_list()
            if visible_now and not any(r["sid"] == self.selected_sid for r in visible_now):
                self.selected_sid = visible_now[0]["sid"]
            self.offset = 0
        elif key == "BACKSPACE":
            if self.search_query:
                self.search_query = self.search_query[:-1]
                self.offset = 0
        elif key == "CTRL_F":
            self._start_prefetch()
            self.search_active = False
        elif isinstance(key, str) and len(key) == 1 and key.isprintable():
            self.search_query += key
            self.offset = 0
            visible_now = self._visible_rows_list()
            if visible_now and not any(r["sid"] == self.selected_sid for r in visible_now):
                self.selected_sid = visible_now[0]["sid"]
        # All other keys (UP, PAGEUP/DOWN, HOME/END, TAB, a, d, p, …) are
        # swallowed while typing the query.

    def _on_key_preview(self, key: str, repeat_count: int) -> None:
        if key in ("p", "ESC"):
            self.preview_sid = None
            self.preview_offset = 0
        elif key in ("LEFT", "RIGHT"):
            visible_now = self._visible_rows_list()
            if visible_now:
                pos = next(
                    (i for i, r in enumerate(visible_now) if r["sid"] == self.preview_sid),
                    self._selected_pos(visible_now),
                )
                delta = -repeat_count if key == "LEFT" else repeat_count
                pos = max(0, min(len(visible_now) - 1, pos + delta))
                self.preview_sid = visible_now[pos]["sid"]
                self.selected_sid = self.preview_sid
                self.preview_offset = 0
        elif key in ("UP", "DOWN", "PAGEUP", "PAGEDOWN", "HOME", "END"):
            term_w, term_h = terminal_size(console=console)
            inner_width = max(1, term_w - 4)
            total = len(self._preview_lines(self.preview_sid, inner_width))
            body_rows, _ = body_content_rows(term_h)
            body_rows = max(1, body_rows - 1)
            self.preview_offset = apply_scroll_keys(
                key,
                repeat_count,
                offset=self.preview_offset,
                total=total,
                body_rows=body_rows,
            )
        elif key == "ENTER":
            row = next((r for r in self.rows if r["sid"] == self.preview_sid), None)
            if row is not None:
                self._activate_row(row)
