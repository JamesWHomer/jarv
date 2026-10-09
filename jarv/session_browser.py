"""Interactive and plain /sessions browser."""

import sys
import time
from collections import OrderedDict
from pathlib import Path

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .command_input import TextInput, _key_available, _read_key_with_repeats
from .display import console, jarv_panel, terminal_size
from .tui_app import AltScreenApp
from .history import (
    detect_terminal,
    load_history,
    load_sessions,
    parse_timestamp,
    save_sessions,
    set_terminal_session,
    utc_now,
)
from .session_render import _history_visual_lines
from .session_browser_work import BrowserWorker
from .session_titles import SessionTitleCache
from .session_browser_live import SessionBrowserLive as Live
from .session_browser_render import (
    beside, conversation_title, first_prompt, fitted, highlighted, highlighted_transcript, match_excerpt,
    one_line, pane_heading, reflow_position, session_row, short_session_id as _short_session_id,
)
from .text_editor import initialize_text_editor, apply_text_editor_key, render_single_line
from .tool_outputs import flatten_content_text
from .session_store import archive_session_files, delete_session_files, unarchive_session_files, mark_session_archived, session_metadata_transaction
from .storage import StorageError
from .tui_frame import panel_width
from .tui_panel import MenuPanel, menu_frame_rows, menu_inner_width
from .tui_layout import append_bottom_footer
from .tui_overlay import (
    SELECTION_KEYS,
    SHIFT_SELECTION_KEYS,
    apply_scroll_keys,
    apply_selection_keys,
    body_content_rows,
    clamp_scroll_offset,
    clamp_selection_scroll,
    scroll_position_hint,
)


def _session_time_label(ts, now) -> str:
    """Format the same last-active label in both session browsers."""
    if ts is None:
        return "—"
    secs = int((now - ts).total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    if secs < 7 * 86400:
        return f"{secs // 86400}d ago"
    return ts.strftime("%b %d")


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
    table.add_column("Conversation")
    table.add_column("Last active", style="dim", no_wrap=True)
    table.add_column("ID prefix", style="dim", no_wrap=True)

    for sid in sorted_sessions:
        meta = sessions[sid]
        ts_str = meta.get("last_message_at") or meta.get("last_used_at")
        ts = parse_timestamp(ts_str)
        time_str = _session_time_label(ts, now)

        snippet = ""
        history_path_str = meta.get("history_file")
        if history_path_str:
            history_path = Path(history_path_str)
            if history_path.exists():
                history = load_history(history_path)
                snippet = first_prompt(history)

        marker = "[green]●[/green]" if sid == current_session_id else ""
        title = Text(conversation_title(meta, snippet), no_wrap=True, overflow="ellipsis")
        if meta.get("archived"):
            title.append(" [archived]", style="dim")
        table.add_row(marker, title, time_str, _short_session_id(sid))

    total = len(sessions)
    shown = len(sorted_sessions)
    footer_parts: list = [table]
    if total > shown:
        footer_parts += [Text(""), Text(f"Showing {shown} most recent of {total} sessions.", style="dim")]
    footer_parts += [Text("Run jarv /sessions <id> to switch to a session.", style="dim italic")]
    console.print(jarv_panel(Group(*footer_parts), title="sessions", subtitle=f"{shown}/{total}"))


def _cmd_sessions_load(prefix: str) -> int:
    data = load_sessions()
    sessions = data["sessions"]
    if not sessions:
        console.print("[yellow]No sessions exist yet.[/yellow]")
        return 1
    if prefix in sessions:
        session_id = prefix
    else:
        matches = [sid for sid in sessions if sid.startswith(prefix)]
        if not matches:
            console.print(f"[bold red]✗[/bold red] [red]No session matches:[/red] [bold]{prefix}[/bold]")
            console.print("[dim]  Run [bold]jarv /sessions[/bold] to see available sessions.[/dim]")
            return 1
        if len(matches) > 1:
            console.print(f"[bold yellow]?[/bold yellow] [yellow]Ambiguous prefix[/yellow] [bold]{prefix}[/bold] [dim]matches {len(matches)} sessions:[/dim]")
            for m in matches:
                console.print(f"  [dim]•[/dim] [cyan]{m}[/cyan]")
            return 1
        session_id = matches[0]
    meta = sessions[session_id]
    if meta.get("archived"):
        path = meta.get("history_file")
        if not path:
            console.print("[red]Archived session files are missing.[/red]")
            return 1
        with session_metadata_transaction(Path(path), data):
            restored = unarchive_session_files(Path(path), session_id)
            if restored is None:
                console.print("[red]Archived session files are missing.[/red]")
                return 1
            meta["history_file"] = str(restored)
            meta.pop("archived", None)
            meta.pop("archived_at", None)
            save_sessions(data)
    set_terminal_session(session_id)
    label = sessions[session_id].get("title") or sessions[session_id].get("label", session_id)
    notice = Text("✓ Loaded ", style="green")
    notice.append(str(label), style="bold")
    notice.append(f" ({_short_session_id(session_id)})", style="dim")
    console.print(notice)
    return 0


def cmd_sessions(args: list | None = None) -> int | None:
    if args:
        if len(args) != 1:
            console.print("[red]Usage: /sessions [id][/red]")
            return 2
        return _cmd_sessions_load(args[0])
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

    # Full indexing stays on the worker. Prepare visible names separately.
    from .history import SESSIONS_FILE
    title_cache = SessionTitleCache(SESSIONS_FILE.with_name("session-titles.json"),
                                    (meta.get("history_file") for meta in sessions.values()),
                                    metadata_path=SESSIONS_FILE)
    rows: list[dict] = []
    for sid in sorted_sessions:
        meta = sessions[sid]
        ts_str = meta.get("last_message_at") or meta.get("last_used_at")
        ts = parse_timestamp(ts_str)
        time_str = _session_time_label(ts, now)

        cached_snippet = meta.get("first_user_snippet")
        if not isinstance(cached_snippet, str):
            cached_snippet = meta.get("first_message")
        if not isinstance(cached_snippet, str):
            cached_snippet = ""
        snippet = one_line(cached_snippet)[:240]
        age = (now.astimezone().date() - ts.astimezone().date()).days if ts else None
        date_group = (
            "Today" if age is not None and age <= 0 else
            "Yesterday" if age == 1 else
            "Earlier this week" if age is not None and age < 7 else
            "Older" if age is not None else "Unknown date"
        )

        rows.append({
            "sid": sid,
            "short_id": _short_session_id(sid),
            "time_str": time_str,
            "date_group": date_group,
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
        title_cache=title_cache,
    )
    screen.prepare_titles()
    screen.run()

    loaded_row = screen.loaded_row
    if loaded_row is not None:
        label = screen._title(loaded_row)
        prefix = "Restored & loaded" if screen.auto_restored else "Loaded"
        notice = Text(f"✓ {prefix} ", style="green")
        notice.append(label, style="bold")
        notice.append(f" ({loaded_row['short_id']})", style="dim")
        console.print(notice)
        return
    console.print("[dim]Sessions closed.[/dim]")


class SessionBrowserScreen(AltScreenApp):
    """The interactive /sessions browser on the single-threaded alt-screen loop.

    Was a ~900-line closure in ``cmd_sessions``; the closures are now methods and
    the bespoke ``Live`` loop is the shared :class:`AltScreenApp` loop. A daemon
    worker indexes history and prepares previews without blocking input. Undo
    remains available for the lifetime of the browser; only the main loop
    accepts worker results, paints, or changes session metadata.
    """

    use_mouse_capture = True
    use_bracketed_paste = False
    clear_on_resize = False
    first_paint_label = "sessions"
    VIEW_MODES = ("active", "archived", "all")
    PREVIEW_CACHE_ENTRIES = 24
    PREVIEW_CACHE_LINES = 40000

    def __init__(self, *, data, sessions, terminals, rows, current_session_id, background=True, title_cache=None):
        super().__init__(
            console=console,
            live_factory=self._browser_live_factory,
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
        # Sids armed for deletion; the second ``d`` only fires if the target set
        # is still the same one that armed it.
        self.arm_delete_sids: frozenset[str] | None = None
        self.flash: tuple[str, str] | None = None  # (message, style) shown above the footer
        self.flash_until = 0.0
        self.search_query = ""
        self.search_active = False  # input bar focused for typing
        self.search_editor: dict = {}
        initialize_text_editor(self.search_editor, "")
        self.search_text_cache: dict[str, str] = {}  # sid -> transcript text
        self.search_folded_cache: dict[str, str] = {}
        self.search_revision = 0
        self.visible_cache = None
        self.excerpt_cache = OrderedDict()
        self.indexed_paths = {}
        self.exchange_counts = {}
        self.row_by_sid = {row["sid"]: row for row in rows}
        # Synchronous mode is useful for static renderers and deterministic
        # state tests. The interactive browser always uses background work.
        self.background = background
        self.title_cache = title_cache
        self.worker = BrowserWorker(lambda result: self.post_app_event("browser_work", result))
        self.history_cache = OrderedDict()  # accessed only by the worker
        self.index_requested = {}
        self.prefetch_started = False
        self.detail_cache: tuple | None = None
        self.undo_actions: list[dict] = []
        self.rename_sid: str | None = None
        self.rename_editor: dict = {}
        self.help_open = False
        self.help_offset = 0
        self.help_total = 0
        # Renaming a search match can retain it until the cursor moves.
        self.ghost_sids: set[str] = set()
        self.selected_sid: str | None = next(
            (r["sid"] for r in rows if r["is_current"] and not r["archived"]),
            next((r["sid"] for r in rows if not r["archived"]), rows[0]["sid"] if rows else None),
        )
        # Contiguous Shift+arrow range selection. ``anchor_sid`` is where the
        # range started; an empty ``marked_sids`` means "act on the cursor row".
        self.anchor_sid: str | None = None
        self.marked_sids: set[str] = set()
        self.persistent_selection = False
        self.offset = 0
        self.pane_focus = "sessions"
        self.preview_sid: str | None = None
        self.preview_return_focus = "sessions"
        self.preview_positions: dict[tuple[str, int, str], int] = {}
        self.preview_cache = OrderedDict()
        self.preview_document_cache: tuple | None = None
        self.preview_documents = OrderedDict()
        self.preview_request = None
        self.preview_request_path = None
        self.preview_checked_at = 0.0
        self.preview_validated = {}
        self.preview_transfers = {}
        self.preview_roundtrip = None
        self.preview_prefetch = {}
        self.preview_prefetch_attempts = {}
        self.preview_direction = 1
        self.loaded_row: dict | None = None
        self.auto_restored = False
        self._last_list_capacity = 1

    # ------------------------------------------------------------------ #
    # AltScreenApp wiring (module symbols resolved at call time so tests
    # patching ``jarv.session_browser.*`` keep driving the loop).
    # ------------------------------------------------------------------ #
    def _read_browser_key(self) -> tuple[str, int]:
        text_mode = self.rename_sid is not None or (self.search_active and self.preview_sid is None)
        return _read_key_with_repeats(
            text_mode=text_mode,
            repeatable=()
            if text_mode
            else (
                "UP",
                "DOWN",
                "LEFT",
                "RIGHT",
                "SHIFT_UP",
                "SHIFT_DOWN",
                "PAGEUP",
                "PAGEDOWN",
                "MOUSE_WHEEL_UP",
                "MOUSE_WHEEL_DOWN",
            ),
            # Raw wheel tokens: the list maps them onto selection movement and
            # the preview takes 3-line steps (see on_key / _on_key_preview).
            translate_mouse_wheel=False,
            # A fast picker can paint each step. Draining a whole repeat burst
            # here makes held arrows skip visibly over intermediate rows.
            max_count=1,
        )

    def _browser_key_available(self) -> bool:
        return _key_available()

    def _browser_terminal_size(self, *, console=None):
        return terminal_size(console=console)

    def _browser_live_factory(self, get_renderable, _console):
        return Live(
            get_renderable=get_renderable,
            console=self.console,
            screen=True,
            auto_refresh=False,
            transient=False,
            vertical_overflow="crop",
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def on_interrupt(self) -> None:
        # Ctrl-C cancels the browser (like the old loop's KeyboardInterrupt break).
        self.stop()

    def on_stop(self) -> None:
        self.worker.close()
        self._commit_pending()

    def on_start(self) -> None:
        self._start_prefetch()

    def prepare_titles(self):
        if self.title_cache is None:
            return
        for row in self.rows:
            snippet = self.title_cache.get(self.sessions[row["sid"]].get("history_file"))
            if snippet is not None:
                row.update(snippet=snippet, snippet_loaded=True)
        visible = self._visible_rows_list()
        selected = self._selected_pos(visible)
        height = terminal_size(console=self.console)[1]
        # Include date headings and either possible position of the cursor.
        # Bound cold work by the viewport and 25 ms, never the full collection.
        deadline = time.monotonic() + .025
        for row in visible[max(0, selected - height):max(height, selected + 1)]:
            if time.monotonic() >= deadline:
                break
            meta = self.sessions[row["sid"]]
            if not row.get("snippet_loaded") and not meta.get("title"):
                snippet = self.title_cache.read(meta.get("history_file"))
                if snippet is not None:
                    row.update(snippet=snippet, snippet_loaded=True)

    @staticmethod
    def _bounded_put(cache, key, value, limit=12):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def _read_document(self, path):
        """Load once for indexing and previewing, off the input thread."""
        try:
            stat = Path(path).stat() if path else None
        except OSError:
            stat = None
        stamp = (path, stat.st_mtime_ns if stat else None, stat.st_size if stat else None)
        cached = self.history_cache.get(path)
        if cached is not None and cached[0] == stamp:
            self.history_cache.move_to_end(path)
            return cached
        error = ""
        try:
            history = load_history(Path(path)) if stat else []
            if not stat:
                error = "History file is unavailable."
        except Exception:
            history, error = [], "Couldn't read this conversation."
        chunks = [flatten_content_text(item.get("content", "")) for item in history if isinstance(item, dict)]
        text = "\n".join(chunks)
        count = sum(item.get("role") == "user" for item in history if isinstance(item, dict))
        record = (stamp, history, text, text.lower(), first_prompt(history)[:240], count, error)
        if not error and self.title_cache is not None and self.title_cache.remember(stamp, record[4]):
            self.title_cache.save()
        self._bounded_put(self.history_cache, path, record, limit=8)
        return record

    def _accept_index(self, sid, record):
        stamp, _, text, folded, snippet, count, _ = record
        if sid not in self.sessions:
            self.index_requested.pop(sid, None)
            return
        if self.sessions[sid].get("history_file") != stamp[0]:
            # Archiving/restoring can move a file while it is being indexed.
            # Retry its new location instead of leaving a permanent gap.
            self.index_requested.pop(sid, None)
            row = self.row_by_sid.get(sid)
            if row is not None:
                self._request_index(row)
            return
        if self.indexed_paths.get(sid) == stamp:
            return
        self.indexed_paths[sid] = stamp
        self.search_text_cache[sid] = text
        self.search_folded_cache[sid] = folded
        self.exchange_counts[sid] = count
        row = self.row_by_sid.get(sid)
        if row is not None:
            row.update(snippet=snippet, snippet_loaded=True)
        self.search_revision += 1

    def _request_index(self, row, *, priority=2):
        sid = row["sid"]
        path = self.sessions.get(sid, {}).get("history_file")
        if sid in self.index_requested and self.index_requested[sid] == path:
            return
        self.index_requested[sid] = path
        self.worker.submit(("index", sid), lambda cancelled: (sid, self._read_document(path)), priority=priority)

    def on_app_event(self, event):
        if event.kind != "browser_work":
            return
        key, version, result, error = event.payload
        if not self.worker.current(key, version):
            return
        if error:
            if key == "preview":
                self.preview_request = None
                self._notify("Couldn't load the preview.", "red")
            elif isinstance(key, tuple) and key[0] == "warm_preview":
                self.preview_prefetch.pop(key[1:], None)
            elif key != "titles":
                sid = key[1]
                path = self.index_requested.get(sid)
                self._accept_index(sid, ((path, None, None), [], "", "", "", 0, error))
            return
        if key == "titles":
            for sid, path, snippet in result:
                row = self.row_by_sid.get(sid)
                if row is not None and self.sessions.get(sid, {}).get("history_file") == path:
                    row.update(snippet=snippet, snippet_loaded=True)
            self.search_revision += 1
            return
        if key == "preview" or (isinstance(key, tuple) and key[0] == "warm_preview"):
            if key != "preview":
                self.preview_prefetch.pop(key[1:], None)
            sid, width, query, record, lines, document, match = result
            if sid not in self.sessions or self.sessions[sid].get("history_file") != record[0][0]:
                if self.preview_request == (sid, width, query):
                    self.preview_request = None
                return
            self._accept_index(sid, record)
            old = self.preview_cache.get((sid, width))
            if old is not None and old[0] != record[0]:
                for saved in list(self.preview_documents):
                    if saved[:2] == (sid, width):
                        self.preview_documents.pop(saved)
            self._bounded_put(self.preview_cache, (sid, width), (record[0], lines), limit=self.PREVIEW_CACHE_ENTRIES)
            self._bounded_put(self.preview_documents, (sid, width, query), (document, match), limit=self.PREVIEW_CACHE_ENTRIES)
            self.preview_validated[(sid, width)] = time.monotonic()
            # Bound retained Rich objects as well as the number of sessions.
            # A single large selected transcript must still remain readable.
            while len(self.preview_cache) > 1 and sum(len(value[1]) for value in self.preview_cache.values()) > self.PREVIEW_CACHE_LINES:
                oldest = next(iter(self.preview_cache))
                if self.preview_request is not None and oldest == self.preview_request[:2]:
                    self.preview_cache.move_to_end(oldest)
                    continue
                self.preview_cache.pop(oldest)
            for saved in list(self.preview_documents):
                if saved[:2] not in self.preview_cache:
                    self.preview_documents.pop(saved)
            transfer = self.preview_transfers.pop((sid, width, query), None)
            self.preview_validated = {saved: checked for saved, checked in self.preview_validated.items()
                                      if saved in self.preview_cache}
            if transfer is not None:
                mapped = reflow_position(transfer[0], transfer[1], document)
                self.preview_positions[(sid, width, query)] = mapped
                self.preview_roundtrip = (sid, transfer[2], width, transfer[1], mapped)
        else:
            self._accept_index(*result)

    def on_tick(self):
        if self.flash is not None and time.monotonic() >= self.flash_until:
            self.flash = None
            self.invalidate()
        # Check the selected transcript for external edits at most once a
        # second. File stats and formatting still happen on the worker.
        if self.background and self.preview_request is not None and time.monotonic() - self.preview_checked_at > 1:
            sid, width, _ = self.preview_request
            self.preview_request = None
            self._request_preview(sid, width, force=True)

    # ------------------------------------------------------------------ #
    # Display helpers
    # ------------------------------------------------------------------ #
    def _notify(self, message: str, style: str = "cyan") -> None:
        self.flash = (one_line(message), style)
        self.flash_until = time.monotonic() + (8.0 if style in ("red", "yellow") else 4.0)

    def _truncate(self, value: str, width: int) -> str:
        return fitted(value, width).plain

    def _max_vis(self, has_status: bool = False) -> int:
        return self._last_list_capacity

    def _fast_search_text(self, r: dict) -> str:
        # Cheap fields available without disk I/O — exact short id, full sid,
        # the user's first-message snippet, and the session label.
        meta = self.sessions.get(r["sid"], {})
        label = meta.get("label", "") if isinstance(meta.get("label"), str) else ""
        return f"{r['short_id']} {r['sid']} {r.get('snippet', '')} {label} {meta.get('title', '')}".lower()

    def _first_user_snippet(self, meta: dict, width: int = 240) -> str:
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
        return first_prompt(history)[:width]

    def _title(self, row: dict) -> str:
        self._ensure_row_metadata(row)
        if not row.get("snippet_loaded") and not self.sessions.get(row["sid"], {}).get("title"):
            return self.sessions.get(row["sid"], {}).get("label") or "Loading conversation…"
        return conversation_title(self.sessions.get(row["sid"], {}), row.get("snippet", ""))

    def _ensure_row_metadata(self, r: dict) -> None:
        meta = self.sessions.get(r["sid"], {})
        if not r.get("snippet_loaded"):
            if self.background:
                self._request_index(r, priority=1)
                return
            r["snippet"] = self._first_user_snippet(meta)
            r["snippet_loaded"] = True

    def _build_search_text(self, sid: str) -> str:
        meta = self.sessions.get(sid, {})
        hp_str = meta.get("history_file")
        chunks: list[str] = []
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
                    chunks.append(flatten_content_text(item.get("content", "")))
        return "\n".join(chunks)

    def _search_text(self, sid: str) -> str:
        cached = self.search_text_cache.get(sid)
        if cached is not None:
            return cached
        if self.background:
            return ""
        text = self._build_search_text(sid)
        self.search_text_cache[sid] = text
        self.search_folded_cache[sid] = text.lower()
        return text

    def _start_prefetch(self) -> None:
        if self.prefetch_started or not self.background:
            return
        self.prefetch_started = True
        if self.title_cache is not None:
            paths = [(row["sid"], self.sessions[row["sid"]].get("history_file")) for row in self.rows]

            def load_titles(cancelled):
                result = []
                for sid, path in paths:
                    if cancelled():
                        break
                    snippet = self.title_cache.read(path)
                    if snippet is not None:
                        result.append((sid, path, snippet))
                self.title_cache.save()
                return result

            self.worker.submit("titles", load_titles, priority=-1)
        for row in self.rows:
            self._request_index(row)

    def _visible_rows_list(self) -> list[dict]:
        q = self.search_query.lower().strip()
        if q:
            self._start_prefetch()
        signature = (q, self.view_mode, frozenset(self.ghost_sids), self.search_revision,
                     tuple((r["sid"], r["archived"], r.get("snippet", ""),
                            self.sessions.get(r["sid"], {}).get("title", "")) for r in self.rows))
        if self.visible_cache is not None and self.visible_cache[0] == signature:
            return self.visible_cache[1]

        def keep(r: dict) -> bool:
            if r["sid"] in self.ghost_sids:
                return True
            if self.view_mode == "active" and r["archived"]:
                return False
            if self.view_mode == "archived" and not r["archived"]:
                return False
            if q:
                if q in self._fast_search_text(r):
                    return True
                if r["sid"] not in self.search_folded_cache and not self.background:
                    self._search_text(r["sid"])
                if q not in self.search_folded_cache.get(r["sid"], ""):
                    return False
            return True

        visible = [r for r in self.rows if keep(r)]
        self.visible_cache = (signature, visible)
        return visible

    def _selected_pos(self, visible: list[dict]) -> int:
        for i, r in enumerate(visible):
            if r["sid"] == self.selected_sid:
                return i
        return 0

    def _clear_selection(self) -> None:
        """Drop the Shift+arrow range back to a lone cursor."""
        self.anchor_sid = None
        self.marked_sids = set()
        self.persistent_selection = False

    def _target_rows(self, visible: list[dict]) -> list[dict]:
        """Rows an action applies to: the marked span, else the cursor row.

        Every row action goes through this, so ``a``/``d`` behave identically
        whether or not a range is active.
        """
        if self.marked_sids:
            marked = [r for r in visible if r["sid"] in self.marked_sids]
            if marked:
                return marked
        if not visible:
            return []
        return [visible[self._selected_pos(visible)]]

    def _extend_selection(self, key: str, repeat_count: int, visible: list[dict]) -> None:
        """Grow or shrink the Shift+arrow range, carrying the cursor with it."""
        if not visible:
            return
        self.persistent_selection = False
        sel = self._selected_pos(visible)
        if self.anchor_sid is None or not any(r["sid"] == self.anchor_sid for r in visible):
            self.anchor_sid = visible[sel]["sid"]
        nav = apply_selection_keys(
            SHIFT_SELECTION_KEYS[key],
            repeat_count,
            selected=sel,
            total=len(visible),
            page=self._max_vis(),
        )
        if nav is None:
            return
        self.selected_sid = visible[nav]["sid"]
        anchor_pos = next(
            (i for i, r in enumerate(visible) if r["sid"] == self.anchor_sid), nav
        )
        lo, hi = (anchor_pos, nav) if anchor_pos <= nav else (nav, anchor_pos)
        self.marked_sids = {r["sid"] for r in visible[lo:hi + 1]}
        self.ghost_sids = set()

    def _subtitle(self) -> str:
        count = len(self.rows)
        selected = f" · {len(self.marked_sids)} selected" if self.marked_sids else ""
        return f"{count} sessions{selected}"

    def _footer_lines(self, width: int, cur: dict | None) -> list[Text]:
        if self.rename_sid is not None:
            primary, secondary = "Enter Save  Esc Cancel", "Empty title uses the first prompt"
        elif self.arm_delete_sids:
            return [fitted(Text("d Confirm · any other key cancels", style="red"), width)]
        elif self.search_active:
            primary, secondary = "Enter / ↓ Results  Esc Clear", "←→ Edit text  Tab View"
        elif cur is None:
            primary, secondary = "Ctrl+F Search  Tab View  Esc Close", "? Help"
        else:
            action = "Restore & resume" if cur["archived"] else "Resume"
            if self.pane_focus == "preview":
                primary = f"↑↓ Scroll  ← Sessions  Enter {action}  Ctrl+F Search  Tab View  Esc Back"
                secondary = "PgUp/PgDn Page  Home/End Jump  p Expand  ? Help"
            else:
                primary = f"↑↓ Move  → Preview  Enter {action}  Ctrl+F Search  Tab View  Esc Close"
                secondary = (f"{len(self.marked_sids)} selected  a {'Restore' if cur['archived'] else 'Archive'}  d Delete  ? Help"
                             if self.marked_sids else "Space Select  a Archive  r Rename  p Expand  ? Help")
                if cur["archived"] and not self.marked_sids:
                    secondary = "Space Select  a Restore  r Rename  p Expand  ? Help"
            if Text(primary).cell_len > width:
                primary = f"Enter {action}  Ctrl+F Search  Esc Close"
            if Text(primary).cell_len > width:
                primary = "Enter Open  Ctrl+F Find  Esc" if width >= 28 else "↵ Open Ctrl+F Find Esc"
        line = fitted(Text(primary, style="dim"), width)
        if line.cell_len + Text(secondary).cell_len + 3 > width:
            secondary = secondary.replace("  ", " ")
        if line.cell_len + Text(secondary).cell_len + 3 > width:
            secondary = "? Help" if self.rename_sid is None and not self.search_active else ""
        if secondary and line.cell_len + len(secondary) + 2 <= width:
            line.append(" " * (width - line.cell_len - len(secondary)) + secondary, style="dim")
        return [line]

    def _build_preview_lines(self, sid: str, width: int) -> list[Text]:
        meta = self.sessions.get(sid, {})
        hp_str = meta.get("history_file")
        if not hp_str:
            return [Text("(no history file)", style="dim")]
        hp = Path(hp_str)
        if not hp.exists():
            return [Text("(history file missing)", style="dim")]
        # A history file deleted or corrupted mid-session must not crash the
        # browser from on_key; degrade to a placeholder line instead (matches
        # _first_user_snippet/_build_search_text).
        try:
            history = load_history(hp)
            if not history:
                return [Text("(empty conversation)", style="dim")]
            return _history_visual_lines(history, width) or [Text("(empty conversation)", style="dim")]
        except Exception:
            return [Text("(couldn't read history)", style="dim")]

    def _preview_lines(self, sid: str, width: int) -> list[Text]:
        if self.background:
            self._request_preview(sid, width)
            cached = self.preview_cache.get((sid, width))
            return cached[1] if cached else [Text("Loading preview…", style="dim")]
        path = self.sessions.get(sid, {}).get("history_file")
        try:
            stat = Path(path).stat() if path else None
        except OSError:
            stat = None
        stamp = (path, stat.st_mtime_ns if stat else None, stat.st_size if stat else None)
        key = (sid, width)
        cached = self.preview_cache.get(key)
        if cached is None or cached[0] != stamp:
            self.preview_cache[key] = (stamp, self._build_preview_lines(sid, width))
        return self.preview_cache[key][1]

    def _request_preview(self, sid: str, width: int, *, force=False):
        query = self.search_query.strip().lower()
        request = (sid, width, query)
        path = self.sessions.get(sid, {}).get("history_file")
        if self.preview_request == request and self.preview_request_path == path and not force:
            return
        self.worker.cancel("preview")
        for key in list(self.preview_transfers):
            if key != request:
                self.preview_transfers.pop(key)
        self.preview_request = request
        self.preview_request_path = path
        self.preview_checked_at = time.monotonic()
        checked = self.preview_validated.get((sid, width), 0)
        if not force and self._cached_preview(request) is not None and self.preview_checked_at - checked < 1:
            # A freshly prepared neighbour needs no extra I/O on selection.
            # Tick still revalidates external edits after one second.
            self.preview_checked_at = checked
            return
        if self.preview_prefetch.get(request) == path and request in self.preview_prefetch:
            self.worker.prioritize(("warm_preview", *request), -2)
            return
        # Refreshes can wait for the current neighbour to finish: its cached
        # contents are already visible. Only a missing preview preempts work.
        priority = 0 if self._cached_preview(request) is not None else -2
        self._submit_preview(request, path, key="preview", priority=priority)

    def _submit_preview(self, request, path, *, key, priority):
        sid, width, query = request
        cached = self.preview_cache.get((sid, width))
        cached_document = self.preview_documents.get(request)

        def prepare(cancelled):
            record = self._read_document(path)
            if cancelled():
                return None
            if cached is not None and cached[0] == record[0]:
                lines = cached[1]
            elif record[-1]:
                lines = [Text(record[-1], style="dim")]
            else:
                lines = _history_visual_lines(record[1], width, cancelled=cancelled) or [Text("(empty conversation)", style="dim")]
            if cancelled():
                return None
            if cached_document is not None and cached is not None and cached[0] == record[0]:
                document, match = cached_document
            else:
                document, match = highlighted_transcript(lines, query)
            return sid, width, query, record, lines, document, match

        self.worker.submit(key, prepare, priority=priority, preemptible=key != "preview")

    def _prefetch_previews(self, visible, width):
        """Keep a small window ready, favouring the direction of travel."""
        if not self.background:
            return
        selected = self._selected_pos(visible)
        offsets = [step * direction for step in range(1, 11)
                   for direction in (self.preview_direction, -self.preview_direction)
                   if direction == self.preview_direction or step <= 3]
        requests = [self._preview_key(visible[selected + offset]["sid"], width)
                    for offset in offsets if 0 <= selected + offset < len(visible)] if width else []
        wanted = set(requests)
        if self.preview_request is not None:
            wanted.add(self.preview_request)
        for request, path in list(self.preview_prefetch.items()):
            if request not in wanted or self.sessions.get(request[0], {}).get("history_file") != path:
                self.worker.cancel(("warm_preview", *request))
                self.preview_prefetch.pop(request)
        self.preview_prefetch_attempts = {request: path for request, path in self.preview_prefetch_attempts.items()
                                          if request in wanted}
        for rank, request in enumerate(requests):
            sid = request[0]
            path = self.sessions.get(sid, {}).get("history_file")
            cached = self.preview_cache.get(request[:2])
            if cached is not None and cached[0][0] == path and request in self.preview_documents:
                continue
            key = ("warm_preview", *request)
            priority = rank / 20
            if request in self.preview_prefetch:
                self.worker.prioritize(key, priority)
            elif request not in self.preview_prefetch_attempts or self.preview_prefetch_attempts[request] != path:
                # Do not repeatedly rebuild oversized or failed neighbours
                # on every frame. Selecting one still requests it immediately.
                self.preview_prefetch_attempts[request] = path
                self.preview_prefetch[request] = path
                self._submit_preview(request, path, key=key, priority=priority)

    def _cached_preview(self, request):
        cached = self.preview_cache.get(request[:2])
        if cached is not None and cached[0][0] == self.sessions.get(request[0], {}).get("history_file"):
            return self.preview_documents.get(request)
        return None

    def _preview_document(self, sid: str, width: int) -> tuple[list[Text], int | None]:
        if self.background:
            self._request_preview(sid, width)
            key = self._preview_key(sid, width)
            cached = self._cached_preview(key)
            if cached is not None:
                self.preview_cache.move_to_end((sid, width))
                self.preview_documents.move_to_end(key)
                return cached
            return [Text("Loading preview…", style="dim")], None
        lines = self._preview_lines(sid, width)
        key = (sid, width, self.search_query, id(lines))
        if self.preview_document_cache is None or self.preview_document_cache[0] != key:
            self.preview_document_cache = (key, *highlighted_transcript(lines, self.search_query))
        return self.preview_document_cache[1:]

    def _preview_key(self, sid: str, width: int) -> tuple[str, int, str]:
        return sid, width, self.search_query.strip().lower()

    def _preview_start(self, lines: list[Text], match: int | None, height: int) -> int:
        if match is not None:
            return max(0, match - 1)
        start = next((i for i in range(len(lines) - 1, -1, -1)
                      if lines[i].plain.startswith("user: ")), 0)
        # Long piped prompts should not hide the answer on first opening.
        answer = next((i for i in range(start + 1, len(lines)) if lines[i].plain == "jarv:"), None)
        return answer if answer is not None and answer - start > max(3, height // 3) else start

    def _preview_window(self, sid: str, width: int, height: int) -> tuple[list[Text], int, int]:
        lines, match = self._preview_document(sid, width)
        key = self._preview_key(sid, width)
        if self.background and self._cached_preview(key) is None:
            return lines[:height], 0, 0
        initial = self._preview_start(lines, match, height)
        # Keep the latest exchange at the top, even when there is room below
        # it. Padding the scroll extent avoids jumping back to older turns.
        extent = max(len(lines), initial + height)
        start = clamp_scroll_offset(self.preview_positions.get(key, initial), extent, max(1, height))
        self.preview_positions[key] = start
        return lines[start:start + height], start, len(lines)

    def _scroll_preview(self, sid: str, width: int, height: int, key: str, repeat: int) -> None:
        _, offset, total = self._preview_window(sid, width, height)
        if self.background and self._cached_preview(self._preview_key(sid, width)) is None:
            return
        lines, match = self._preview_document(sid, width)
        extent = max(total, self._preview_start(lines, match, height) + height)
        self.preview_positions[self._preview_key(sid, width)] = apply_scroll_keys(
            key, repeat, offset=offset, total=extent, body_rows=height,
        )

    def _full_preview_geometry(self) -> tuple[int, int, int, int]:
        term_w, term_h = terminal_size(console=console)
        width = panel_width(term_w)
        body_rows, _ = body_content_rows(term_h)
        return width, term_h, menu_inner_width(width), max(1, body_rows - 1)

    def _render_preview(self) -> Panel:
        width, term_h, inner_width, body_rows = self._full_preview_geometry()
        sid = self.preview_sid or ""
        lines, start, total = self._preview_window(sid, inner_width, body_rows)
        meta = self.sessions.get(sid, {})
        row = next((r for r in self.rows if r["sid"] == sid), None)
        title = self._title(row) if row is not None else meta.get("title", "Untitled conversation")
        parts = [fitted(Text(title, style="bold"), inner_width), *lines]
        if self.flash is not None:
            parts = parts[:max(1, body_rows - 1)]
            parts.extend(Text("") for _ in range(max(0, body_rows - 1 - len(parts))))
            parts.append(fitted(Text(self.flash[0], style=self.flash[1]), inner_width))
        if term_h >= 6:
            position = scroll_position_hint(start, min(total, start + body_rows), total)
            action = "Restore & resume" if meta.get("archived") else "Resume"
            options = [
                f"↑↓ Scroll   ← Back   p Collapse   Enter {action}   Esc Back   ·   {position}",
                f"↑↓ Scroll   ← Back   Enter {action}   Esc Back",
                "↑↓ Scroll  ↵ Restore+resume  Esc Back" if meta.get("archived") else "↑↓ Scroll  Enter Resume  Esc Back",
                "↕ Scroll ↵ Restore+resume Esc" if meta.get("archived") else "↕ Scroll ↵ Resume Esc Back",
            ]
            controls = next((value for value in options if Text(value).cell_len <= inner_width), "↑↓ Scroll  Esc Back")
            append_bottom_footer(parts, term_h, Text(controls, style="dim"))
        return jarv_panel(Group(*parts), "preview", subtitle=_short_session_id(sid) or None,
                          padding=(0, 1), width=width, height=term_h)

    def _search_bar(self, width: int) -> Text:
        line = Text("Search: ", style="bold cyan" if self.search_active else "dim")
        indexed = sum(sid in self.indexed_paths for sid in self.sessions)
        remaining = len(self.sessions) - indexed
        indexing = self.background and self.search_query and remaining > 0
        hint = f"Searching… {indexed}/{len(self.sessions)}" if indexing else "Ctrl+F Search"
        room = max(1, width - len(hint) - line.cell_len - 2) if width >= 60 else max(1, width - line.cell_len)
        if self.search_active:
            line.append_text(render_single_line(self.search_editor, room, text_style="bold"))
        else:
            line.append_text(fitted(Text(self.search_query or "conversations…", style="bold" if self.search_query else "dim"), room))
        if width >= 60 and line.cell_len + len(hint) + 2 <= width:
            line.append(" " * (width - line.cell_len - len(hint)) + hint, style="dim")
        return fitted(line, width)

    def _tabs(self, width: int) -> Text:
        active = sum(not r["archived"] for r in self.rows)
        counts = {"active": active, "archived": len(self.rows) - active, "all": len(self.rows)}
        compact = width < 48
        line = Text()
        for index, mode in enumerate(self.VIEW_MODES):
            count = counts[mode]
            label = (mode.title() if not compact else {"active": "Act", "archived": "Arc", "all": "All"}[mode])
            label = f"{label} {count}"
            if index:
                line.append("  " if compact else "   ")
            line.append(f"[{label}]" if mode == self.view_mode else label,
                        style="bold cyan" if mode == self.view_mode else "dim")
        return fitted(line, width)

    def _status_lines(self, width: int, visible: list[dict]) -> list[Text]:
        if self.rename_sid is not None:
            line = Text("Rename: ", style="bold cyan")
            line.append_text(render_single_line(self.rename_editor, max(1, width - 8), text_style="bold"))
            return [fitted(line, width)]
        targets = self._target_rows(visible)
        if self.arm_delete_sids and self.arm_delete_sids == {r["sid"] for r in targets}:
            if len(targets) == 1:
                name = self._truncate(self._title(targets[0]), max(1, width - 24))
                question = f'Delete "{name}" permanently?'
            else:
                question = f"Delete {len(targets)} sessions permanently?"
            return [fitted(Text(question, style="bold red"), width)]
        if self.flash is None and not self.undo_actions:
            return []
        message, style = self.flash or ("", "dim")
        undo = ""
        if self.undo_actions:
            verb = {"did_archive": "archive", "did_unarchive": "restore", "did_delete": "delete"}[self.undo_actions[-1]["kind"]]
            undo = f"u Undo {verb}"
            if len(undo) + 16 > width:
                undo = "u Undo"
        if style == "red" and width < 60:
            undo = ""  # Give the failure reason priority on small terminals.
        line = fitted(Text(message, style=style), max(0, width - len(undo) - 2) if undo else width)
        if undo:
            line.append(" " * max(2, width - line.cell_len - len(undo)) + undo, style="bold cyan")
        return [fitted(line, width)]

    def _list_lines(self, visible: list[dict], width: int, height: int) -> list[Text]:
        if not visible:
            pending = self.background and self.search_query and any(sid not in self.indexed_paths for sid in self.sessions)
            message = ("Searching conversations…" if pending else
                       "No conversations match your search." if self.search_query else
                       "No archived conversations." if self.view_mode == "archived" else
                       "No active conversations." if self.view_mode == "active" and self.rows else
                       "No conversations yet.")
            hint = "Esc Clear search · Tab Change view" if self.search_query else "Tab Change view" if self.rows else "Start chatting to create a session."
            return [fitted(Text(message, style="dim"), width), fitted(Text(hint, style="dim"), width)][:height]
        # Offsets count physical lines, including group labels and search excerpts.
        entries: list[tuple[str, object]] = []
        cursor_line = 0
        previous_group = None
        group_dates = height >= 8 and not self.search_query
        excerpts = bool(self.search_query) and height >= 5
        selected = self._selected_pos(visible)
        for index, row in enumerate(visible):
            group = row.get("date_group", "Recent")
            if group_dates and group != previous_group:
                entries.append(("group", group))
                previous_group = group
            if index == selected:
                cursor_line = len(entries)
            entries.append(("row", row))
            if excerpts:
                entries.append(("excerpt", row))
        self.offset = clamp_selection_scroll(self.offset, cursor_line, len(entries), max(1, height))
        window = entries[self.offset:self.offset + height]
        if window and window[-1][0] == "group":
            window.pop()  # Keep a date heading with at least one conversation.
        self._last_list_capacity = max(1, sum(kind == "row" for kind, _ in window))
        lines = []
        for kind, value in window:
            if kind == "group":
                lines.append(fitted(Text("  " + str(value), style="dim"), width))
                continue
            row = value
            if kind == "excerpt":
                key = (row["sid"], self.search_query, width, self.indexed_paths.get(row["sid"]))
                excerpt = self.excerpt_cache.get(key)
                if excerpt is None:
                    excerpt = match_excerpt(self._search_text(row["sid"]), self.search_query, max(1, width - 4))
                    self._bounded_put(self.excerpt_cache, key, excerpt, limit=128)
                lines.append(fitted(highlighted("    " + (excerpt or "Matched session details"), self.search_query, "dim"), width))
                continue
            lines.append(session_row(row, self._title(row), width,
                                     selected=row["sid"] == self.selected_sid,
                                     focused=row["sid"] == self.selected_sid and self.pane_focus == "sessions" and not self.search_active and self.rename_sid is None,
                                     marked=row["sid"] in self.marked_sids, selecting=bool(self.marked_sids),
                                     query=self.search_query, armed=bool(self.arm_delete_sids and row["sid"] in self.arm_delete_sids)))
        return lines

    def _preview_messages(self, row: dict) -> tuple[list[tuple[str, str]], str]:
        meta = self.sessions.get(row["sid"], {})
        history = []
        unavailable = ""
        try:
            path = Path(meta["history_file"]) if meta.get("history_file") else None
            if path is None or not path.exists():
                unavailable = "History file is unavailable."
            else:
                stat = path.stat()
                key = (row["sid"], str(path), stat.st_mtime_ns, stat.st_size)
                if self.detail_cache is not None and self.detail_cache[0] == key:
                    return self.detail_cache[1], ""
                history = load_history(path)
        except Exception:
            unavailable = "Couldn't read this conversation."
        messages = [(item.get("role"), flatten_content_text(item.get("content", "")))
                    for item in history if isinstance(item, dict) and item.get("role") in ("user", "assistant")]
        messages = [(role, text) for role, text in messages if text.strip()]
        if not unavailable:
            self.detail_cache = (key, messages)
        return messages, unavailable

    def _detail_header(self, row: dict | None, width: int) -> list[Text]:
        if row is None:
            return []
        if self.background:
            exchanges = self.exchange_counts.get(row["sid"])
        else:
            messages, _ = self._preview_messages(row)
            exchanges = sum(role == "user" for role, _ in messages)
        title = fitted(Text(self._title(row), style="bold"), width)
        state = " · Archived" if row.get("archived") else " · Current" if row.get("is_current") else ""
        count = f" · {exchanges} exchange{'s' if exchanges != 1 else ''}" if exchanges is not None else ""
        metadata = f"{row.get('time_str', '—')}{count}{state}"
        return [title, fitted(Text(metadata, style="dim"), width), Text("")]

    def _detail_body_height(self, row: dict | None, width: int, height: int) -> int:
        return max(1, height - len(self._detail_header(row, width)) - 2)

    def _detail_lines(self, row: dict | None, width: int, height: int) -> list[Text]:
        if row is None:
            return [Text("No conversation selected.", style="dim")]
        header = self._detail_header(row, width)
        body_height = self._detail_body_height(row, width, height)
        lines, start, total = self._preview_window(row["sid"], width, body_height)
        parts = header + lines
        parts.extend(Text("") for _ in range(max(0, height - 2 - len(parts))))
        position = scroll_position_hint(start, min(total, start + body_height), total) if total else ""
        hint = "↑↓ Scroll · ← Sessions" if self.pane_focus == "preview" else "→ Preview"
        footer = Text(hint, style="cyan" if self.pane_focus == "preview" else "dim")
        if footer.cell_len + len(position) + 2 <= width:
            footer.append(" " * (width - footer.cell_len - len(position)) + position, style="dim")
        parts.extend([footer, fitted(Text("ID " + row["sid"], style="dim"), width)])
        return parts[:height]

    def _render_help(self, width: int, height: int) -> Panel:
        inner = menu_inner_width(width)
        shortcuts = [
            "Navigate", "↑↓ / mouse wheel    Navigate or scroll the active pane",
            "← / →    Switch between sessions and preview",
            "PgUp/PgDn · Home/End    Jump through the active pane",
            "Enter    Resume the cursor row (restore it first if archived)",
            "Ctrl+F    Search titles, IDs and conversation text",
            "Tab    Cycle Active → Archived → All (left to right)",
            "Esc / q    Return to sessions, clear selection/search, then close",
            "", "Manage (sessions pane)", "r    Rename · an empty title restores the prompt-based title",
            "Space    Toggle a row; selections stay as you move",
            "Shift+↑↓    Select a range; a plain arrow ends a range",
            "a    Archive / restore selected rows (cursor sets direction)",
            "d, then d    Delete the named conversation or selection",
            "u    Undo actions, most recent first, until this browser closes",
            "", "Preview", "→    Read the preview · ← returns to sessions",
            "p    Expand to full screen · p/←/Esc returns",
            "↑↓ / wheel / PgUp/PgDn / Home/End    Scroll the transcript",
            "On a narrow terminal, → opens the full-screen preview",
            "Enter    Resume the conversation being previewed",
        ]
        lines = []
        for value in shortcuts:
            lines.extend(Text(value, style="bold cyan" if value in ("Navigate", "Manage (sessions pane)", "Preview") else "").wrap(self.console, inner))
        self.help_total = len(lines)
        capacity = max(1, height - menu_frame_rows() - 2)
        self.help_offset = clamp_scroll_offset(self.help_offset, len(lines), capacity)
        parts = lines[self.help_offset:self.help_offset + capacity]
        append_bottom_footer(parts, height, fitted(Text("↑↓ Scroll   ? / Esc Back", style="dim"), inner), crop=True)
        return self._panel(parts, width, height, "sessions · shortcuts")

    def _panel(self, parts, width: int, height: int, title: str = "sessions") -> Panel:
        return MenuPanel(Group(*parts), title=f"[bold bright_white]jarv ▸ {title}[/bold bright_white]",
                         title_align="left", subtitle=self._subtitle(), subtitle_align="right",
                         border_style="cyan", box=box.ROUNDED, padding=(0, 1), width=width, height=height)

    def _list_layout(self, visible: list[dict]) -> dict:
        term_w, term_h = terminal_size(console=console)
        width = panel_width(term_w)
        inner = menu_inner_width(width)
        cur = visible[self._selected_pos(visible)] if visible else None
        available = max(1, term_h - menu_frame_rows())
        # Keep one status row even when idle. Reuse the old header spacer so
        # notifications, confirmations and undo never move the list/preview.
        status = self._status_lines(inner, visible) or [Text("")]
        header = [self._tabs(inner), self._search_bar(inner)]
        footer = self._footer_lines(inner, cur)
        if term_h < 12:
            footer = footer[-1:]
        if available < len(header) + len(status) + len(footer) + 1:
            header = header[:max(0, available - len(status) - len(footer) - 1)]
        body_height = max(1, available - len(header) - len(status) - len(footer))
        split = term_w >= 100 and body_height >= 9
        if self.pane_focus == "preview" and (not split or cur is None):
            self.pane_focus = "sessions"
            footer = self._footer_lines(inner, cur)
            if term_h < 12:
                footer = footer[-1:]
        # Titles need less space than a transcript. Give the preview the extra
        # width, while keeping very wide terminals from stretching the list.
        left_width = min(56, (inner - 3) * 2 // 5) if split else inner
        return dict(width=width, height=term_h, inner=inner, header=header, status=status, footer=footer,
                    body_height=body_height, split=split, left_width=left_width,
                    right_width=inner - left_width - 3 if split else 0, current=cur)

    def on_resize(self, size: tuple[int, int]) -> None:
        if self.preview_sid is None:
            self._list_layout(self._visible_rows_list())

    def render(self) -> Panel:
        term_w, term_h = terminal_size(console=console)
        if self.help_open:
            return self._render_help(panel_width(term_w), term_h)
        if self.preview_sid is not None:
            return self._render_preview()
        visible = self._visible_rows_list()
        layout = self._list_layout(visible)
        cur = layout["current"]
        if cur is not None:
            self.selected_sid = cur["sid"]
        body_height, list_width = layout["body_height"], layout["left_width"]
        show_heading = body_height >= 8
        count = (f"{len(visible)} result{'s' if len(visible) != 1 else ''}" if self.search_query else
                 f"{self._selected_pos(visible) + 1 if visible else 0} of {len(visible)}")
        list_active = self.pane_focus == "sessions" and not self.search_active and self.rename_sid is None
        heading = pane_heading("SESSIONS", list_width, active=list_active, count=count)
        left = ([heading] if show_heading else []) + self._list_lines(visible, list_width, body_height - int(show_heading))
        if layout["split"]:
            right_width = layout["right_width"]
            preview_active = self.pane_focus == "preview" and not self.search_active
            preview_heading = pane_heading("PREVIEW", right_width, active=preview_active)
            right = [preview_heading] + self._detail_lines(cur, right_width, body_height - 1)
            body = beside(left, right, list_width, right_width, body_height)
        else:
            body = left[:body_height]
            body.extend(Text("") for _ in range(body_height - len(body)))
        self._prefetch_previews(visible, layout["right_width"])
        parts = layout["header"] + body + layout["status"] + layout["footer"]
        return self._panel([fitted(line, layout["inner"]) for line in parts], layout["width"], term_h)

    # ------------------------------------------------------------------ #
    # Undo lasts until the browser closes. Delete files only when leaving.
    # ------------------------------------------------------------------ #
    def _finalize_action(self, action: dict) -> None:
        if action["kind"] == "did_delete":
            for entry in action["entries"]:
                hp_str = entry.get("history_path")
                if hp_str:
                    # A terminal can create a fresh session with this identity
                    # while the browser is open. Never remove its new files.
                    with session_metadata_transaction(Path(hp_str), self.data):
                        current = load_sessions()["sessions"]
                        reused = entry["sid"] in current or any(
                            meta.get("history_file") == hp_str for meta in current.values()
                        )
                        if not reused:
                            delete_session_files(Path(hp_str))

    def _commit_pending(self) -> None:
        for action in self.undo_actions:
            self._finalize_action(action)
        self.undo_actions.clear()

    def _start_undo(self, action: dict) -> None:
        self.undo_actions.append(action)

    def _take_last_action(self) -> dict | None:
        return self.undo_actions.pop() if self.undo_actions else None

    def _do_undo(self) -> tuple[tuple[str, str], list[str]] | None:
        """Returns ((flash_msg, flash_style), restored_sids) or None."""
        action = self._take_last_action()
        if action is None:
            return None
        kind = action["kind"]
        if kind in ("did_archive", "did_unarchive"):
            restored: list[str] = []
            failed: list[str] = []
            reason = ""
            for sid in action["sids"]:
                row = next((r for r in self.rows if r["sid"] == sid), None)
                if row is None:
                    continue
                try:
                    moved = (self._unarchive_row(
                        row, terminal_bindings=action.get("terminal_bindings", {}),
                        is_current=sid in action.get("current_sids", ()),
                    ) if kind == "did_archive" else self._archive_row(row))
                    if not moved:
                        reason = "history files are missing or empty"
                except (OSError, StorageError) as exc:
                    moved, reason = False, str(exc.__cause__ or exc)
                if moved:
                    restored.append(sid)
                else:
                    failed.append(sid)
            if failed:
                # Retain only unfinished work so another u can retry it.
                action["sids"] = failed
                self.undo_actions.append(action)
                message = (f"Undid {len(restored)}; couldn't undo {len(failed)}: {reason}" if restored else
                           f"Couldn't undo: {reason}")
                return ((message, "yellow" if restored else "red"), restored)
            if not restored:
                return (("○ nothing left to undo", "dim"), [])
            label = self._batch_label(restored)
            if kind == "did_archive":
                return ((f"↺ restored {label}", "green"), restored)
            return ((f"↺ archived {label}", "cyan"), restored)
        if kind == "did_delete":
            # Ascending row_index so each insert lands where the row used to be.
            entries = sorted(action["entries"], key=lambda e: e["row_index"])
            for entry in entries:
                sid = entry["sid"]
                self.sessions[sid] = entry["meta"]
                for term_id in entry["removed_terminals"]:
                    self.terminals[term_id] = sid
                row_index = entry["row_index"]
                if 0 <= row_index <= len(self.rows):
                    self.rows.insert(row_index, entry["row"])
                else:
                    self.rows.append(entry["row"])
                if self.background:
                    self._request_index(entry["row"])
            save_sessions(self.data)
            sids = [e["sid"] for e in entries]
            return ((f"↺ restored {self._batch_label(sids)}", "green"), sids)
        return None

    # ------------------------------------------------------------------ #
    # Row actions
    # ------------------------------------------------------------------ #
    def _activate_row(self, row: dict) -> None:
        if row["archived"]:
            try:
                if not self._unarchive_row(row):
                    self._notify("Archived session files are missing; session was not loaded.", "red")
                    return
            except (OSError, StorageError) as exc:
                self._notify(f"Couldn't restore session: {exc.__cause__ or exc}", "red")
                return
            self.auto_restored = True
        set_terminal_session(row["sid"])
        self.loaded_row = row
        self.stop()

    def _batch_label(self, sids) -> str:
        """Flash wording: a conversation title for one row, a count for a batch."""
        sids = list(sids)
        if len(sids) == 1:
            row = next((r for r in self.rows if r["sid"] == sids[0]), None)
            return self._title(row) if row else _short_session_id(sids[0])
        return f"{len(sids)} sessions"

    def _archive_row(self, row: dict) -> bool:
        """Move one session into the archive. False when there was nothing to move."""
        sid = row["sid"]
        meta = self.sessions.get(sid, {})
        hp_str = meta.get("history_file")
        if not hp_str:
            return False
        with session_metadata_transaction(Path(hp_str), self.data):
            archived_path = archive_session_files(Path(hp_str))
            if archived_path is None:
                return False
            mark_session_archived(self.data, sid, archived_path)
            save_sessions(self.data)
        row["archived"] = True
        row["is_current"] = False
        return True

    def _unarchive_row(self, row: dict, *, terminal_bindings=None, is_current=False) -> bool:
        """Restore files and metadata together; leave missing archives unchanged."""
        sid = row["sid"]
        meta = self.sessions.get(sid, {})
        hp_str = meta.get("history_file")
        if not hp_str:
            return False
        with session_metadata_transaction(Path(hp_str), self.data):
            restored = unarchive_session_files(Path(hp_str), sid)
            if restored is None:
                return False
            meta["history_file"] = str(restored)
            meta.pop("archived", None)
            meta.pop("archived_at", None)
            if terminal_bindings:
                latest_terminals = load_sessions()["terminals"]
                for terminal, mapped_sid in terminal_bindings.items():
                    if mapped_sid == sid and terminal not in latest_terminals:
                        self.terminals.setdefault(terminal, sid)
            save_sessions(self.data)
        row["archived"] = False
        if is_current and sid in self.terminals.values():
            row["is_current"] = True
        return True

    def _reconcile_action_selection(self, previous: list[dict], changed: list[str]) -> None:
        """Keep the nearest surviving row selected after changing a filter."""
        self.ghost_sids.clear()
        visible = self._visible_rows_list()
        visible_sids = {row["sid"] for row in visible}
        if self.selected_sid not in visible_sids:
            old_pos = self._selected_pos(previous)
            # Prefer the next surviving row; at the end use the preceding one.
            after = [r for r in previous[old_pos:] if r["sid"] in visible_sids]
            before = [r for r in previous[:old_pos] if r["sid"] in visible_sids]
            row = after[0] if after else before[-1] if before else visible[0] if visible else None
            self.selected_sid = row["sid"] if row else None
        self.marked_sids.difference_update(changed)
        self.marked_sids.intersection_update(visible_sids)
        self.anchor_sid = None
        if not self.marked_sids:
            self._clear_selection()

    def _archive_selection(self, rows: list[dict], cur: dict | None) -> None:
        """Archive or unarchive every targeted row as one undoable action.

        The cursor row picks the direction and rows already in that state are
        skipped, so a span mixing active and archived sessions can't half-flip.
        """
        if cur is None or not rows:
            return
        unarchiving = cur["archived"]
        previous = self._visible_rows_list()
        target_sids = {row["sid"] for row in rows}
        terminal_bindings = {terminal: sid for terminal, sid in self.terminals.items() if sid in target_sids}
        current_sids = {row["sid"] for row in rows if row.get("is_current")}
        moved: list[str] = []
        failed: list[str] = []
        reason = ""
        for row in rows:
            if row["archived"] != unarchiving:
                continue
            try:
                success = self._unarchive_row(row) if unarchiving else self._archive_row(row)
                if not success:
                    reason = "files are missing" if unarchiving else "history is missing or empty"
            except (OSError, StorageError) as exc:
                success, reason = False, str(exc.__cause__ or exc)
            if success:
                moved.append(row["sid"])
            else:
                failed.append(row["sid"])

        verb = "restore" if unarchiving else "archive"
        if moved:
            self._start_undo({"kind": "did_unarchive" if unarchiving else "did_archive", "sids": moved,
                              "terminal_bindings": terminal_bindings, "current_sids": current_sids})
            self._reconcile_action_selection(previous, moved)
        if failed:
            message = (f"{verb.title()}d {len(moved)}; couldn't {verb} {len(failed)}: {reason}" if moved else
                       f"Couldn't {verb}: {reason} · {self._batch_label(failed)}")
            self._notify(message, "yellow" if moved else "red")
        elif moved:
            self._notify(f"✓ {verb}d {self._batch_label(moved)}", "green" if unarchiving else "cyan")
        else:
            self._notify(f"Selected sessions are already {'active' if unarchiving else 'archived'}.", "dim")

    def _handle_delete_key(self, rows: list[dict], visible: list[dict]) -> None:
        if not rows:
            return
        target_sids = frozenset(r["sid"] for r in rows)
        if self.arm_delete_sids != target_sids:
            # First press, or the target set moved since the last one: (re)arm
            # rather than deleting something the prompt never named.
            self.arm_delete_sids = target_sids
            return

        label = self._batch_label(target_sids)
        entries: list[dict] = []
        for row in rows:
            sid = row["sid"]
            meta = self.sessions.get(sid, {})
            removed_terminals: list[str] = []
            for term_id, mapped_sid in list(self.terminals.items()):
                if mapped_sid == sid:
                    removed_terminals.append(term_id)
                    self.terminals.pop(term_id)
            entries.append({
                "sid": sid,
                "row": dict(row),
                "meta": dict(meta),
                "row_index": next(
                    (i for i, r in enumerate(self.rows) if r["sid"] == sid), len(self.rows)
                ),
                "removed_terminals": removed_terminals,
                "history_path": meta.get("history_file"),
            })
            self.sessions.pop(sid, None)

        # Land the cursor where the top of the deleted span used to be.
        first_pos = min(
            (i for i, r in enumerate(visible) if r["sid"] in target_sids),
            default=0,
        )
        self.rows[:] = [r for r in self.rows if r["sid"] not in target_sids]
        save_sessions(self.data)
        self._clear_selection()
        self.ghost_sids = set()
        new_visible = self._visible_rows_list()
        if new_visible:
            self.selected_sid = new_visible[min(first_pos, len(new_visible) - 1)]["sid"]
        else:
            self.selected_sid = None
        self._notify(f"✓ deleted {label}", "green")
        self.arm_delete_sids = None
        self._start_undo({"kind": "did_delete", "entries": entries})

    # ------------------------------------------------------------------ #
    # Key handling
    # ------------------------------------------------------------------ #
    def _open_preview(self, row: dict) -> None:
        self.preview_return_focus = self.pane_focus
        layout = self._list_layout(self._visible_rows_list())
        if layout["split"]:
            _, _, width, _ = self._full_preview_geometry()
            self._transfer_preview_position(row["sid"], layout["right_width"], width)
        self.preview_sid = row["sid"]

    def _close_preview(self) -> None:
        sid = self.preview_sid
        _, _, width, _ = self._full_preview_geometry()
        self.preview_sid = None
        self.pane_focus = self.preview_return_focus
        layout = self._list_layout(self._visible_rows_list())
        if sid is not None and layout["split"]:
            self._transfer_preview_position(sid, width, layout["right_width"])

    def _transfer_preview_position(self, sid: str, source_width: int, target_width: int) -> None:
        source_key = self._preview_key(sid, source_width)
        if source_key not in self.preview_positions or source_width == target_width:
            return
        offset = self.preview_positions[source_key]
        previous = self.preview_roundtrip
        if previous is not None and previous[:3] == (sid, target_width, source_width) and previous[4] == offset:
            self.preview_positions[self._preview_key(sid, target_width)] = previous[3]
            self.preview_roundtrip = None
            return
        if self.background:
            source = self.preview_documents.get(source_key)
            if source is None:
                return
            target_key = self._preview_key(sid, target_width)
            target = self.preview_documents.get(target_key)
            if target is None:
                self.preview_transfers[target_key] = (source[0], offset, source_width)
                self._request_preview(sid, target_width)
            else:
                mapped = reflow_position(source[0], offset, target[0])
                self.preview_positions[target_key] = mapped
                self.preview_roundtrip = (sid, source_width, target_width, offset, mapped)
            return
        source, _ = self._preview_document(sid, source_width)
        target, _ = self._preview_document(sid, target_width)
        mapped = reflow_position(source, offset, target)
        self.preview_positions[self._preview_key(sid, target_width)] = mapped
        self.preview_roundtrip = (sid, source_width, target_width, offset, mapped)

    def _cycle_view(self) -> None:
        index = self.VIEW_MODES.index(self.view_mode)
        self.view_mode = self.VIEW_MODES[(index + 1) % len(self.VIEW_MODES)]
        self.offset = 0
        self.pane_focus = "sessions"
        self.ghost_sids.clear()
        self._clear_selection()
        visible = self._visible_rows_list()
        if visible and not any(row["sid"] == self.selected_sid for row in visible):
            self.selected_sid = visible[0]["sid"]

    def _on_key_rename(self, key: str, repeat: int) -> None:
        if not isinstance(key, TextInput) and key == "ESC":
            self.rename_sid = None
            return
        if not isinstance(key, TextInput) and key == "ENTER":
            meta = self.sessions.get(self.rename_sid)
            if meta is not None:
                previous = dict(meta)
                title = one_line(self.rename_editor["buffer"])[:240]
                if title:
                    meta["title"] = title
                else:
                    meta.pop("title", None)
                try:
                    save_sessions(self.data)
                except Exception as exc:
                    meta.clear()
                    meta.update(previous)
                    self._notify(f"Couldn't rename session: {exc}", "red")
                    self.rename_sid = None
                    return
                self._notify("✓ Session renamed" if title else "✓ Title reset to the first prompt", "green")
                # A title-only search must not make the renamed row disappear
                # before the user sees the result.
                self.ghost_sids.add(self.rename_sid)
            self.rename_sid = None
            return
        apply_text_editor_key(self.rename_editor, key, repeat)
        if len(self.rename_editor["buffer"]) > 240:
            self.rename_editor["buffer"] = self.rename_editor["buffer"][:240]
            self.rename_editor["cursor"] = min(self.rename_editor["cursor"], 240)

    def on_key(self, key: str, repeat: int) -> None:
        repeat_count = repeat

        if isinstance(key, TextInput) and not (
            self.rename_sid is not None or (self.search_active and self.preview_sid is None)
        ):
            self.arm_delete_sids = None
            return
        if self.rename_sid is not None:
            self._on_key_rename(key, repeat)
            return
        if self.help_open:
            if key in ("?", "ESC"):
                self.help_open = False
            else:
                _, height = terminal_size(console=console)
                self.help_offset = apply_scroll_keys(key, repeat, offset=self.help_offset,
                                                    total=self.help_total, body_rows=max(1, height - menu_frame_rows() - 2))
            return
        # Search-input mode intercepts most keys (only outside preview).
        if self.search_active and self.preview_sid is None:
            self._on_key_search(key, repeat)
            return

        # Preview mode intercepts most keys.
        if self.preview_sid is not None:
            self._on_key_preview(key, repeat_count)
            return

        if key == "ESC" and self.arm_delete_sids is not None:
            self.arm_delete_sids = None
            return

        if key != "d":
            self.arm_delete_sids = None

        visible = self._visible_rows_list()
        n_vis = len(visible)
        sel = self._selected_pos(visible) if visible else 0
        cur = visible[sel] if visible else None
        layout = self._list_layout(visible)

        if self.pane_focus == "preview":
            if key in ("LEFT", "ESC"):
                self.pane_focus = "sessions"
                return
            if key in SELECTION_KEYS or key.startswith("MOUSE_WHEEL_"):
                if cur is not None:
                    width = layout["right_width"]
                    height = self._detail_body_height(cur, width, layout["body_height"] - 1)
                    self._scroll_preview(cur["sid"], width, height, key, repeat_count)
                return
            # Row-management shortcuts belong to the sessions pane. Global
            # search, view, help, resume, expand, and undo remain available.
            if key not in ("ENTER", "p", "?", "CTRL_F", "TAB", "u"):
                return
        elif key == "RIGHT":
            if cur is not None:
                if layout["split"]:
                    self.pane_focus = "preview"
                else:
                    self._open_preview(cur)
            return
        elif key == "LEFT":
            return

        if key in SHIFT_SELECTION_KEYS:
            self.preview_direction = -1 if key == "SHIFT_UP" else 1
            self._extend_selection(key, repeat_count, visible)
            return

        nav_key = {
            "MOUSE_WHEEL_UP": "UP",
            "MOUSE_WHEEL_DOWN": "DOWN",
            "MOUSE_WHEEL_PAGEUP": "PAGEUP",
            "MOUSE_WHEEL_PAGEDOWN": "PAGEDOWN",
        }.get(key, key)
        if nav_key in SELECTION_KEYS:
            nav = apply_selection_keys(
                nav_key, repeat_count, selected=sel, total=n_vis, page=self._max_vis()
            )
            if nav is not None:
                if nav != sel:
                    self.preview_direction = 1 if nav > sel else -1
                self.selected_sid = visible[nav]["sid"]
            self.ghost_sids = set()
            # An unmodified move ends the range -- Shift is what holds it open.
            if not self.persistent_selection:
                self._clear_selection()
        elif key == "ENTER":
            if cur is not None:
                # Loading is inherently single-session, so the span goes away.
                self._clear_selection()
                self._activate_row(cur)
        elif key == "ESC":
            if self.marked_sids:
                self._clear_selection()
            elif self.search_query:
                self.search_query = ""
                self.offset = 0
                self.ghost_sids.clear()
            else:
                self.stop()
        elif key == "CTRL_F":
            self._start_prefetch()
            self._clear_selection()
            self.ghost_sids.clear()
            self.pane_focus = "sessions"
            initialize_text_editor(self.search_editor, self.search_query)
            self.search_active = True
        elif key == "TAB":
            self._cycle_view()
        elif key == "p":
            if cur is not None:
                self._open_preview(cur)
        elif key == "?":
            self.help_open = True
            self.help_offset = 0
        elif key == "r" and cur is not None:
            self.rename_sid = cur["sid"]
            initialize_text_editor(self.rename_editor, self._title(cur))
            self.rename_editor["selection_anchor"] = 0
        elif key == " " and cur is not None:
            self.anchor_sid = None
            self.persistent_selection = True
            if cur["sid"] in self.marked_sids:
                self.marked_sids.remove(cur["sid"])
            else:
                self.marked_sids.add(cur["sid"])
        elif key == "a":
            self._archive_selection(self._target_rows(visible), cur)
        elif key == "d":
            self._handle_delete_key(self._target_rows(visible), visible)
        elif key == "u":
            result = self._do_undo()
            if result is not None:
                notification, restored_sids = result
                self._notify(*notification)
                if restored_sids:
                    self.ghost_sids.clear()
                    first = next(r for r in self.rows if r["sid"] == restored_sids[0])
                    if self.view_mode != "all":
                        self.view_mode = "archived" if first["archived"] else "active"
                    self.selected_sid = restored_sids[0]
                    if not any(r["sid"] == self.selected_sid for r in self._visible_rows_list()):
                        self.search_query = ""
                    # Re-mark a restored batch so a follow-up action can retarget it.
                    if len(restored_sids) > 1:
                        self.anchor_sid = restored_sids[0]
                        self.marked_sids = set(restored_sids)
                    else:
                        self._clear_selection()
            else:
                self._notify("Nothing to undo.", "dim")

    def _on_key_search(self, key: str, repeat: int = 1) -> None:
        if not isinstance(key, TextInput) and key == "ESC":
            self.search_active = False
            self.search_query = ""
            self.offset = 0
            self.ghost_sids.clear()
            self._clear_selection()
        elif not isinstance(key, TextInput) and key == "TAB":
            self.search_active = False
            self._cycle_view()
        elif not isinstance(key, TextInput) and key in ("ENTER", "DOWN", "CTRL_F"):
            self.search_active = False
            visible = self._visible_rows_list()
            if visible and not any(r["sid"] == self.selected_sid for r in visible):
                self.selected_sid = visible[0]["sid"]
            self.offset = 0
        else:
            if apply_text_editor_key(self.search_editor, key, repeat):
                self.search_query = self.search_editor["buffer"]
                self.offset = 0
                self._clear_selection()
                visible = self._visible_rows_list()
                if visible and not any(r["sid"] == self.selected_sid for r in visible):
                    self.selected_sid = visible[0]["sid"]

    def _on_key_preview(self, key: str, repeat_count: int) -> None:
        if key in ("p", "ESC", "LEFT"):
            self._close_preview()
        elif key == "ENTER":
            row = next((r for r in self.rows if r["sid"] == self.preview_sid), None)
            if row is not None:
                self._activate_row(row)
        else:
            _, _, width, height = self._full_preview_geometry()
            self._scroll_preview(self.preview_sid or "", width, height, key, repeat_count)
