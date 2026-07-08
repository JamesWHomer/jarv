"""Alternate-screen heads-up mode UI."""

from __future__ import annotations

import argparse
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

from rich.cells import cell_len
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from .cancellation import CancellationToken, TurnCancelled
from .clipboard import copy_to_clipboard, read_clipboard_image, read_clipboard_text
from .command_input import (
    PasteRegistry,
    TextInput,
    _key_available,
    _read_key_with_repeats,
    disable_mouse_capture,
    enable_mouse_wheel_reporting,
    strip_sgr_mouse_sequences,
)
from .command_menu import MenuEntry, argument_entries, filter_entries, menu_entries
from .command_registry import COMMANDS, parse_command_alias
from .agent_ui import (
    _THINKING_FRAMES,
    STREAM_PREVIEW_REFRESH_INTERVAL,
    response_wait_label,
    thought_complete_indicator,
    tool_activity_label,
    tool_complete_indicator,
)
from .config import get_setting
from .display import (
    console,
    flatten_headings,
    rendered_text_lines,
    terminal_size,
    tool_card,
)
from .history import forget_current_session, load_history, prepare_session_context
from .intro_animation import render_intro
from .model_catalog import get_image_output_capability
from .session_render import (
    _history_content_to_str,
    _status_renderable,
    _tool_call_output,
    _tool_call_renderable,
)
from .text_editor import (
    apply_text_editor_key,
    initialize_text_editor,
    render_visual_line_window,
    selection_bounds,
)
from .tui_app import AltScreenApp
from .tui_frame import (
    FrameLayout,
    assemble_body,
    box_bottom,
    box_row,
    box_tab_top,
    box_top,
    build_frame,
    compose_subtitle,
    compose_title,
    compute_layout,
    overlay_menu,
    panel_width as _panel_width,
    transcript_rows as _transcript_rows_for,
    window_transcript,
)
from .tui_layout import clip_text
from .tui_overlay import scroll_key_delta
from .usage import format_cost, known_context_window, load_usage, usage_cost_summary, usage_file_for


SlashHandler = Callable[
    [str, list[str], dict, object, argparse.Namespace | None, bool],
    tuple[dict, object],
]
MaybeCommand = Callable[[str, list[str]], tuple[bool, str, list[str]] | None]

_FULLSCREEN_SLASH_COMMANDS = frozenset({
    "/about",
    "/config",
    "/help",
    "/history",
    "/settings",
    "/setup",
    "/usage",
})

# Rows the slash-command autocomplete menu shows at once before it windows
# around the selection and appends a "+N more" tail: roughly a third of the
# body on tall terminals, clamped between these two bounds.
_SLASH_MENU_MIN_ROWS = 6
_SLASH_MENU_MAX_ROWS = 10

# Key hints painted beside the popup's bottom row (in the row the regular
# footer hints occupy while the popup is closed). The short form steps in when
# the popup leaves little room; below that the row stays blank rather than
# showing hints cropped mid-word.
_SLASH_MENU_HINTS = "↑↓ select   Tab complete   Enter run   Esc close"
_SLASH_MENU_HINTS_SHORT = "↑↓ · Tab · Enter · Esc"

_COMMAND_CONFIRM_YES = frozenset({"1", "c", "cmd", "command", "run", "y", "yes"})
_SESSION_SWITCHING_SLASH_COMMANDS = frozenset({
    "/archive",
    "/new",
    "/session",
    "/sessions",
})
_HISTORY_SYNC_SLASH_COMMANDS = frozenset({
    "/redo",
    "/undo",
})
_HEADSUP_REPEATABLE_KEYS = frozenset({
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "CTRL_LEFT",
    "CTRL_RIGHT",
    "SHIFT_LEFT",
    "SHIFT_RIGHT",
    "CTRL_SHIFT_LEFT",
    "CTRL_SHIFT_RIGHT",
    "PAGEUP",
    "PAGEDOWN",
    "MOUSE_WHEEL_UP",
    "MOUSE_WHEEL_DOWN",
    "MOUSE_WHEEL_PAGEUP",
    "MOUSE_WHEEL_PAGEDOWN",
})
_SGR_MOUSE_TEXT_LOOKBACK = 64

# Length of the quick, non-blocking dissolve that plays when the idle intro is
# dismissed by the user's first message.
_OUTRO_DURATION = 0.4
_USAGE_STATUS_CACHE_TTL = 0.5

# Aqua used to light up a recognised slash command (the "/name" token) as it is
# typed in the input box, matching the cyan the autocomplete menu paints its
# commands with.
_VALID_COMMAND_STYLE = "cyan"

# A collapsed paste renders as a soft cyan chip so it reads as a placeholder
# token -- distinct from the white draft text, but in the same accent family as
# the input box's border. It is deleted and unboxed as a single unit.
_PASTE_MARKER_STYLE = "dim cyan"

# An active Shift/Ctrl+Shift selection paints as a solid cyan band so the
# selected run reads clearly against the white draft text, while the reverse
# cursor cell at the moving edge stays visible on top of it.
_SELECTION_STYLE = "black on cyan"


def _sanitize_editor_key(key: str) -> str:
    if not isinstance(key, TextInput):
        return key
    text = strip_sgr_mouse_sequences(str(key))
    if not text:
        return "OTHER"
    if text == key:
        return key
    return TextInput(text)


@dataclass
class TranscriptEntry:
    kind: str
    renderable: RenderableType
    spacer_before: bool = False
    _render_cache_width: int | None = field(default=None, init=False, repr=False)
    _render_cache_lines: list[Text] | None = field(default=None, init=False, repr=False)

    def rendered_lines(self, width: int) -> list[Text]:
        if self._render_cache_width == width and self._render_cache_lines is not None:
            return self._render_cache_lines
        lines = rendered_text_lines(self.renderable, width)
        self._render_cache_width = width
        self._render_cache_lines = lines
        return lines

    def invalidate(self) -> None:
        """Drop cached lines so the next render recomputes from the live renderable.

        Most entries are immutable once rendered, but a live tool card keeps the
        same entry while its clock-driven footer animates; without this its first
        frame would stay cached forever (the frozen "deciding next input… 0s" bug).
        """
        self._render_cache_width = None
        self._render_cache_lines = None


def _context_fill_style(percent: float | None) -> str:
    if percent is None:
        return "dim"
    if percent >= 90:
        return "bold bright_red"
    if percent >= 70:
        return "bold yellow"
    return "cyan"


def _model_status(config: dict) -> str:
    status_parts = [
        str(config.get("provider", "openai")),
        str(get_setting(config, "model")),
    ]
    reasoning_effort = str(config.get("reasoning_effort") or "").strip()
    if reasoning_effort:
        status_parts.append(reasoning_effort)
    return " / ".join(status_parts)


class HeadsupAgentUI:
    """Adapter used by run_agent to render into the heads-up app."""

    def __init__(self, app: "HeadsupApp"):
        self.app = app
        self._stream_text = ""
        self._stream_index: int | None = None
        self._response_status_index: int | None = None
        self._tool_status_index: int | None = None
        self._tool_live_kind: str | None = None
        self._response_started_at = 0.0
        self._has_reasoning = False
        self._tool_started_at = 0.0
        self._tool_names: tuple[str, ...] = ()
        self._response_waiting = False
        self._tool_waiting = False
        self._animated_live_tool_keys: set[str] = set()
        self._stream_dirty = False
        self._last_stream_refresh_at: float | None = None

    def start_turn(self, query: str, _config: dict) -> None:
        self._stream_text = ""
        self._stream_index = None
        self._response_status_index = None
        self._tool_status_index = None
        self._tool_live_kind = None
        self._has_reasoning = False
        self._tool_names = ()
        self._response_waiting = False
        self._tool_waiting = False
        self._animated_live_tool_keys.clear()
        self._stream_dirty = False
        self._last_stream_refresh_at = None
        self.app.add_user_message(query)

    def start_response_wait(self, start_time: float) -> None:
        self._response_started_at = start_time
        self._has_reasoning = False
        self._response_waiting = True
        self._response_status_index = self.app.upsert_status(
            self._response_status_index,
            self._response_wait_text(),
        )

    def set_response_wait_has_reasoning(self, has_reasoning: bool) -> None:
        self._has_reasoning = has_reasoning
        self._response_status_index = self.app.upsert_status(
            self._response_status_index,
            self._response_wait_text(),
        )

    def complete_response_phase(self, status_text: str) -> None:
        self._flush_stream()
        self._response_waiting = False
        self._response_status_index = self.app.upsert_status(
            self._response_status_index,
            thought_complete_indicator(status_text),
        )
        self._response_status_index = None

    def start_tool_activity(self, start_time: float) -> None:
        self._flush_stream()
        self._tool_started_at = start_time
        self._tool_names = ()
        self._tool_waiting = True
        self._tool_status_index = self.app.upsert_status(
            self._tool_status_index,
            self._tool_wait_text(),
        )

    def update_tool_activity(self, tool_names: tuple[str, ...]) -> None:
        self._tool_names = tool_names
        self._tool_status_index = self.app.upsert_status(
            self._tool_status_index,
            self._tool_wait_text(),
        )

    def complete_tool_phase(self, status_text: str) -> None:
        self._tool_waiting = False
        self._tool_status_index = self.app.upsert_status(
            self._tool_status_index,
            tool_complete_indicator(status_text),
        )
        self._tool_status_index = None

    def begin_assistant_message(self) -> None:
        """Start a fresh streamed message for a new turn within the same run.

        ``start_turn`` runs once per ``run_agent`` call, but a single run can
        stream several assistant messages across tool rounds. Without resetting
        the stream cursor here, a later turn's text would upsert onto the prior
        message's entry index and jump above the tool cards in between. Clearing
        the cursor makes the next flush append a new entry in order.
        """
        self._flush_stream()
        self._stream_text = ""
        self._stream_index = None
        self._stream_dirty = False
        self._last_stream_refresh_at = None

    def append_stream_delta(self, delta: str) -> None:
        self._stream_text += delta
        self._stream_dirty = True
        now = time.perf_counter()
        if (
            self._last_stream_refresh_at is None
            or now - self._last_stream_refresh_at >= STREAM_PREVIEW_REFRESH_INTERVAL
        ):
            self._flush_stream(now)

    def replace_stream_text(self, text: str) -> None:
        self._stream_text = text
        self._stream_dirty = True
        self._flush_stream()

    def finish_assistant_message(self, text: str) -> None:
        if text and text != self._stream_text:
            self.replace_stream_text(text)
        elif self._stream_dirty:
            self._flush_stream()
        elif self._stream_index is not None:
            self.app.invalidate_usage_status()
            self.app.refresh()

    def retry_stream(self) -> None:
        self._stream_text = ""
        self._stream_index = None
        self._response_waiting = False
        self._response_status_index = None
        self._tool_waiting = False
        self._tool_status_index = None
        self.app.add_notice(Text("Retrying response stream...", style="yellow"))

    def show_tool_card(self, renderable: RenderableType) -> None:
        self._flush_stream()
        live_kind = type(renderable).__name__
        if live_kind == "InteractiveCommandCard":
            # One growing slot for the whole interactive run_command session:
            # upsert the same card (animating its footer) while it runs, then
            # finalize on exit so a later command opens a fresh slot rather than
            # overwriting this finished transcript.
            if getattr(renderable, "exited", False):
                self.app.replace_live_tool(live_kind, renderable)
                self._animated_live_tool_keys.discard(live_kind)
            else:
                self.app.upsert_live_tool(live_kind, renderable)
                self._animated_live_tool_keys.add(live_kind)
            return
        if live_kind == "RunningCommandCard":
            self.app.upsert_live_tool(live_kind, renderable)
            self._tool_live_kind = live_kind
            self._animated_live_tool_keys.add(live_kind)
            return
        if live_kind == "SpawnPanel":
            self.app.upsert_live_tool(live_kind, renderable)
            return
        if self._tool_live_kind is not None:
            self.app.replace_live_tool(self._tool_live_kind, renderable)
            self._animated_live_tool_keys.discard(self._tool_live_kind)
            self._tool_live_kind = None
            return
        self.app.add_tool(renderable)

    def show_error(self, message: str) -> None:
        self.app.add_notice(Text(message, style="bold red"))

    def show_notice(self, renderable: RenderableType) -> None:
        self.app.add_notice(renderable)

    def show_usage_line(self, renderable: RenderableType) -> None:
        self.app.add_usage(renderable)

    def ask_user(self, question: str, _config: dict) -> str:
        question_renderable = Markdown(flatten_headings(question))
        self.app.upsert_live_tool(
            "ask_user",
            tool_card(
                "ask_user",
                question_renderable,
                status="waiting",
                status_style="blue",
                display_mode="fullscreen",
            ),
        )
        answer = self.app.read_answer("answer> ", echo_answer=False)
        answer_line = Text("> ", style="bold cyan")
        answer_line.append(answer, style="bright_white")
        self.app.replace_live_tool(
            "ask_user",
            tool_card(
                "ask_user",
                Group(question_renderable, answer_line),
                status="done",
                display_mode="fullscreen",
            ),
        )
        return answer

    def bind_cancel_token(self, token: CancellationToken) -> None:
        self.app.bind_cancel_token(token)

    def unbind_cancel_token(self) -> None:
        self._response_waiting = False
        self._tool_waiting = False
        self._animated_live_tool_keys.clear()
        self.app.unbind_cancel_token()

    def _response_wait_text(self) -> Text:
        elapsed = int(max(0.0, time.perf_counter() - self._response_started_at))
        frame = _THINKING_FRAMES[int(time.perf_counter() * 10) % len(_THINKING_FRAMES)]
        label = response_wait_label(self._has_reasoning)
        return Text(f"{frame}  {label}\u2026  {elapsed}s")

    def _tool_wait_text(self) -> Text:
        elapsed = int(max(0.0, time.perf_counter() - self._tool_started_at))
        frame = _THINKING_FRAMES[int(time.perf_counter() * 10) % len(_THINKING_FRAMES)]
        label = (
            tool_activity_label(self._tool_names)
            if self._tool_names
            else "Preparing action"
        )
        return Text(f"{frame}  {label}\u2026  {elapsed}s")

    def has_active_animation(self) -> bool:
        """Whether a spinner or live tool card still needs periodic repaints.

        Polled by the heads-up loop's ``on_tick`` (which replaced this UI's old
        background ticker thread) to decide when to keep animating.
        """
        return bool(
            self._response_waiting
            or self._tool_waiting
            or self._animated_live_tool_keys
        )

    def _refresh_wait_statuses(self) -> bool:
        active = False
        if self._response_waiting:
            active = True
            self._response_status_index = self.app.upsert_status(
                self._response_status_index,
                self._response_wait_text(),
            )
        if self._tool_waiting:
            active = True
            self._tool_status_index = self.app.upsert_status(
                self._tool_status_index,
                self._tool_wait_text(),
            )
        if self._animated_live_tool_keys:
            active = True
            # Snapshot: the worker thread may add/discard keys as cards open and
            # exit while the loop thread iterates here.
            for key in tuple(self._animated_live_tool_keys):
                self.app.invalidate_live_tool(key)
            self.app.refresh()
        return active

    def _flush_stream(self, now: float | None = None) -> None:
        if not self._stream_dirty:
            return
        self._stream_index = self.app.upsert_assistant_message(
            self._stream_index,
            self._stream_text or " ",
        )
        self._stream_dirty = False
        self._last_stream_refresh_at = time.perf_counter() if now is None else now


def _update_outcome_lines(outcome) -> tuple[Text, list[Text]]:
    """Heads-up rendering of an UpdateOutcome: final status line + notices.

    The restart hints differ from the CLI's on purpose: here the user is
    already inside jarv, so "run jarv again" reads wrong — the hint says to
    restart/exit instead.
    """
    if outcome.kind in ("updated", "staged"):
        version = f"v{outcome.latest}" if outcome.latest else "the new version"
        line = Text("✓ ", style="bold green")
        if outcome.kind == "updated":
            line.append(f"Updated to {version}", style="green")
            line.append(" — restart jarv to start using it.", style="dim")
        else:
            line.append(f"Update to {version} staged", style="green")
            line.append(" — exit jarv and run it again to finish.", style="dim")
        return line, []
    if outcome.kind == "current":
        detail = f" ({outcome.detail})" if outcome.detail else ""
        return Text(f"✓ {outcome.message}{detail}", style="dim"), []
    details = [
        Text(line, style="dim")
        for line in (outcome.detail or "").splitlines()
        if line.strip()
    ]
    if outcome.kind == "editable":
        return Text(f"⚠ {outcome.message}", style="yellow"), details
    line = Text("✗ ", style="bold red")
    line.append(outcome.message, style="red")
    return line, details


class _UpdateTask:
    """One background /update run, shown as a live status line in the transcript.

    The update flow blocks on the network and the installer subprocess for up
    to minutes, so it runs on a daemon worker thread while the loop keeps the
    UI responsive; on_tick animates the spinner between stage changes, the same
    way agent-turn wait statuses animate. A single status entry is upserted in
    place from stage to stage and finalized with the outcome.
    """

    def __init__(self, app: "HeadsupApp"):
        self.app = app
        self.done = False
        self._stage = "Checking for updates"
        self._stage_started_at = time.perf_counter()
        self._status_index: int | None = None

    def start(self) -> None:
        self._paint()
        threading.Thread(target=self._worker, name="headsup-update", daemon=True).start()

    def refresh_animation(self) -> None:
        """Advance the spinner; called from the loop's on_tick."""
        self._paint()

    def _worker(self) -> None:
        from .commands import UpdateOutcome, perform_update

        try:
            outcome = perform_update(self._set_stage)
        except Exception as exc:  # a failed update must never take the UI down
            outcome = UpdateOutcome("failed", "Update failed.", detail=str(exc))
        self._finish(outcome)

    def _set_stage(self, text: str) -> None:
        with self.app.lock:
            self._stage = text
            self._stage_started_at = time.perf_counter()
        self._paint()

    def _paint(self) -> None:
        with self.app.lock:
            if self.done:
                return
            elapsed = int(max(0.0, time.perf_counter() - self._stage_started_at))
            frame = _THINKING_FRAMES[int(time.perf_counter() * 10) % len(_THINKING_FRAMES)]
            self._status_index = self.app.upsert_status(
                self._status_index,
                Text(f"{frame}  {self._stage}…  {elapsed}s"),
            )

    def _finish(self, outcome) -> None:
        summary, details = _update_outcome_lines(outcome)
        with self.app.lock:
            self.done = True
            self._status_index = self.app.upsert_status(self._status_index, summary)
        for line in details:
            self.app.add_notice(line)
        self.app._update_finished(self)


class HeadsupApp(AltScreenApp):
    # Heads-up keeps its own SGR-mouse handling, so it disables Rich/terminal
    # mouse capture rather than letting the base app capture the mouse.
    clear_on_resize = False
    translate_mouse_wheel = False
    # Windows: read pastes as one bracketed-paste block rather than raw chars,
    # so a multi-line paste can't fragment / submit its first line mid-paste.
    use_vt_input = True
    first_paint_label = "headsup"

    def __init__(
        self,
        config: dict,
        client,
        *,
        args: argparse.Namespace | None,
        agent_loader: tuple[dict, threading.Event],
        handle_slash: SlashHandler,
        maybe_command: MaybeCommand,
        render_console: Console = console,
    ):
        super().__init__(
            console=render_console,
            repeatable_keys=_HEADSUP_REPEATABLE_KEYS,
            live_factory=self._build_live,
            read_key_fn=self._read_headsup_key,
            key_available_fn=self._headsup_key_available,
        )
        self.config = dict(config)
        self.client = client
        self.args = args
        self.agent_import, self.agent_ready = agent_loader
        self.handle_slash = handle_slash
        self.maybe_command = maybe_command
        self.entries: list[TranscriptEntry] = [self._initial_notice_entry()]
        self.editor: dict = {}
        initialize_text_editor(self.editor, "")
        self._pastes = PasteRegistry()
        self.scroll_offset = 0
        self.lock = threading.RLock()
        self._exit_armed = False
        self._cancel_token: CancellationToken | None = None
        self._live_tool_index: dict[str, int] = {}
        self._refresh_suspended = 0
        self._foreground_input_active = False
        self._foreground_input_thread: threading.Thread | None = None
        self._idle_anim_started_at = 0.0
        self._idle_anim_stop = threading.Event()
        self._outro_started_at = 0.0
        self._active_ui: HeadsupAgentUI | None = None
        self._last_wait_tick = 0.0
        self._last_anim_frame = 0.0
        self._agent_busy = False
        self._agent_thread: threading.Thread | None = None
        self._update_task: _UpdateTask | None = None
        self._queued_queries: deque[tuple[str, Callable | None]] = deque()
        self._answer_request: dict | None = None
        self._answer_request_completed: dict = {}
        self._answer_condition = threading.Condition(self.lock)
        self.incognito = bool(getattr(args, "incognito", False))
        self._started_new_session = bool(getattr(args, "new", False)) and not self.incognito
        self._initial_history_synced = False
        if self._started_new_session:
            forget_current_session()
        self.session_context = prepare_session_context(persist_metadata=False)
        self.usage_path = usage_file_for(self.session_context.history_file)
        self._prompt_history = (
            []
            if self.incognito or self._started_new_session
            else self._load_prompt_history()
        )
        self._prompt_history_index: int | None = None
        self._prompt_history_draft = ""
        self._slash_menu_index = 0
        self._slash_menu_scroll = 0
        # Esc hides the popup for the exact draft it was dismissed on; any edit
        # changes the buffer and so re-enables it (no reset bookkeeping needed).
        self._slash_menu_dismissed_for: str | None = None
        self._prompt_notice: RenderableType | None = None
        self._usage_status_cache: tuple[float, int, object, str | None, Text] | None = None

    # ------------------------------------------------------------------ #
    # AltScreenApp wiring: resolve patchable module symbols at call time so
    # tests that patch ``jarv.headsup.*`` keep driving the loop.
    # ------------------------------------------------------------------ #
    def _build_live(self, get_renderable, _console):
        self._begin_idle_animation()
        return Live(
            get_renderable=get_renderable,
            console=self.console,
            screen=True,
            auto_refresh=False,
            transient=False,
            vertical_overflow="crop",
        )

    def _read_headsup_key(self) -> tuple[str, int]:
        return _read_key_with_repeats(
            text_mode=True,
            batch_text=True,
            repeatable=_HEADSUP_REPEATABLE_KEYS,
            translate_mouse_wheel=False,
        )

    def _headsup_key_available(self) -> bool:
        return _key_available()

    # ------------------------------------------------------------------ #
    # Lifecycle (single-threaded loop owned by AltScreenApp)
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self._sync_initial_transcript_from_history()
        super().run()

    def on_start(self) -> None:
        # Capture the wheel as SGR mouse events so MOUSE_WHEEL_* tokens reach
        # on_key and scroll the transcript -- rather than letting the terminal's
        # alternate-scroll mode translate the wheel into prompt-history arrows.
        # Runs after the alt screen / VT output is live (on_start fires inside the
        # screen context), so the enable sequence reaches the terminal.
        enable_mouse_wheel_reporting()
        self._foreground_input_active = True
        self._foreground_input_thread = threading.current_thread()
        self._begin_idle_animation()

    def on_stop(self) -> None:
        self._idle_anim_stop.set()
        if self._answer_request is not None:
            self._cancel_answer()
        self._foreground_input_active = False
        self._foreground_input_thread = None
        self._cancel_active_turn(clear_queue=True)
        self._wait_for_agent_idle(timeout=5.0)
        disable_mouse_capture()

    def on_interrupt(self) -> None:
        # Ctrl+C copies an active selection (and clears it) rather than exiting,
        # so the escape hatch survives: with nothing selected it still dismisses.
        if self._copy_selection_to_clipboard():
            return
        if self._handle_prompt_dismiss():
            self.stop()

    @contextmanager
    def suspended(self):
        # Nested full-screen views (/help, /settings, /tree, the session browser)
        # set their own mouse modes and reset them on exit, which leaves SGR mouse
        # reporting off. Re-assert it on resume so the wheel keeps scrolling the
        # transcript instead of reverting to alternate-scroll arrows.
        with super().suspended():
            yield
        enable_mouse_wheel_reporting()

    def on_key(self, key: str, repeat: int) -> None:
        if key == "ENTER":
            if self._answer_request is not None:
                self._complete_answer()
                return
            menu_matches = self._slash_menu_matches()
            if menu_matches and not self._menu_enter_submits_draft():
                # While a *partial* token is being typed, Enter accepts the
                # highlighted entry: a draft like "/sett" runs /settings instead
                # of an unknown command, and "/usage d" runs "/usage day". A
                # fully typed command or argument ("/usage", "/usage day") falls
                # through to run exactly as typed — see _menu_enter_submits_draft.
                self._slash_menu_accept(menu_matches[self._slash_menu_index], run=True)
                return
            raw_query = self._pastes.expand(str(self.editor.get("buffer", "")))
            query = raw_query.strip("\r\n")
            if not query.strip():
                initialize_text_editor(self.editor, "")
                self._pastes.clear()
                self._clear_prompt_notice()
                self._exit_armed = False
                return
            self._record_prompt_history(query)
            initialize_text_editor(self.editor, "")
            self._pastes.clear()
            self._clear_prompt_notice()
            self._exit_armed = False
            if self._handle_query(query) == "exit":
                self.stop()
            return
        if key == "ESC":
            if self._answer_request is not None:
                self._cancel_answer()
                return
            if self._slash_menu_matches():
                # Layered dismissal: close the popup first and keep the draft.
                # The next edit changes the buffer and reopens it.
                self._slash_menu_dismissed_for = str(self.editor.get("buffer", ""))
                self.refresh()
                return
            if self._handle_prompt_dismiss():
                self.stop()
            return
        if key in ("CTRL_V", "ALT_V"):
            self._paste_from_system_clipboard()
            return
        scroll_delta = scroll_key_delta(key, repeat)
        if scroll_delta is not None:
            self._scroll_transcript(scroll_delta)
            return
        if key in {"UP", "DOWN", "TAB"}:
            # The autocomplete menu owns these keys whenever it is open, taking
            # priority over prompt-history navigation (a "/se" draft is a single
            # line, so history nav would otherwise capture the arrows).
            menu_matches = self._slash_menu_matches()
            if menu_matches:
                if key == "TAB":
                    # Tab only completes text into the box — it never runs the
                    # command; Enter is the sole key that executes.
                    self._slash_menu_accept(menu_matches[self._slash_menu_index], run=False)
                elif key == "UP":
                    self._slash_menu_index = max(0, self._slash_menu_index - repeat)
                else:
                    self._slash_menu_index = min(
                        len(menu_matches) - 1, self._slash_menu_index + repeat
                    )
                self.refresh()
                return
        if (
            key in {"UP", "DOWN"}
            and self._answer_request is None
            and not self._prompt_has_multiline_draft()
        ):
            if self._navigate_prompt_history(key, repeat):
                self.scroll_offset = 0
                self._clear_prompt_notice()
                self._exit_armed = False
            return
        changed, user_text = self._apply_editor_key(key, repeat)
        if changed or user_text:
            if self._answer_request is None:
                self._reset_prompt_history_navigation()
            self.scroll_offset = 0
            self._clear_prompt_notice()
            self._exit_armed = False
            # Re-typing always re-highlights the top match and reopens a
            # dismissed popup.
            self._slash_menu_index = 0
            self._slash_menu_dismissed_for = None

    def on_tick(self) -> None:
        now = time.perf_counter()
        repaint = False
        if self._idle_animation_active() or (
            self._outro_started_at and now - self._outro_started_at < _OUTRO_DURATION
        ):
            # The intro/outro frame is derived from the clock in render(), so a
            # plain repaint advances the animation. Gate it to the animation
            # cadence: the loop wakes far more often than that for input, and
            # repainting the starfield every wake would multiply its CPU cost.
            if now - self._last_anim_frame >= self.frame_interval:
                self._last_anim_frame = now
                repaint = True
        elif self._outro_started_at:
            self._outro_started_at = 0.0
            repaint = True
        ui = self._active_ui
        update_task = self._update_task
        animating = (ui is not None and ui.has_active_animation()) or update_task is not None
        if animating and now - self._last_wait_tick >= 0.2:
            self._last_wait_tick = now
            # Recomputes the spinner text in place (and invalidates).
            if ui is not None and ui.has_active_animation():
                ui._refresh_wait_statuses()
            if update_task is not None:
                update_task.refresh_animation()
        if repaint:
            self.invalidate()

    def render(self) -> RenderableType:
        term_w, term_h = terminal_size(console=self.console)
        layout = compute_layout(term_w, term_h)
        inner_width = layout.inner_width
        model_status = _model_status(self.config)
        title = compose_title(model_status, layout.panel_width)

        with self.lock:
            # The slash-command popup and the input box are drawn as one seamless
            # unit: a compact, left-aligned popup (top edge + suggestion rows) floats
            # over the bottom of the transcript, the footer row becomes the edge that
            # docks it onto the field, and the input box drops its own top so that
            # docking edge serves as it.
            menu_lines = self._slash_menu_box(inner_width, layout)
            menu_open = bool(menu_lines)
            prompt_lines = self._prompt_lines(
                inner_width, max_lines=layout.max_prompt_rows, menu_open=menu_open
            )
            if menu_open:
                footer = box_tab_top(inner_width, cell_len(menu_lines[0].plain))
            else:
                footer = self._footer_line(inner_width)
            # Opening the popup makes the input box one row shorter (the divider
            # replaces its top border). That freed row is *not* given to the
            # transcript -- it becomes the popup's hint row, sitting exactly where
            # the footer hints sit while the popup is closed -- so the transcript
            # keeps the same height and nothing already on screen moves.
            rows = _transcript_rows_for(
                layout.body_height, len(prompt_lines) + (1 if menu_open else 0)
            )
            transcript = self._transcript_lines(inner_width)
            visible, self.scroll_offset = window_transcript(transcript, rows, self.scroll_offset)
            show_intro = (
                not self._idle_anim_stop.is_set()
                and self.scroll_offset == 0
                and self._answer_request is None
                and all(entry.kind == "notice" for entry in self.entries)
            )
            outro_started_at = self._outro_started_at

        # The transcript height is the same whether the popup is open or closed,
        # so the intro's centred logo and seeded starfield never re-roll or jump
        # as the popup opens.
        intro = None
        if show_intro:
            intro = render_intro(
                inner_width,
                rows,
                time.perf_counter() - self._idle_anim_started_at,
            )
        elif outro_started_at:
            exit_progress = (time.perf_counter() - outro_started_at) / _OUTRO_DURATION
            if exit_progress < 1.0:
                intro = render_intro(
                    inner_width,
                    rows,
                    time.perf_counter() - self._idle_anim_started_at,
                    exit=exit_progress,
                )
        if intro is not None:
            visible = intro + [Text("")] * max(0, rows - len(intro))

        body_rows = rows
        if menu_open:
            # The row freed by the input box's dropped top edge carries the
            # popup's key hints; the popup's bottom row is overlaid onto its left
            # cells, so the hints read beside the popup -- in the very cells the
            # footer hints occupy while the popup is closed.
            popup_width = cell_len(menu_lines[0].plain)
            visible = [Text("")] * max(0, rows - len(visible)) + list(visible)
            visible.append(self._slash_menu_hint_row(inner_width, popup_width))
            body_rows = rows + 1
            # Paint the slash popup over the bottom of the body so it replaces the
            # stars beneath it and docks seamlessly onto the input box below.
            visible = overlay_menu(
                visible, menu_lines, rows=body_rows, width=inner_width, console=self.console
            )
        parts = assemble_body(visible, footer, prompt_lines, layout.body_height, body_rows)
        subtitle = self._panel_subtitle(layout.panel_width)
        return build_frame(
            parts,
            title=title,
            subtitle=subtitle,
            panel_width=layout.panel_width,
            term_h=layout.term_h,
        )

    def _panel_subtitle(self, panel_width: int) -> Text:
        # Working directory on the bottom-left, usage status on the bottom-right.
        # compose_subtitle keeps the usage intact and truncates the dir first.
        usage = self._usage_status(max(1, panel_width))
        return compose_subtitle(self._cwd_label(), usage, panel_width)

    def _cwd_label(self) -> str:
        """The working directory for the bottom bar.

        $HOME is collapsed to ~ only for *sub*directories (``~\\Desktop``); the
        home directory itself stays absolute, since a lone ``~`` reads oddly.
        """
        try:
            cwd = os.getcwd()
        except OSError:
            return ""
        home = os.path.expanduser("~")
        if home and home != "~" and cwd.startswith(home + os.sep):
            cwd = "~" + cwd[len(home):]
        return cwd

    def refresh(self) -> None:
        # The loop thread is the sole painter; producers (the agent worker,
        # animations, slash output) only request a repaint. ``_refresh_suspended``
        # still drops requests while console output is being captured.
        if self._refresh_suspended:
            return
        self.invalidate()

    def invalidate_usage_status(self) -> None:
        self._usage_status_cache = None

    def _idle_animation_active(self) -> bool:
        if self._idle_anim_stop.is_set():
            return False
        if self.scroll_offset:
            return False
        with self.lock:
            if self._answer_request is not None:
                return False
            return all(entry.kind == "notice" for entry in self.entries)

    def _begin_idle_animation(self) -> None:
        # The intro is now driven by the loop's on_tick rather than a dedicated
        # thread; this only arms the animation state when it should be visible.
        if not self._idle_animation_active():
            return
        self._idle_anim_started_at = time.perf_counter()
        self._idle_anim_stop.clear()

    def _restart_idle_animation(self) -> None:
        self._idle_anim_stop.set()
        self._outro_started_at = 0.0
        self._idle_anim_started_at = time.perf_counter()
        self._idle_anim_stop.clear()
        self.invalidate()

    def _dismiss_intro(self) -> None:
        """Tear down the idle intro, playing a quick outro if it's on screen.

        Called the first time real transcript content lands. If the intro is
        currently visible we start a short, non-blocking dissolve (advanced by
        on_tick) before the transcript takes over; otherwise we just mark it
        dismissed.

        Runs on the agent worker thread, so the stop flag and outro timestamp are
        mutated under ``self.lock`` (a re-entrant RLock) to stay consistent with
        the loop thread reading them together in render()/on_tick -- otherwise the
        loop could see the stop set but the outro timestamp not yet written and
        cut the intro abruptly for a frame.
        """
        with self.lock:
            if self._idle_anim_stop.is_set():
                return
            was_visible = self._idle_animation_active()
            self._idle_anim_stop.set()
            if was_visible and self._foreground_input_active:
                self._outro_started_at = time.perf_counter()

    def add_user_message(self, query: str) -> None:
        line = Text()
        line.append("\u203a ", style="dim cyan")
        line.append(query, style="bright_white")
        with self.lock:
            # A new turn always jumps back to the bottom so the user sees their
            # message and the streamed response, even if they had scrolled up.
            self.scroll_offset = 0
        self._append(
            "user",
            line,
            spacer_before=len(self.entries) > 0,
        )

    def upsert_status(self, index: int | None, renderable: RenderableType) -> int:
        return self._upsert(index, "status", renderable)

    def upsert_assistant_message(self, index: int | None, text: str) -> int:
        return self._upsert(
            index,
            "assistant",
            Markdown(flatten_headings(text or " ")),
        )

    def add_usage(self, renderable: RenderableType) -> None:
        self._append("usage", renderable)

    def add_tool(self, renderable: RenderableType) -> None:
        self._append("tool", renderable)

    def upsert_live_tool(self, key: str, renderable: RenderableType) -> None:
        index = self._live_tool_index.get(key)
        self._live_tool_index[key] = self._upsert(index, "tool", renderable)

    def replace_live_tool(self, key: str, renderable: RenderableType) -> None:
        index = self._live_tool_index.pop(key, None)
        self._upsert(index, "tool", renderable)

    def invalidate_live_tool(self, key: str) -> None:
        """Bust an animated live tool's render cache so its footer ticks.

        Status spinners animate by upserting a fresh entry each tick; the live
        tool cards (running- and interactive-command) are too large to rebuild,
        so they keep one entry and we just invalidate its cache before a repaint.
        """
        with self.lock:
            index = self._live_tool_index.get(key)
            if index is not None and 0 <= index < len(self.entries):
                self.entries[index].invalidate()

    def add_notice(self, renderable: RenderableType) -> None:
        self._append("notice", renderable)

    def set_prompt_notice(self, renderable: RenderableType | None) -> None:
        with self.lock:
            self._prompt_notice = renderable
        self.refresh()

    def _clear_prompt_notice(self) -> None:
        with self.lock:
            self._prompt_notice = None
        self.refresh()

    def read_answer(self, label: str, *, echo_answer: bool = True) -> str:
        # Normal heads-up turns run on a worker thread while the loop owns the
        # screen, so the answer is collected by the loop (resize and repaint keep
        # working). _read_answer_direct is the defensive fallback for the
        # degenerate case where read_answer is invoked on the loop thread itself
        # (routing that to the foreground path would deadlock the loop).
        if self._foreground_input_active:
            if self._foreground_input_thread is threading.current_thread():
                return self._read_answer_direct(label, echo_answer=echo_answer)
            return self._read_answer_from_foreground(label, echo_answer=echo_answer)
        return self._read_answer_direct(label, echo_answer=echo_answer)

    def _read_answer_direct(self, label: str, *, echo_answer: bool = True) -> str:
        previous = dict(self.editor)
        initialize_text_editor(self.editor, "")
        with self.lock:
            self._answer_request = {
                "label": label,
                "answer": None,
                "cancelled": False,
                "previous": previous,
                "echo_answer": echo_answer,
            }
        try:
            while True:
                # This nested modal read blocks the main loop, so paint in place.
                self.paint_now()
                try:
                    key, repeat = _read_key_with_repeats(
                        text_mode=True,
                        batch_text=True,
                    )
                except KeyboardInterrupt:
                    raise
                if key == "ENTER":
                    answer = str(self.editor.get("buffer", "")).strip()
                    if echo_answer:
                        self.add_notice(Text(f"{label}{answer}", style="dim"))
                    return answer
                if key == "ESC":
                    if self._cancel_token is not None:
                        self._cancel_token.cancel()
                        raise TurnCancelled
                    return "[no response]"
                self._apply_editor_key(key, repeat)
        finally:
            self.editor = previous
            with self._answer_condition:
                self._answer_request = None
                self._answer_request_completed = {}
                self._answer_condition.notify_all()

    def _current_prompt_edit_width(self) -> int:
        term_w, _term_h = terminal_size(console=self.console)
        inner_width = max(1, _panel_width(max(20, term_w)) - 4)
        return self._prompt_edit_width(inner_width)

    def _editor_selection_span(self) -> tuple[int, int] | None:
        """The active Shift-selection span over the draft buffer, if any."""
        return selection_bounds(self.editor)

    def _copy_selection_to_clipboard(self) -> bool:
        """Copy an active selection to the clipboard; True when it handled Ctrl+C.

        The selection is cleared afterwards so a follow-up Ctrl+C falls through
        to the normal interrupt/dismiss path -- quitting still works, it just
        takes a second press while text is selected. Paste-chip markers in the
        span are expanded so the real pasted text lands on the clipboard.
        """
        span = self._editor_selection_span()
        if span is None:
            return False
        start, end = span
        buffer = str(self.editor.get("buffer", ""))
        selected = self._pastes.expand(buffer[start:end])
        self.editor["selection_anchor"] = None
        if copy_to_clipboard(selected):
            count = len(selected)
            unit = "character" if count == 1 else "characters"
            self.set_prompt_notice(
                Text(f"Copied {count} {unit} to clipboard.", style="dim")
            )
        else:
            self.set_prompt_notice(
                Text("Couldn't reach the clipboard.", style="yellow")
            )
        return True

    def _paste_from_system_clipboard(self) -> None:
        """Handle Ctrl+V / Alt+V: attach a copied image, else paste text.

        Terminals only deliver clipboard *text* through the input stream, so a
        copied image never arrives as a paste -- the keystroke itself is the
        cue to read the OS clipboard directly. An image is materialised as a
        file and inserted as an ``[Image #N]`` chip whose submitted expansion
        is a reference to that file; the read tool turns local image files into
        image blocks for vision-capable models. With no image on the clipboard
        the key pastes the clipboard's text instead, which covers terminals
        that pass Ctrl+V through without handling it. (Reading the clipboard
        can shell out -- ~a second on Windows -- and briefly blocks the loop,
        like the nested modal reads do.)
        """
        if self._answer_request is not None:
            return
        image = read_clipboard_image()
        if image is not None:
            marker = self._pastes.attach("Image", f"[attached image: {image.path}]")
            buffer, cursor = self._editor_buffer_cursor()
            self.editor["buffer"] = buffer[:cursor] + marker + buffer[cursor:]
            self.editor["cursor"] = cursor + len(marker)
            self.editor["preferred_visual_column"] = None
            self.editor["selection_anchor"] = None
            self._after_clipboard_paste()
            capability = get_image_output_capability(self.config)
            if capability.supported:
                self.set_prompt_notice(
                    Text(f"Image attached: {image.path.name}", style="dim")
                )
            else:
                self.set_prompt_notice(
                    Text(
                        "Image attached, but the active model can't view images "
                        f"({capability.reason}).",
                        style="yellow",
                    )
                )
            return
        text = read_clipboard_text()
        if text:
            changed, _ = self._apply_editor_key(TextInput(text), 1)
            if changed:
                self._after_clipboard_paste()
            return
        self.set_prompt_notice(Text("Nothing to paste on the clipboard.", style="yellow"))

    def _after_clipboard_paste(self) -> None:
        """The same draft-changed bookkeeping on_key does after an editor key."""
        self._reset_prompt_history_navigation()
        self.scroll_offset = 0
        self._clear_prompt_notice()
        self._exit_armed = False
        self._slash_menu_index = 0
        self._slash_menu_dismissed_for = None

    def _apply_editor_key(self, key: str, repeat: int) -> tuple[bool, bool]:
        if isinstance(key, TextInput):
            value = str(self.editor.get("buffer", ""))
            cursor = max(0, min(int(self.editor.get("cursor", len(value))), len(value)))
            start = max(0, cursor - _SGR_MOUSE_TEXT_LOOKBACK)
            existing_tail = value[start:cursor]
            combined = existing_tail + str(key)
            stripped = strip_sgr_mouse_sequences(combined)
            if stripped != combined:
                self.editor["buffer"] = value[:start] + stripped + value[cursor:]
                self.editor["cursor"] = start + len(stripped)
                self.editor["preferred_visual_column"] = None
                self.editor["selection_anchor"] = None
                return stripped != existing_tail, bool(stripped)

        key = _sanitize_editor_key(key)
        if key == "CTRL_N":
            key = "ENTER"
        has_selection = self._editor_selection_span() is not None
        # An active selection takes priority over the paste-chip shortcuts: a
        # Backspace/Delete deletes the selection (handled by the editor), and a
        # re-paste replaces it rather than unboxing an adjacent chip.
        if (
            key in ("BACKSPACE", "DELETE")
            and not has_selection
            and self._delete_adjacent_paste(key)
        ):
            return True, False
        if isinstance(key, TextInput) and self._answer_request is None:
            content = str(key)
            if not has_selection and self._unbox_duplicate_paste(content):
                return True, True
            marker = self._pastes.collapse(content)
            if marker is not None:
                key = TextInput(marker)
        changed = apply_text_editor_key(
            self.editor,
            key,
            repeat,
            content_width=self._current_prompt_edit_width(),
            allow_newlines=True,
        )
        if changed and has_selection:
            # A selection replace/delete may have cut through a paste chip; drop
            # any markers no longer present in the buffer.
            self._pastes.prune(str(self.editor.get("buffer", "")))
        return changed, isinstance(key, TextInput)

    def _editor_buffer_cursor(self) -> tuple[str, int]:
        buffer = str(self.editor.get("buffer", ""))
        cursor = max(0, min(int(self.editor.get("cursor", len(buffer))), len(buffer)))
        return buffer, cursor

    def _delete_adjacent_paste(self, key: str) -> bool:
        """Erase a whole ``[Pasted text]`` chip in one Backspace/Delete."""
        buffer, cursor = self._editor_buffer_cursor()
        index = cursor - 1 if key == "BACKSPACE" else cursor
        if not 0 <= index < len(buffer):
            return False
        span = self._pastes.span_covering(buffer, index)
        if span is None:
            return False
        start, end = span
        self.editor["buffer"] = buffer[:start] + buffer[end:]
        self.editor["cursor"] = start
        self.editor["preferred_visual_column"] = None
        self._pastes.prune(self.editor["buffer"])
        return True

    def _unbox_duplicate_paste(self, content: str) -> bool:
        """Expand an adjacent chip to one plain copy when its block is re-pasted."""
        buffer, cursor = self._editor_buffer_cursor()
        span = self._pastes.duplicate_span(buffer, cursor, content)
        if span is None:
            return False
        start, end = span
        text = content.replace("\r\n", "\n").replace("\r", "\n")
        self.editor["buffer"] = buffer[:start] + text + buffer[end:]
        self.editor["cursor"] = start + len(text)
        self.editor["preferred_visual_column"] = None
        self._pastes.prune(self.editor["buffer"])
        return True

    def bind_cancel_token(self, token: CancellationToken) -> None:
        # The single-threaded loop owns stdin and turns ESC into a cancel via
        # on_key, so we just record the token. (A background esc-listener thread
        # was removed: it only ever ran for the non-foreground adapter path that
        # no longer exists, and it raced the polled reader for the same stdin.)
        self._cancel_token = token

    def unbind_cancel_token(self) -> None:
        self._cancel_token = None

    def _handle_prompt_dismiss(self) -> bool:
        if str(self.editor.get("buffer", "")):
            initialize_text_editor(self.editor, "")
            self._pastes.clear()
            self._reset_prompt_history_navigation()
            self.set_prompt_notice(Text("Draft cleared.", style="dim"))
            return False
        if self._cancel_active_turn():
            self.set_prompt_notice(Text("Cancelling current turn.", style="yellow"))
            return False
        if self._exit_armed:
            return True
        self._exit_armed = True
        self.set_prompt_notice(Text("Press Esc or Ctrl+C again to exit.", style="yellow"))
        return False

    def _handle_query(self, query: str) -> str | None:
        if not query:
            return None
        if query.lower() in {"exit", "quit"}:
            return "exit"
        if "\n" in query:
            self._run_agent_query(query)
            return None

        parts = query.split()
        if len(parts) > 1 and parts[0].lower() == "jarv" and parts[1].startswith("/"):
            parts = parts[1:]

        if parts[0].startswith("/"):
            command = parts[0].lower()
            if command in {"/exit", "/quit"}:
                return "exit"
            self._run_slash(command, parts[1:])
            return None

        alias = self._command_alias(parts[0], parts[1:])
        if alias is not None:
            command, rest = alias
            if self._confirm_command_alias(command, rest, query):
                self._run_slash(command, rest)
                return None
            self._run_agent_query(query)
            return None

        result = self.maybe_command(parts[0], parts[1:])
        if result is not None:
            _, command, rest = result
            self._run_slash(command, rest)
            return None

        self._run_agent_query(query)
        return None

    def _command_alias(self, first_word: str, rest: list[str]) -> tuple[str, list[str]] | None:
        return parse_command_alias(first_word, rest)

    def _confirm_command_alias(self, command: str, rest: list[str], full_input: str) -> bool:
        command_text = " ".join([command] + rest)
        message = Text()
        message.append("Did you mean ", style="yellow")
        message.append(command_text, style="bold cyan")
        message.append(" or a message?", style="yellow")
        self.add_notice(message)
        self.add_notice(
            Text(
                f"1 run command   2 send message: {full_input}",
                style="dim",
                no_wrap=True,
                overflow="ellipsis",
            )
        )
        answer = self.read_answer("choice> ").strip().lower()
        return answer in _COMMAND_CONFIRM_YES

    def _run_slash(self, command: str, rest: list[str]) -> None:
        if command == "/tree":
            self._run_tree()
            return
        if command == "/btw":
            self._run_btw(rest)
            return
        if command == "/update":
            # Blocks on the network and the installer for up to minutes, so it
            # runs like an agent turn — worker thread plus a live status entry —
            # instead of freezing the loop in the captured-output path below.
            self._start_update()
            return
        if command in _FULLSCREEN_SLASH_COMMANDS or (
            command in {"/session", "/sessions"} and not rest
        ):
            self._run_interactive_slash(command, rest)
            return
        with self._captured_console_output() as capture:
            self.config, self.client = self.handle_slash(
                command,
                rest,
                self.config,
                self.client,
                self.args,
                True,
            )
        output = capture.get().strip()
        notice = Text.from_ansi(output) if output else None
        if not self._sync_after_slash(command, notice):
            if notice:
                self.add_notice(notice)

    def _run_interactive_slash(self, command: str, rest: list[str]) -> None:
        with self.suspended():
            self.config, self.client = self.handle_slash(
                command,
                rest,
                self.config,
                self.client,
                self.args,
                True,
            )
        self._sync_after_slash(command, None)

    def _run_tree(self) -> None:
        """Open the prompt-tree view and apply the chosen fork/edit/resume.

        Branch operations are pure disk writes (the next turn reloads history), so
        after the view returns we re-sync the on-screen transcript from disk and,
        for an edit, pre-fill the editor with the selected prompt to revise.
        """
        if self.incognito:
            self.add_notice(
                Text("/tree is unavailable in incognito — history isn't saved.", style="yellow")
            )
            return

        from . import session_tree
        from .tree_browser import run_tree_screen

        with self.suspended():
            outcome = run_tree_screen(self.session_context, self.config)

        if outcome.action not in ("open", "fork", "edit"):
            return
        session_tree.checkout(self.session_context.history_file, leaf_id=outcome.leaf_id)
        self._sync_transcript_from_history()
        if outcome.action == "edit" and outcome.prefill is not None:
            with self.lock:
                initialize_text_editor(self.editor, outcome.prefill)

    def _run_btw(self, rest: list[str]) -> None:
        """Ask an aside that doesn't derail the main thread.

        The question runs as a normal turn (so its answer streams live), then
        :meth:`_after_btw` moves that one exchange off the active path into the
        branch sidecar. The aside stays visible in the transcript and in /tree, but
        the next main message continues from before it -- so it never enters the
        main thread's future context.
        """
        question = " ".join(rest).strip()
        if not question:
            self.add_notice(
                Text("Usage: /btw <question> — ask an aside without derailing the thread.", style="dim")
            )
            return
        if self.incognito:
            self.add_notice(Text("Incognito: /btw runs as a normal turn (no aside is saved).", style="dim"))
            self._run_agent_query(question)
            return
        self._run_agent_query(question, on_complete=self._after_btw)

    def _after_btw(self, result) -> None:
        if result is None or getattr(result, "cancelled", False) or isinstance(
            getattr(result, "error", None), str
        ):
            return  # leave an incomplete aside in place; the user can /tree it
        from . import session_tree
        from .history import branches_file_for, load_branches, load_history
        from .session_tree import build_tree

        history_file = self.session_context.history_file
        model = build_tree(load_history(history_file), load_branches(branches_file_for(history_file)))
        active = model.active_path
        if len(active) < 2:
            return  # the aside is the only exchange — nothing to return to
        if session_tree.checkout(history_file, leaf_id=active[-2].frame_id):
            self.add_notice(Text("↩ Set aside — kept in /tree, out of the main thread.", style="dim cyan"))

    def _run_agent_query(self, query: str, on_complete: Callable | None = None) -> None:
        if self.live is not None:
            self._queue_or_start_agent_query(query, on_complete)
            return
        result = self._run_agent_query_now(query)
        if on_complete is not None:
            on_complete(result)

    def _run_agent_query_now(self, query: str):
        try:
            self.agent_ready.wait()
            if "error" in self.agent_import:
                raise self.agent_import["error"]
            ui = HeadsupAgentUI(self)
            # The loop's on_tick polls the active UI to animate spinners and live
            # tool cards (this replaced the UI's old background ticker thread).
            self._active_ui = ui
            try:
                result = self.agent_import["module"].run_agent(
                    query,
                    self.config,
                    self.client,
                    heads_up=True,
                    incognito=self.incognito,
                    ui=ui,
                )
            finally:
                self._active_ui = None
            if getattr(result, "cancelled", False) is True:
                prompt = result.prompt or query
                with self.lock:
                    can_restore_prompt = (
                        not str(self.editor.get("buffer", ""))
                        and self._answer_request is None
                    )
                    if can_restore_prompt:
                        initialize_text_editor(self.editor, prompt)
                self.add_notice(Text("Cancelled.", style="yellow"))
            elif isinstance(getattr(result, "error", None), str):
                self.add_notice(Text("Turn failed.", style="red"))
            return result
        except KeyboardInterrupt:
            with self.lock:
                if not str(self.editor.get("buffer", "")) and self._answer_request is None:
                    initialize_text_editor(self.editor, query)
            self.add_notice(Text("Cancelled.", style="yellow"))
            return None

    def _queue_or_start_agent_query(self, query: str, on_complete: Callable | None = None) -> None:
        with self.lock:
            if self._agent_busy:
                self._queued_queries.append((query, on_complete))
                queued_position = len(self._queued_queries)
            else:
                self._agent_busy = True
                queued_position = 0
        if queued_position:
            self.add_notice(Text(f"Queued message #{queued_position}.", style="dim"))
            return
        self._start_agent_thread(query, on_complete)

    def _start_agent_thread(self, query: str, on_complete: Callable | None = None) -> None:
        thread = threading.Thread(
            target=self._agent_worker,
            args=(query, on_complete),
            name="headsup-agent-turn",
            daemon=True,
        )
        with self.lock:
            self._agent_thread = thread
        thread.start()

    def _agent_worker(self, query: str, on_complete: Callable | None = None) -> None:
        try:
            result = self._run_agent_query_now(query)
            if on_complete is not None:
                # A post-turn hook (e.g. /btw's return) must not wedge the queue.
                try:
                    on_complete(result)
                except Exception:
                    pass
        finally:
            next_query: str | None = None
            next_complete: Callable | None = None
            with self.lock:
                if self._queued_queries:
                    next_query, next_complete = self._queued_queries.popleft()
                else:
                    self._agent_busy = False
                    self._agent_thread = None
            if next_query is not None:
                self._start_agent_thread(next_query, next_complete)

    def _start_update(self) -> None:
        """Run /update in the background with live progress in the transcript."""
        with self.lock:
            if self._update_task is not None:
                self.add_notice(Text("An update is already running.", style="yellow"))
                return
            task = _UpdateTask(self)
            self._update_task = task
        task.start()

    def _update_finished(self, task: _UpdateTask) -> None:
        with self.lock:
            if self._update_task is task:
                self._update_task = None

    def _cancel_active_turn(self, *, clear_queue: bool = False) -> bool:
        with self.lock:
            token = self._cancel_token
            if clear_queue:
                self._queued_queries.clear()
        if token is None or token.cancelled:
            return False
        token.cancel()
        return True

    def _wait_for_agent_idle(self, *, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self.lock:
                thread = self._agent_thread
                busy = self._agent_busy
            if not busy or thread is None or thread is threading.current_thread():
                return
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining == 0.0:
                return
            thread.join(timeout=remaining)

    def _read_answer_from_foreground(self, label: str, *, echo_answer: bool = True) -> str:
        previous = dict(self.editor)
        with self._answer_condition:
            initialize_text_editor(self.editor, "")
            self._answer_request = {
                "label": label,
                "answer": None,
                "cancelled": False,
                "previous": previous,
                "echo_answer": echo_answer,
            }
            self._reset_prompt_history_navigation()
        self.refresh()

        with self._answer_condition:
            while self._answer_request is not None:
                self._answer_condition.wait()
            request = self._answer_request_completed
        if request.get("cancelled"):
            raise TurnCancelled
        return str(request.get("answer") or "")

    def _complete_answer(self) -> None:
        answer = str(self.editor.get("buffer", "")).strip()
        with self._answer_condition:
            request = self._answer_request
            if request is None:
                return
            if request.get("echo_answer", True):
                label = request.get("label", "answer> ")
                self.add_notice(Text(f"{label}{answer}", style="dim"))
            previous = dict(request.get("previous") or {})
            if previous:
                self.editor = previous
            else:
                initialize_text_editor(self.editor, "")
            request["answer"] = answer
            request["cancelled"] = False
            self._answer_request_completed = request
            self._answer_request = None
            self._answer_condition.notify_all()
        self.refresh()

    def _cancel_answer(self) -> None:
        with self._answer_condition:
            request = self._answer_request
            if request is None:
                return
            previous = dict(request.get("previous") or {})
            if previous:
                self.editor = previous
            else:
                initialize_text_editor(self.editor, "")
            token = self._cancel_token
            if token is not None:
                token.cancel()
            request["cancelled"] = True
            self._answer_request_completed = request
            self._answer_request = None
            self._answer_condition.notify_all()
        self.refresh()

    @contextmanager
    def _captured_console_output(self):
        live = self.live
        self._refresh_suspended += 1
        render_hook_suspended = False
        try:
            render_hooks = getattr(self.console, "_render_hooks", None)
            if live is not None and render_hooks and render_hooks[-1] is live:
                self.console.pop_render_hook()
                render_hook_suspended = True
            try:
                with self.console.capture() as capture:
                    yield capture
            finally:
                if render_hook_suspended and live is not None:
                    self.console.push_render_hook(live)
        finally:
            self._refresh_suspended = max(0, self._refresh_suspended - 1)
            if not self._refresh_suspended:
                # Slash output runs on the loop thread; repaint in place now.
                self.paint_now()

    def _initial_notice_entry(self) -> TranscriptEntry:
        return TranscriptEntry(
            "notice",
            Text("Heads-up mode. Type /help for commands.", style="dim"),
        )

    def _refresh_session_context(self) -> bool:
        old_session_id = self.session_context.session_id
        self.session_context = prepare_session_context(persist_metadata=False)
        self.usage_path = usage_file_for(self.session_context.history_file)
        return self.session_context.session_id != old_session_id

    def _sync_initial_transcript_from_history(self) -> None:
        if self._initial_history_synced:
            return
        self._initial_history_synced = True
        if self.incognito or self._started_new_session:
            return
        self._sync_transcript_from_history()

    def _sync_after_slash(
        self,
        command: str,
        notice: RenderableType | None,
    ) -> bool:
        if command in _HISTORY_SYNC_SLASH_COMMANDS:
            self._refresh_session_context()
            self._sync_transcript_from_history(notice)
            return True
        if command == "/new":
            self._refresh_session_context()
            self._sync_transcript_from_history()
            self._restart_idle_animation()
            self.set_prompt_notice(None)
            return True
        if command in _SESSION_SWITCHING_SLASH_COMMANDS:
            changed = self._refresh_session_context()
            if changed:
                self._sync_transcript_from_history()
                self.set_prompt_notice(notice)
                return True
        return False

    def _sync_transcript_from_history(
        self,
        trailing_notice: RenderableType | None = None,
    ) -> None:
        history = load_history(self.session_context.history_file)
        entries: list[TranscriptEntry] = [self._initial_notice_entry()]
        for item_index, item in enumerate(history):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "status":
                content = _history_content_to_str(item.get("content", "")).strip()
                if content:
                    entries.append(
                        TranscriptEntry(
                            "status",
                            _status_renderable(item),
                        )
                    )
                continue
            if item.get("type") == "function_call":
                entries.append(
                    TranscriptEntry(
                        "tool",
                        _tool_call_renderable(
                            item,
                            _tool_call_output(
                                history,
                                item_index,
                                item.get("call_id"),
                            ),
                        ),
                        spacer_before=len(entries) > 1,
                    )
                )
                continue
            if item.get("type") == "function_call_output":
                continue
            role = str(item.get("role", "")).lower()
            content = _history_content_to_str(item.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                line = Text()
                line.append("\u203a ", style="dim cyan")
                line.append(content, style="bright_white")
                entries.append(
                    TranscriptEntry(
                        "user",
                        line,
                        spacer_before=len(entries) > 1,
                    )
                )
            elif role == "assistant":
                entries.append(
                    TranscriptEntry(
                        "assistant",
                        Markdown(flatten_headings(content)),
                    )
                )
        if trailing_notice is not None:
            entries.append(
                TranscriptEntry(
                    "notice",
                    trailing_notice,
                    spacer_before=len(entries) > 1,
                )
            )
        with self.lock:
            self.entries = entries
            self._live_tool_index.clear()
            self.scroll_offset = 0
            self._prompt_history = self._history_user_messages(history)
            self._reset_prompt_history_navigation()
        self.refresh()

    def _load_prompt_history(self) -> list[str]:
        try:
            return self._history_user_messages(load_history(self.session_context.history_file))
        except Exception:
            return []

    def _history_user_messages(self, history: list) -> list[str]:
        messages: list[str] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            if str(item.get("role", "")).lower() != "user":
                continue
            content = _history_content_to_str(item.get("content", "")).strip()
            if content:
                messages.append(content)
        return messages

    def _record_prompt_history(self, query: str) -> None:
        if query:
            self._prompt_history.append(query)
        self._reset_prompt_history_navigation()

    def _reset_prompt_history_navigation(self) -> None:
        self._prompt_history_index = None
        self._prompt_history_draft = ""

    def _prompt_has_multiline_draft(self) -> bool:
        return "\n" in str(self.editor.get("buffer", ""))

    def _navigate_prompt_history(self, key: str, repeat: int) -> bool:
        if not self._prompt_history:
            return False
        repeat = max(1, repeat)
        if self._prompt_history_index is None:
            if key != "UP":
                return False
            self._prompt_history_draft = str(self.editor.get("buffer", ""))
            index = len(self._prompt_history)
        else:
            index = self._prompt_history_index

        if key == "UP":
            index = max(0, index - repeat)
            value = self._prompt_history[index]
            self._prompt_history_index = index
        else:
            index += repeat
            if index >= len(self._prompt_history):
                value = self._prompt_history_draft
                self._reset_prompt_history_navigation()
            else:
                value = self._prompt_history[index]
                self._prompt_history_index = index

        initialize_text_editor(self.editor, value)
        return True

    # ------------------------------------------------------------------ #
    # Slash-command autocomplete menu (derived state above the input box)
    # ------------------------------------------------------------------ #
    def _slash_menu_context(self) -> tuple[str | None, str] | None:
        """What the menu is completing: ``(command, query)``, or None when closed.

        ``command`` is None while the command token itself is being typed
        (query is the text after the leading ``/``). Once the draft reads
        ``/name <partial>`` for a command with declared argument choices,
        ``command`` is its name and ``query`` the partial first argument. Any
        further whitespace (a second argument, or a space after a complete
        first one) closes the menu, as does answering an agent question or an
        Esc dismissal of this exact draft.
        """
        if self._answer_request is not None:
            return None
        buffer = str(self.editor.get("buffer", ""))
        if not buffer.startswith("/") or "\n" in buffer:
            return None
        if buffer == self._slash_menu_dismissed_for:
            return None
        rest = buffer[1:]
        if not any(ch.isspace() for ch in rest):
            return (None, rest)
        name, _, remainder = rest.partition(" ")
        arg = remainder.lstrip()
        if not name or any(ch.isspace() for ch in arg):
            return None
        name = name.lower()
        if not argument_entries(name):
            return None
        return (name, arg)

    def _slash_menu_entries(self, command: str | None) -> list[MenuEntry]:
        return menu_entries() if command is None else argument_entries(command)

    def _slash_menu_matches(self) -> list[MenuEntry]:
        """Visible menu entries for the current draft (clamps the highlight)."""
        context = self._slash_menu_context()
        if context is None:
            self._slash_menu_index = 0
            self._slash_menu_scroll = 0
            return []
        command, query = context
        matches = filter_entries(self._slash_menu_entries(command), query)
        if not matches:
            self._slash_menu_index = 0
            self._slash_menu_scroll = 0
            return []
        self._slash_menu_index = max(0, min(self._slash_menu_index, len(matches) - 1))
        return matches

    def _menu_enter_submits_draft(self) -> bool:
        """Whether Enter should submit the draft as typed instead of accepting.

        True when there is nothing left to complete: the draft is exactly a
        recognised command with no arguments ("/usage" runs immediately rather
        than gaining a trailing space), the argument token typed so far exactly
        names a choice ("/usage day"), or the argument position is still empty
        ("/usage " submits plain /usage rather than force-picking a choice).
        """
        context = self._slash_menu_context()
        if context is None:
            return True
        command, query = context
        if command is None:
            return self._draft_is_complete_command()
        if not query:
            return True
        needle = query.lower()
        return any(entry.name.lower() == needle for entry in argument_entries(command))

    def _slash_menu_accept(self, entry: MenuEntry, *, run: bool) -> None:
        """Complete ``entry`` into the box; with ``run`` (Enter), execute it too.

        Only entries marked ``runs_on_enter`` execute — the rest ("/usage ",
        "/set model ") still expect input, so accepting them just fills the box
        and lets the menu move on to the next token (or close).
        """
        self._slash_menu_index = 0
        self._slash_menu_scroll = 0
        if run and entry.runs_on_enter:
            command = entry.insert.strip()
            self._record_prompt_history(command)
            initialize_text_editor(self.editor, "")
            self._pastes.clear()
            self._clear_prompt_notice()
            self._exit_armed = False
            if self._handle_query(command) == "exit":
                self.stop()
            return
        initialize_text_editor(self.editor, entry.insert)
        self._clear_prompt_notice()
        self._exit_armed = False
        self.refresh()

    def _slash_menu_box(self, width: int, layout: FrameLayout) -> list[Text]:
        """Build the slash-command popup as the open top of the input-box unit.

        Returns the popup's lines -- a rounded top edge plus one row per visible
        command, and *no* bottom border. The box is compact (sized to its widest
        row) and left-aligned, so the transcript/starfield shows through to its
        right. It is painted over the bottom of the transcript by
        :func:`overlay_menu`; the frame's footer row then draws the edge that docks
        it onto the input field below (see ``render``), so the popup and field read
        as one box. Returns ``[]`` when inactive.
        """
        matches = self._slash_menu_matches()
        if not matches:
            return []
        context = self._slash_menu_context()
        query = context[1] if context else ""
        selected = self._slash_menu_index

        gap = 2
        # The box may grow at most to the field's content area (its width minus a
        # border + gutter on each side); within that it shrinks to fit its rows.
        # Rows carry no selection prefix -- the caret lives in the box's left
        # gutter (see below) so each command's "/" sits in the same column as the
        # input field's draft slash.
        max_content = max(1, width - 4)
        labels = [
            entry.display + (f" {entry.arg_hint}" if entry.arg_hint else "")
            for entry in matches
        ]
        name_col = max((cell_len(label) for label in labels), default=0)
        name_col = max(1, min(name_col, max(1, max_content - gap - 8)))

        # A taller terminal earns a taller window: about a third of the body,
        # clamped between the min and max row counts.
        max_rows = max(_SLASH_MENU_MIN_ROWS, min(_SLASH_MENU_MAX_ROWS, layout.body_height // 3))
        available = max(1, min(max_rows, max(1, layout.body_height - 3)))
        total = len(matches)
        windowed = total > available
        if not windowed:
            start, count = 0, total
            self._slash_menu_scroll = 0
        else:
            # Reserve the bottom row for the scroll-position tail.
            count = available - 1
            # Keep a persistent scroll anchor and only move it when the selection
            # leaves the viewport, so pressing UP/DOWN moves the highlight within
            # the visible rows rather than dragging the whole window each time.
            start = self._slash_menu_scroll
            if selected < start:
                start = selected
            elif selected >= start + count:
                start = selected - count + 1
            start = max(0, min(start, total - count))
            self._slash_menu_scroll = start

        rows: list[tuple[Text, bool]] = [
            (
                self._slash_menu_row(
                    matches[index],
                    index == selected,
                    query,
                    name_col,
                    gap,
                    max_content,
                ),
                index == selected,
            )
            for index in range(start, start + count)
        ]
        # When the list is windowed, the reserved tail row stays put so the box
        # keeps a constant height. It counts the entries *below* the window (so it
        # ticks down as the user scrolls) and reads "no more" once the window
        # reaches the end -- it never reports every off-window entry, and never
        # vanishes mid-scroll and jostles the layout.
        if windowed:
            hidden = total - (start + count)
            tail = f"+{hidden} more" if hidden > 0 else "no more"
            rows.append((Text(tail, style="dim", no_wrap=True, overflow="crop"), False))
        # Shrink-wrap the box to its widest row so it stays compact and left of the
        # transcript, then frame each row to that one width. The selected row's
        # caret is painted into the box gutter, so command text (and its leading
        # "/") starts at the same column as the input field's draft.
        content = max(1, min(max((cell_len(row.plain) for row, _ in rows), default=1), max_content))
        box_width = content + 4
        framed = [box_top(box_width)]
        for row, is_selected in rows:
            gutter = Text("›", style="bold cyan") if is_selected else " "
            framed.append(box_row(row, box_width, gutter=gutter))
        return framed

    def _slash_menu_row(
        self,
        entry: MenuEntry,
        selected: bool,
        query: str,
        name_col: int,
        gap: int,
        width: int,
    ) -> Text:
        line = Text(no_wrap=True, overflow="crop")

        # Keep the command (and its leading "/") aqua on every row — including
        # the selected one, which used to render bright white. The match accent
        # just brightens within the same cyan family so the highlight stays aqua.
        base_style = "bold cyan" if selected else "cyan"
        accent_style = "bold bright_cyan" if selected else "bold cyan"
        display = entry.display
        needle = query.lower()
        pos = display.lower().find(needle) if needle else -1
        if pos >= 0:
            a_start, a_end = pos, pos + len(needle)
            line.append(display[:a_start], style=base_style)
            line.append(display[a_start:a_end], style=accent_style)
            line.append(display[a_end:], style=base_style)
        else:
            line.append(display, style=base_style)

        label_len = cell_len(display)
        if entry.arg_hint:
            line.append(" ")
            line.append(entry.arg_hint, style="dim italic")
            label_len += 1 + cell_len(entry.arg_hint)

        line.append(" " * max(1, name_col + gap - label_len))
        summary_avail = max(0, width - name_col - gap)
        # The summary stays an even dim on every row; the highlighted row is marked
        # by its gutter caret and brighter command, not a white summary that would
        # make the selected row read as a different colour from the rest.
        line.append(clip_text(entry.summary, summary_avail), style="dim")
        return line

    def _slash_menu_hint_row(self, width: int, popup_width: int) -> Text:
        """The popup's key hints, painted beside its bottom row.

        This row sits where the regular footer hints sit while the popup is
        closed; the popup's bottom row is overlaid onto its left cells, so the
        hints start just past the popup's right border. When the popup leaves
        too little room even for the short form, the row stays blank -- a
        mid-word crop would read worse than no hints.
        """
        pad = popup_width + 2
        available = width - pad
        for hints in (_SLASH_MENU_HINTS, _SLASH_MENU_HINTS_SHORT):
            if cell_len(hints) <= available:
                line = Text(no_wrap=True, overflow="crop")
                line.append(" " * pad)
                line.append(hints, style="dim italic")
                return line
        return Text("")

    def _scroll_transcript(self, delta: int) -> None:
        self.scroll_offset = max(0, self.scroll_offset + delta)

    def _append(self, kind: str, renderable: RenderableType, *, spacer_before: bool = False) -> None:
        if kind != "notice":
            self._dismiss_intro()
        with self.lock:
            self.entries.append(TranscriptEntry(kind, renderable, spacer_before=spacer_before))
            # Follow the newest line only while pinned to the bottom. When the user
            # has scrolled up (scroll_offset > 0) to read history, preserve their
            # position instead of snapping back down on every streamed line --
            # add_user_message resets to the bottom when a new turn begins.
        self.refresh()

    def _upsert(self, index: int | None, kind: str, renderable: RenderableType) -> int:
        if kind != "notice":
            self._dismiss_intro()
        with self.lock:
            if index is not None and 0 <= index < len(self.entries):
                self.entries[index] = TranscriptEntry(kind, renderable)
                result = index
            else:
                self.entries.append(TranscriptEntry(kind, renderable))
                result = len(self.entries) - 1
            # See _append: stay put when the user has scrolled up; otherwise the
            # offset is already 0 and the view keeps following the newest content.
        self.refresh()
        return result

    def _transcript_lines(self, width: int) -> list[Text]:
        lines: list[Text] = []
        for entry in self.entries:
            if entry.spacer_before:
                lines.append(Text(""))
            rendered = entry.rendered_lines(width)
            lines.extend(rendered or [Text("")])
        return lines or [Text("")]

    def _prompt_label(self) -> str:
        request = self._answer_request
        return str(request.get("label") if request is not None else "")

    def _prompt_edit_width(self, width: int) -> int:
        label = self._prompt_label()
        if label:
            return max(1, width - cell_len(label))
        return max(1, width - 4)

    def _prompt_lines(self, width: int, *, max_lines: int, menu_open: bool = False) -> list[Text]:
        label = self._prompt_label()
        edit_width = self._prompt_edit_width(width)
        marker_spans = self._pastes.marker_spans(str(self.editor.get("buffer", "")))
        rendered, _cursor_idx, _start = render_visual_line_window(
            self.editor,
            edit_width,
            max_lines=max(1, max_lines - (0 if label else 2)),
            text_style="white",
            cursor_style="reverse",
            highlight_spans=marker_spans,
            highlight_style=_PASTE_MARKER_STYLE,
            selection_span=self._editor_selection_span(),
            selection_style=_SELECTION_STYLE,
        )
        if not rendered:
            rendered = [Text(" ", style="reverse")]
        if not label:
            if _start == 0:
                self._highlight_command_token(rendered)
            return self._prompt_input_box_lines(
                rendered, width, max_lines=max_lines, menu_open=menu_open
            )

        line = Text(label, style="bold cyan", no_wrap=True, overflow="crop")
        line.append_text(rendered[0])
        lines = [line]
        continuation = " " * cell_len(label)
        for visual_line in rendered[1:]:
            wrapped = Text(continuation, style="dim")
            wrapped.append_text(visual_line)
            lines.append(wrapped)
        return lines[:max(1, max_lines)]

    def _highlight_command_token(self, rendered: list[Text]) -> None:
        """Paint a recognised "/command" token aqua on the first prompt row.

        Only the leading command word is restyled (any trailing arguments stay
        white), matching the cyan the autocomplete menu uses so a valid command
        visibly "lights up" as it is completed. Called only when the window is
        anchored at the buffer's start, so the token is always on ``rendered[0]``.
        """
        length = self._command_highlight_len()
        if length and rendered:
            rendered[0].stylize(_VALID_COMMAND_STYLE, 0, length)

    def _command_highlight_len(self) -> int:
        """Length of the leading "/command" token when it names a real command.

        Returns 0 unless the single-line draft starts with a slash command jarv
        recognises (the registry plus the headsup-only ``/exit``/``/quit``). The
        leading slash is included so the whole token is highlighted.
        """
        if self._answer_request is not None:
            return 0
        buffer = str(self.editor.get("buffer", ""))
        if not buffer.startswith("/") or "\n" in buffer:
            return 0
        token = buffer.split(None, 1)[0]
        name = token[1:].lower()
        if name in COMMANDS or token.lower() in {"/exit", "/quit"}:
            return len(token)
        return 0

    def _draft_is_complete_command(self) -> bool:
        """Whether the draft is exactly a complete, recognised "/command".

        True when the single-line draft is a recognised slash command with no
        arguments and no trailing whitespace (so the autocomplete menu is still
        open). Lets Enter run a fully typed command immediately — even one that
        *accepts* parameters, like "/usage" — instead of completing it with a
        trailing space the way Tab does.
        """
        buffer = str(self.editor.get("buffer", ""))
        return bool(buffer) and self._command_highlight_len() == len(buffer)

    def _prompt_input_box_lines(
        self, rendered: list[Text], width: int, *, max_lines: int, menu_open: bool = False
    ) -> list[Text]:
        # When the slash popup is open it supplies the top edge and the divider
        # that closes onto this field, so we omit our own top border and let the
        # divider serve as it -- the popup and field then frame as one box.
        rows: list[Text] = [] if menu_open else [box_top(width)]
        for visual_line in rendered[:max(1, max_lines - 2)]:
            rows.append(box_row(visual_line, width))
        rows.append(box_bottom(width))
        return rows[:max(1, max_lines)]

    def _footer_line(self, width: int) -> Text:
        with self.lock:
            notice = self._prompt_notice
            has_draft = bool(str(self.editor.get("buffer", "")))
            has_selection = self._editor_selection_span() is not None
            answering = self._answer_request is not None
            cancelling = self._cancel_token is not None
            exit_armed = self._exit_armed

        if notice is not None:
            lines = rendered_text_lines(notice, width)
            footer = lines[0].copy() if lines else Text("")
            footer.truncate(max(1, width), overflow="ellipsis")
            footer.no_wrap = True
            footer.overflow = "crop"
            return footer

        if exit_armed:
            value = "Esc/Ctrl+C exit   Any other key continue"
        elif answering:
            value = "Enter answer   Ctrl+N newline   Esc no response/cancel"
        elif has_selection:
            value = "Enter send   Ctrl+C copy selection   Esc clear draft   Wheel/PgUp/PgDn scroll"
        elif has_draft:
            value = "Enter send   Ctrl+N newline   Esc/Ctrl+C clear draft   Wheel/PgUp/PgDn scroll"
        elif cancelling:
            value = "Enter send   Ctrl+N newline   Esc/Ctrl+C cancel turn   Wheel/PgUp/PgDn scroll"
        else:
            value = "Enter send   Ctrl+N newline   Esc/Ctrl+C clear/exit/cancel   Wheel/PgUp/PgDn scroll   /exit quit"
        return Text(
            clip_text(value, width),
            style="dim italic",
            no_wrap=True,
            overflow="crop",
        )

    def _usage_status(self, width: int) -> Text:
        session_id = self.session_context.session_id
        now = time.monotonic()
        cached = self._usage_status_cache
        if (
            cached is not None
            and now - cached[0] <= _USAGE_STATUS_CACHE_TTL
            and cached[1] == width
            and cached[2] == self.usage_path
            and cached[3] == session_id
        ):
            return cached[4].copy()

        try:
            usage = load_usage(self.usage_path, session_id, warn=False)
        except Exception:
            usage = {}
        totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else {}
        last_root = usage.get("last_root_request") if isinstance(usage.get("last_root_request"), dict) else None

        status = Text(no_wrap=True, overflow="crop")
        cost = usage_cost_summary(totals)
        if cost["exact_requests"] or cost["estimated_requests"] or cost["has_tracked_cost"]:
            if cost["estimated_requests"] and not cost["exact_requests"]:
                status.append("est. ", style="dim")
            status.append(format_cost(cost["total_usd"]), style="green")
        else:
            status.append("$0.00", style="green")
        if cost["unknown_requests"] or cost["contract_requests"]:
            status.append(" incomplete", style="yellow")

        status.append(" · ", style="dim")
        context_percent: float | None = None
        request_count = int(totals.get("request_count") or 0)
        if request_count == 0 and last_root is None:
            context_percent = 0.0
        elif isinstance(last_root, dict):
            model = str(last_root.get("model") or self.config.get("model") or "")
            context_window = known_context_window(model, config=self.config)
            if context_window:
                input_tokens = int(last_root.get("input_tokens") or 0)
                context_percent = min(max((input_tokens / context_window) * 100, 0.0), 999.9)

        if context_percent is None:
            status.append("context unknown", style="dim")
        else:
            context_label = "0%" if context_percent == 0.0 else f"{context_percent:.1f}%"
            status.append(context_label, style=_context_fill_style(context_percent))
            status.append(" full", style="dim")

        status.truncate(max(1, width), overflow="ellipsis")
        self._usage_status_cache = (
            now,
            width,
            self.usage_path,
            session_id,
            status.copy(),
        )
        return status


def run_heads_up_mode(
    config: dict,
    client,
    *,
    args: argparse.Namespace | None,
    agent_loader: tuple[dict, threading.Event],
    handle_slash: SlashHandler,
    maybe_command: MaybeCommand,
) -> None:
    app = HeadsupApp(
        config,
        client,
        args=args,
        agent_loader=agent_loader,
        handle_slash=handle_slash,
        maybe_command=maybe_command,
    )
    app.run()
