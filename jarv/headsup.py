"""Alternate-screen heads-up mode UI."""

from __future__ import annotations

import argparse
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
from .command_input import (
    TextInput,
    _key_available,
    _read_key,
    _read_key_with_repeats,
    disable_mouse_capture,
    requeue_key,
    strip_sgr_mouse_sequences,
)
from .command_registry import parse_command_alias
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
    refresh_on_resize,
    rendered_text_lines,
    terminal_size,
    tool_card,
)
from .history import forget_current_session, load_history, prepare_session_context
from .intro_animation import render_intro
from .session_render import (
    _history_content_to_str,
    _status_renderable,
    _tool_call_output,
    _tool_call_renderable,
)
from .text_editor import apply_text_editor_key, initialize_text_editor, render_visual_line_window
from .tui_app import AltScreenApp
from .tui_frame import (
    assemble_body,
    build_frame,
    compose_title,
    compute_layout,
    panel_width as _panel_width,
    transcript_rows as _transcript_rows_for,
    window_transcript,
)
from .tui_layout import clip_text
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


class HeadsupApp(AltScreenApp):
    # Heads-up keeps its own SGR-mouse handling, so it disables Rich/terminal
    # mouse capture rather than letting the base app capture the mouse.
    clear_on_resize = False
    translate_mouse_wheel = False
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
            refresh_on_resize_fn=self._headsup_refresh_on_resize,
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
        self.scroll_offset = 0
        self.lock = threading.RLock()
        self._exit_armed = False
        self._cancel_token: CancellationToken | None = None
        self._esc_listener_stop: threading.Event | None = None
        self._esc_listener_thread: threading.Thread | None = None
        self._esc_listener_paused = threading.Event()
        self._live_tool_index: dict[str, int] = {}
        self._refresh_suspended = 0
        self._foreground_input_active = False
        self._foreground_input_thread: threading.Thread | None = None
        self._idle_anim_started_at = 0.0
        self._idle_anim_stop = threading.Event()
        self._outro_started_at = 0.0
        self._active_ui: HeadsupAgentUI | None = None
        self._last_wait_tick = 0.0
        self._agent_busy = False
        self._agent_thread: threading.Thread | None = None
        self._queued_queries: deque[str] = deque()
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
        self._prompt_notice: RenderableType | None = None
        self._usage_status_cache: tuple[float, int, object, str | None, Text] | None = None

    # ------------------------------------------------------------------ #
    # AltScreenApp wiring: resolve patchable module symbols at call time so
    # tests that patch ``jarv.headsup.*`` keep driving the loop.
    # ------------------------------------------------------------------ #
    def _build_live(self, get_renderable, _console):
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

    def _headsup_refresh_on_resize(self, live, *, on_change):
        return refresh_on_resize(live, on_change=on_change)

    # ------------------------------------------------------------------ #
    # Lifecycle (single-threaded loop owned by AltScreenApp)
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self._sync_initial_transcript_from_history()
        super().run()

    def on_start(self) -> None:
        disable_mouse_capture()
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
        if self._handle_prompt_dismiss():
            self.stop()

    def on_key(self, key: str, repeat: int) -> None:
        if key == "ENTER":
            if self._answer_request is not None:
                self._complete_answer()
                return
            raw_query = str(self.editor.get("buffer", ""))
            query = raw_query.strip("\r\n")
            if not query.strip():
                initialize_text_editor(self.editor, "")
                self._clear_prompt_notice()
                self._exit_armed = False
                return
            self._record_prompt_history(query)
            initialize_text_editor(self.editor, "")
            self._clear_prompt_notice()
            self._exit_armed = False
            if self._handle_query(query) == "exit":
                self.stop()
            return
        if key == "ESC":
            if self._answer_request is not None:
                self._cancel_answer()
                return
            if self._handle_prompt_dismiss():
                self.stop()
            return
        if key == "PAGEUP":
            self._scroll_transcript(5 * repeat)
            return
        if key == "PAGEDOWN":
            self._scroll_transcript(-5 * repeat)
            return
        if key == "MOUSE_WHEEL_UP":
            self._scroll_transcript(3 * repeat)
            return
        if key == "MOUSE_WHEEL_DOWN":
            self._scroll_transcript(-3 * repeat)
            return
        if key == "MOUSE_WHEEL_PAGEUP":
            self._scroll_transcript(5 * repeat)
            return
        if key == "MOUSE_WHEEL_PAGEDOWN":
            self._scroll_transcript(-5 * repeat)
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

    def on_tick(self) -> None:
        now = time.perf_counter()
        repaint = False
        if self._idle_animation_active() or (
            self._outro_started_at and now - self._outro_started_at < _OUTRO_DURATION
        ):
            # The intro/outro frame is derived from the clock in render(), so a
            # plain repaint advances the animation.
            repaint = True
        elif self._outro_started_at:
            self._outro_started_at = 0.0
            repaint = True
        ui = self._active_ui
        if ui is not None and ui.has_active_animation():
            if now - self._last_wait_tick >= 0.2:
                self._last_wait_tick = now
                # Recomputes the spinner text in place (and invalidates).
                ui._refresh_wait_statuses()
        if repaint:
            self.invalidate()

    def render(self) -> RenderableType:
        term_w, term_h = terminal_size(console=self.console)
        layout = compute_layout(term_w, term_h)
        inner_width = layout.inner_width
        model_status = _model_status(self.config)
        title = compose_title(model_status, layout.panel_width)

        with self.lock:
            prompt_lines = self._prompt_lines(inner_width, max_lines=layout.max_prompt_rows)
            footer = self._footer_line(inner_width)
            rows = _transcript_rows_for(layout.body_height, len(prompt_lines))
            transcript = self._transcript_lines(inner_width)
            visible, self.scroll_offset = window_transcript(transcript, rows, self.scroll_offset)
            show_intro = (
                not self._idle_anim_stop.is_set()
                and self.scroll_offset == 0
                and self._answer_request is None
                and all(entry.kind == "notice" for entry in self.entries)
            )
            outro_started_at = self._outro_started_at

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
            visible = intro

        parts = assemble_body(visible, footer, prompt_lines, layout.body_height, rows)
        subtitle = self._panel_subtitle(inner_width)
        return build_frame(
            parts,
            title=title,
            subtitle=subtitle,
            panel_width=layout.panel_width,
            term_h=layout.term_h,
        )

    def _panel_subtitle(self, width: int) -> Text:
        subtitle = Text(no_wrap=True, overflow="crop")
        subtitle.append_text(self._usage_status(width))
        subtitle.truncate(max(1, width), overflow="ellipsis")
        return subtitle

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
        """
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
        if self._foreground_input_active:
            if self._foreground_input_thread is threading.current_thread():
                return self._read_answer_direct(label, echo_answer=echo_answer)
            return self._read_answer_from_foreground(label, echo_answer=echo_answer)
        return self._read_answer_direct(label, echo_answer=echo_answer)

    def _read_answer_direct(self, label: str, *, echo_answer: bool = True) -> str:
        previous = dict(self.editor)
        initialize_text_editor(self.editor, "")
        self._pause_esc_listener()
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
            self._resume_esc_listener()
            self.editor = previous
            with self._answer_condition:
                self._answer_request = None
                self._answer_request_completed = {}
                self._answer_condition.notify_all()

    def _current_prompt_edit_width(self) -> int:
        term_w, _term_h = terminal_size(console=self.console)
        inner_width = max(1, _panel_width(max(20, term_w)) - 4)
        return self._prompt_edit_width(inner_width)

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
                return stripped != existing_tail, bool(stripped)

        key = _sanitize_editor_key(key)
        if key == "CTRL_N":
            key = "ENTER"
        changed = apply_text_editor_key(
            self.editor,
            key,
            repeat,
            content_width=self._current_prompt_edit_width(),
            allow_newlines=True,
        )
        return changed, isinstance(key, TextInput)

    def bind_cancel_token(self, token: CancellationToken) -> None:
        self.unbind_cancel_token()
        self._cancel_token = token
        if self._foreground_input_active:
            return
        self._esc_listener_stop = threading.Event()
        self._esc_listener_paused.clear()
        self._esc_listener_thread = threading.Thread(
            target=self._esc_cancel_loop,
            name="headsup-esc-cancel",
            daemon=True,
        )
        self._esc_listener_thread.start()

    def unbind_cancel_token(self) -> None:
        stop = self._esc_listener_stop
        thread = self._esc_listener_thread
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.2)
        self._esc_listener_stop = None
        self._esc_listener_thread = None
        self._cancel_token = None
        self._esc_listener_paused.clear()

    def _pause_esc_listener(self) -> None:
        self._esc_listener_paused.set()

    def _resume_esc_listener(self) -> None:
        self._esc_listener_paused.clear()

    def _esc_cancel_loop(self) -> None:
        stop = self._esc_listener_stop
        if stop is None:
            return
        while not stop.is_set():
            if self._esc_listener_paused.is_set():
                stop.wait(0.05)
                continue
            if not _key_available():
                stop.wait(0.05)
                continue
            key = _read_key(text_mode=True)
            if key == "ESC":
                token = self._cancel_token
                if token is not None:
                    token.cancel()
                return
            requeue_key(key)

    def _handle_prompt_dismiss(self) -> bool:
        if str(self.editor.get("buffer", "")):
            initialize_text_editor(self.editor, "")
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
        live = self.live
        with self._preserve_alt_screen():
            if live is not None:
                live.stop()
            try:
                self.config, self.client = self.handle_slash(
                    command,
                    rest,
                    self.config,
                    self.client,
                    self.args,
                    True,
                )
            finally:
                if live is not None:
                    live.start(refresh=True)
        self._sync_after_slash(command, None)

    def _run_agent_query(self, query: str) -> None:
        if self.live is not None:
            self._queue_or_start_agent_query(query)
            return
        self._run_agent_query_now(query)

    def _run_agent_query_now(self, query: str) -> None:
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
        except KeyboardInterrupt:
            with self.lock:
                if not str(self.editor.get("buffer", "")) and self._answer_request is None:
                    initialize_text_editor(self.editor, query)
            self.add_notice(Text("Cancelled.", style="yellow"))

    def _queue_or_start_agent_query(self, query: str) -> None:
        with self.lock:
            if self._agent_busy:
                self._queued_queries.append(query)
                queued_position = len(self._queued_queries)
            else:
                self._agent_busy = True
                queued_position = 0
        if queued_position:
            self.add_notice(Text(f"Queued message #{queued_position}.", style="dim"))
            return
        self._start_agent_thread(query)

    def _start_agent_thread(self, query: str) -> None:
        thread = threading.Thread(
            target=self._agent_worker,
            args=(query,),
            name="headsup-agent-turn",
            daemon=True,
        )
        with self.lock:
            self._agent_thread = thread
        thread.start()

    def _agent_worker(self, query: str) -> None:
        try:
            self._run_agent_query_now(query)
        finally:
            next_query: str | None = None
            with self.lock:
                if self._queued_queries:
                    next_query = self._queued_queries.popleft()
                else:
                    self._agent_busy = False
                    self._agent_thread = None
            if next_query is not None:
                self._start_agent_thread(next_query)

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

    @contextmanager
    def _preserve_alt_screen(self):
        original = self.console.set_alt_screen

        def set_alt_screen(enable: bool = True) -> bool:
            if enable is False:
                return True
            return original(enable)

        self.console.set_alt_screen = set_alt_screen
        try:
            yield
        finally:
            self.console.set_alt_screen = original

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

    def _scroll_transcript(self, delta: int) -> None:
        self.scroll_offset = max(0, self.scroll_offset + delta)

    def _append(self, kind: str, renderable: RenderableType, *, spacer_before: bool = False) -> None:
        if kind != "notice":
            self._dismiss_intro()
        with self.lock:
            self.entries.append(TranscriptEntry(kind, renderable, spacer_before=spacer_before))
            self.scroll_offset = 0
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
            self.scroll_offset = 0
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

    def _prompt_lines(self, width: int, *, max_lines: int) -> list[Text]:
        label = self._prompt_label()
        edit_width = self._prompt_edit_width(width)
        rendered, _cursor_idx, _start = render_visual_line_window(
            self.editor,
            edit_width,
            max_lines=max(1, max_lines - (0 if label else 2)),
            text_style="white",
            cursor_style="reverse",
        )
        if not rendered:
            rendered = [Text(" ", style="reverse")]
        if not label:
            return self._prompt_input_box_lines(rendered, width, max_lines=max_lines)

        line = Text(label, style="bold cyan", no_wrap=True, overflow="crop")
        line.append_text(rendered[0])
        lines = [line]
        continuation = " " * cell_len(label)
        for visual_line in rendered[1:]:
            wrapped = Text(continuation, style="dim")
            wrapped.append_text(visual_line)
            lines.append(wrapped)
        return lines[:max(1, max_lines)]

    def _prompt_input_box_lines(self, rendered: list[Text], width: int, *, max_lines: int) -> list[Text]:
        border_style = "dim cyan"
        field_width = max(1, width - 2)
        content_width = max(1, width - 4)
        has_horizontal_padding = width >= 4
        top = Text("\u256d" + "\u2500" * field_width + "\u256e", style=border_style, no_wrap=True)
        bottom = Text("\u2570" + "\u2500" * field_width + "\u256f", style=border_style, no_wrap=True)
        rows: list[Text] = [top]
        for visual_line in rendered[:max(1, max_lines - 2)]:
            line = Text(no_wrap=True, overflow="crop")
            line.append("\u2502", style=border_style)
            if has_horizontal_padding:
                line.append(" ")
            line.append_text(visual_line)
            padding = max(0, content_width - cell_len(visual_line.plain))
            if padding:
                line.append(" " * padding)
            if has_horizontal_padding:
                line.append(" ")
            line.append("\u2502", style=border_style)
            rows.append(line)
        rows.append(bottom)
        return rows[:max(1, max_lines)]

    def _footer_line(self, width: int) -> Text:
        with self.lock:
            notice = self._prompt_notice
            has_draft = bool(str(self.editor.get("buffer", "")))
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
