"""Alternate-screen heads-up mode UI."""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

from rich import box
from rich.cells import cell_len
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
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
    response_wait_label,
    thought_complete_indicator,
    tool_activity_label,
    tool_complete_indicator,
)
from .config import DEFAULT_CONFIG
from .display import (
    TITLE_STYLE,
    console,
    flatten_headings,
    mark_first_paint,
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
from .text_editor import apply_text_editor_key, initialize_text_editor, render_single_line
from .tui_layout import append_bottom_footer, clip_text
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


def _panel_width(terminal_width: int) -> int:
    return max(1, terminal_width - 1)


def _context_fill_style(percent: float | None) -> str:
    if percent is None:
        return "dim"
    if percent >= 90:
        return "bold bright_red"
    if percent >= 70:
        return "bold yellow"
    return "cyan"


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
        self._ticker_lock = threading.Lock()
        self._ticker_thread: threading.Thread | None = None
        self._animated_live_tool_keys: set[str] = set()

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
        self.app.add_user_message(query)

    def start_response_wait(self, start_time: float) -> None:
        self._response_started_at = start_time
        self._has_reasoning = False
        self._response_waiting = True
        self._response_status_index = self.app.upsert_status(
            self._response_status_index,
            self._response_wait_text(),
        )
        self._ensure_ticker()

    def set_response_wait_has_reasoning(self, has_reasoning: bool) -> None:
        self._has_reasoning = has_reasoning
        self._response_status_index = self.app.upsert_status(
            self._response_status_index,
            self._response_wait_text(),
        )

    def complete_response_phase(self, status_text: str) -> None:
        self._response_waiting = False
        self._response_status_index = self.app.upsert_status(
            self._response_status_index,
            thought_complete_indicator(status_text),
        )
        self._response_status_index = None

    def start_tool_activity(self, start_time: float) -> None:
        self._tool_started_at = start_time
        self._tool_names = ()
        self._tool_waiting = True
        self._tool_status_index = self.app.upsert_status(
            self._tool_status_index,
            self._tool_wait_text(),
        )
        self._ensure_ticker()

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
        self._stream_index = self.app.upsert_assistant_message(
            self._stream_index,
            self._stream_text,
        )

    def replace_stream_text(self, text: str) -> None:
        self._stream_text = text
        self._stream_index = self.app.upsert_assistant_message(
            self._stream_index,
            self._stream_text,
        )

    def finish_assistant_message(self, text: str) -> None:
        if text and text != self._stream_text:
            self.replace_stream_text(text)
        elif self._stream_index is not None:
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
        live_kind = type(renderable).__name__
        if live_kind == "RunningCommandCard":
            self.app.upsert_live_tool(live_kind, renderable)
            self._tool_live_kind = live_kind
            self._animated_live_tool_keys.add(live_kind)
            self._ensure_ticker()
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

    def _ensure_ticker(self) -> None:
        with self._ticker_lock:
            if self._ticker_thread is not None and self._ticker_thread.is_alive():
                return
            self._ticker_thread = threading.Thread(
                target=self._ticker_loop,
                name="headsup-status-ticker",
                daemon=True,
            )
            self._ticker_thread.start()

    def _ticker_loop(self) -> None:
        while True:
            time.sleep(0.25)
            if not self._refresh_wait_statuses():
                return

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


class HeadsupApp:
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
        self.config = dict(config)
        self.client = client
        self.args = args
        self.agent_import, self.agent_ready = agent_loader
        self.handle_slash = handle_slash
        self.maybe_command = maybe_command
        self.console = render_console
        self.entries: list[TranscriptEntry] = [self._initial_notice_entry()]
        self.editor: dict = {}
        initialize_text_editor(self.editor, "")
        self.scroll_offset = 0
        self.live: Live | None = None
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
        self._idle_anim_thread: threading.Thread | None = None
        self._outro_started_at = 0.0
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

    def run(self) -> None:
        self._sync_initial_transcript_from_history()
        disable_mouse_capture()
        try:
            with Live(
                get_renderable=self.render,
                console=self.console,
                screen=True,
                auto_refresh=False,
                transient=False,
                vertical_overflow="crop",
            ) as live, refresh_on_resize(live, on_change=self.refresh):
                mark_first_paint("headsup")
                self.live = live
                self._foreground_input_active = True
                self._foreground_input_thread = threading.current_thread()
                self._start_idle_animation()
                try:
                    while True:
                        self.refresh()
                        try:
                            key, repeat = _read_key_with_repeats(
                                text_mode=True,
                                batch_text=True,
                                repeatable=_HEADSUP_REPEATABLE_KEYS,
                                translate_mouse_wheel=False,
                            )
                        except KeyboardInterrupt:
                            if self._handle_prompt_dismiss():
                                break
                            continue

                        if key == "ENTER":
                            if self._answer_request is not None:
                                self._complete_answer()
                                continue
                            query = str(self.editor.get("buffer", "")).strip()
                            self._record_prompt_history(query)
                            initialize_text_editor(self.editor, "")
                            self._clear_prompt_notice()
                            self._exit_armed = False
                            if self._handle_query(query) == "exit":
                                break
                        elif key == "ESC":
                            if self._answer_request is not None:
                                self._cancel_answer()
                                continue
                            if self._handle_prompt_dismiss():
                                break
                        elif key == "PAGEUP":
                            self._scroll_transcript(5 * repeat)
                        elif key == "PAGEDOWN":
                            self._scroll_transcript(-5 * repeat)
                        elif key == "MOUSE_WHEEL_UP":
                            self._scroll_transcript(3 * repeat)
                        elif key == "MOUSE_WHEEL_DOWN":
                            self._scroll_transcript(-3 * repeat)
                        elif key == "MOUSE_WHEEL_PAGEUP":
                            self._scroll_transcript(5 * repeat)
                        elif key == "MOUSE_WHEEL_PAGEDOWN":
                            self._scroll_transcript(-5 * repeat)
                        elif key in {"UP", "DOWN"} and self._answer_request is None:
                            if self._navigate_prompt_history(key, repeat):
                                self.scroll_offset = 0
                                self._clear_prompt_notice()
                                self._exit_armed = False
                        else:
                            changed, user_text = self._apply_editor_key(key, repeat)
                            if changed or user_text:
                                if self._answer_request is None:
                                    self._reset_prompt_history_navigation()
                                self.scroll_offset = 0
                                self._clear_prompt_notice()
                                self._exit_armed = False
                finally:
                    self._idle_anim_stop.set()
                    if self._answer_request is not None:
                        self._cancel_answer()
                    self._foreground_input_active = False
                    self._foreground_input_thread = None
                    self._cancel_active_turn(clear_queue=True)
                    self._wait_for_agent_idle(timeout=5.0)
                    self.live = None
        finally:
            disable_mouse_capture()

    def render(self) -> RenderableType:
        term_w, term_h = terminal_size(console=self.console)
        term_w = max(20, term_w)
        term_h = max(8, term_h)
        panel_width = _panel_width(term_w)
        inner_width = max(1, panel_width - 4)
        body_height = max(3, term_h - 2)
        prompt_rows = 3
        transcript_rows = max(1, body_height - prompt_rows)

        with self.lock:
            transcript = self._transcript_lines(inner_width)
            max_scroll = max(0, len(transcript) - transcript_rows)
            self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))
            end = len(transcript) - self.scroll_offset
            start = max(0, end - transcript_rows)
            visible = list(transcript[start:end])
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
                transcript_rows,
                time.perf_counter() - self._idle_anim_started_at,
            )
        elif outro_started_at:
            exit_progress = (time.perf_counter() - outro_started_at) / _OUTRO_DURATION
            if exit_progress < 1.0:
                intro = render_intro(
                    inner_width,
                    transcript_rows,
                    time.perf_counter() - self._idle_anim_started_at,
                    exit=exit_progress,
                )
        if intro is not None:
            visible = intro

        while len(visible) < transcript_rows:
            visible.insert(0, Text(""))

        footer = self._footer_line(inner_width)
        prompt = self._prompt_line(inner_width)
        parts: list[RenderableType] = visible
        append_bottom_footer(
            parts,
            body_height,
            prompt,
            border_rows=0,
            footer_rows=2,
            crop=True,
        )
        parts[-2] = footer
        model_status = f"{self.config.get('provider', 'openai')} / {self.config.get('model', DEFAULT_CONFIG['model'])}"
        subtitle = self._panel_subtitle(inner_width)
        return Panel(
            Group(*parts),
            title=self._panel_title(model_status, panel_width),
            title_align="left",
            subtitle=subtitle,
            subtitle_align="right",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            width=panel_width,
            height=term_h,
        )

    def _panel_title(self, model_status: str, panel_width: int) -> Text:
        left = "jarv \u25b8 heads-up"
        title_width = max(1, panel_width - 5)
        title = Text(no_wrap=True, overflow="crop")
        title.append(left, style=TITLE_STYLE)
        remaining = title_width - cell_len(left)
        if remaining <= 1:
            return title
        status = clip_text(model_status, remaining - 1)
        separator_width = title_width - cell_len(left) - cell_len(status) - 2
        if separator_width > 0:
            title.append(" ")
            title.append("─" * separator_width, style="cyan")
            title.append(" ")
        else:
            title.append(" ")
        title.append(status, style="dim")
        return title

    def _panel_subtitle(self, width: int) -> Text:
        subtitle = Text(no_wrap=True, overflow="crop")
        subtitle.append_text(self._usage_status(width))
        subtitle.truncate(max(1, width), overflow="ellipsis")
        return subtitle

    def refresh(self) -> None:
        if self._refresh_suspended:
            return
        if self.live is not None:
            with self.lock:
                self.live.refresh()

    def _idle_animation_active(self) -> bool:
        if self._idle_anim_stop.is_set():
            return False
        if self.scroll_offset:
            return False
        with self.lock:
            if self._answer_request is not None:
                return False
            return all(entry.kind == "notice" for entry in self.entries)

    def _start_idle_animation(self) -> None:
        if self._idle_anim_thread is not None:
            return
        if not self._idle_animation_active():
            return
        self._idle_anim_started_at = time.perf_counter()
        self._idle_anim_stop.clear()
        thread = threading.Thread(
            target=self._idle_animation_loop,
            name="headsup-intro-anim",
            daemon=True,
        )
        self._idle_anim_thread = thread
        thread.start()

    def _restart_idle_animation(self) -> None:
        thread = self._idle_anim_thread
        self._idle_anim_stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.2)
        self._idle_anim_thread = None
        self._outro_started_at = 0.0
        self._idle_anim_started_at = time.perf_counter()
        self._idle_anim_stop.clear()
        if self._foreground_input_active:
            self._start_idle_animation()

    def _dismiss_intro(self) -> None:
        """Tear down the idle intro, playing a quick outro if it's on screen.

        Called the first time real transcript content lands. If the intro is
        currently visible we kick off a short, non-blocking dissolve before the
        transcript takes over; otherwise we just mark it dismissed.
        """
        if self._idle_anim_stop.is_set():
            return
        was_visible = self._idle_animation_active()
        self._idle_anim_stop.set()
        if was_visible and self._foreground_input_active:
            self._outro_started_at = time.perf_counter()
            self._start_outro_animation()

    def _start_outro_animation(self) -> None:
        if not self._foreground_input_active:
            return
        thread = threading.Thread(
            target=self._outro_animation_loop,
            name="headsup-intro-outro",
            daemon=True,
        )
        thread.start()

    def _outro_animation_loop(self) -> None:
        # ~30 fps for the brief dissolve, then land on the cleared frame.
        deadline = self._outro_started_at + _OUTRO_DURATION
        while time.perf_counter() < deadline:
            if not self._foreground_input_active:
                return
            self.refresh()
            time.sleep(1 / 30)
        self._outro_started_at = 0.0
        self.refresh()

    def _idle_animation_loop(self) -> None:
        # ~14 fps; the wait() both paces the loop and exits promptly on stop.
        current = threading.current_thread()
        while not self._idle_anim_stop.wait(1 / 14):
            if not self._foreground_input_active:
                continue
            if not self._idle_animation_active():
                break
            self.refresh()
        if self._idle_anim_thread is current:
            self._idle_anim_thread = None

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
                self.refresh()
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
        changed = apply_text_editor_key(
            self.editor,
            key,
            repeat,
            content_width=1,
            allow_newlines=False,
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
            result = self.agent_import["module"].run_agent(
                query,
                self.config,
                self.client,
                heads_up=True,
                incognito=self.incognito,
                ui=ui,
            )
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
            self.refresh()

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
            rendered = rendered_text_lines(entry.renderable, width)
            lines.extend(rendered or [Text("")])
        return lines or [Text("")]

    def _prompt_line(self, width: int) -> Text:
        request = self._answer_request
        label = str(request.get("label") if request is not None else "jarv> ")
        line = Text(label, style="bold cyan", no_wrap=True, overflow="crop")
        edit_width = max(1, width - len(label))
        line.append_text(
            render_single_line(
                self.editor,
                edit_width,
                text_style="bright_white",
                cursor_style="reverse",
            )
        )
        return line

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
            value = "Enter answer   Esc no response/cancel"
        elif has_draft:
            value = "Enter send   Esc/Ctrl+C clear draft   Wheel/PgUp/PgDn scroll"
        elif cancelling:
            value = "Enter send   Esc/Ctrl+C cancel turn   Wheel/PgUp/PgDn scroll"
        else:
            value = "Enter send   Esc/Ctrl+C clear/exit/cancel   Wheel/PgUp/PgDn scroll   /exit quit"
        return Text(
            clip_text(value, width),
            style="dim italic",
            no_wrap=True,
            overflow="crop",
        )

    def _usage_status(self, width: int) -> Text:
        try:
            usage = load_usage(self.usage_path, self.session_context.session_id, warn=False)
        except Exception:
            usage = {}
        totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else {}
        last_root = usage.get("last_root_request") if isinstance(usage.get("last_root_request"), dict) else None

        status = Text(no_wrap=True, overflow="crop")
        cost = usage_cost_summary(totals)
        if cost["exact_requests"] or cost["estimated_requests"] or cost["has_tracked_cost"]:
            if cost["estimated_requests"] and not cost["exact_requests"]:
                status.append("est. ", style="dim")
            else:
                status.append("cost ", style="dim")
            status.append(format_cost(cost["total_usd"]), style="green")
        else:
            status.append("cost ", style="dim")
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
            context_label = "0% full" if context_percent == 0.0 else f"{context_percent:.1f}% full"
            status.append(
                context_label,
                style=_context_fill_style(context_percent),
            )

        status.truncate(max(1, width), overflow="ellipsis")
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
