"""Alternate-screen heads-up mode UI."""

from __future__ import annotations

import argparse
import threading
import time
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
    mouse_capture,
    requeue_key,
)
from .agent import (
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
    refresh_on_resize,
    rendered_text_lines,
    terminal_size,
)
from .history import load_history, prepare_session_context
from .session_render import (
    _history_content_to_str,
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

_COMMAND_TAKES_REST = {
    "setup": True,
    "help": False,
    "about": False,
    "update": False,
    "new": False,
    "archive": False,
    "session": True,
    "sessions": True,
    "history": False,
    "usage": True,
    "set": True,
    "unset": True,
    "config": False,
    "settings": False,
    "undo": True,
    "redo": True,
}

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
            return
        if live_kind == "SpawnPanel":
            self.app.upsert_live_tool(live_kind, renderable)
            return
        if self._tool_live_kind is not None:
            self.app.replace_live_tool(self._tool_live_kind, renderable)
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
        self.app.add_tool(
            Panel(
                Markdown(flatten_headings(question)),
                title="[bold blue]Ask user[/bold blue]",
                title_align="left",
                border_style="blue",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
        return self.app.read_answer("answer> ")

    def bind_cancel_token(self, token: CancellationToken) -> None:
        self.app.bind_cancel_token(token)

    def unbind_cancel_token(self) -> None:
        self._response_waiting = False
        self._tool_waiting = False
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
        self.session_context = prepare_session_context(persist_metadata=False)
        self.usage_path = usage_file_for(self.session_context.history_file)

    def run(self) -> None:
        with Live(
            get_renderable=self.render,
            console=self.console,
            screen=True,
            auto_refresh=False,
            transient=False,
            vertical_overflow="crop",
        ) as live, refresh_on_resize(live, on_change=self.refresh), mouse_capture():
            self.live = live
            while True:
                self.refresh()
                try:
                    key, repeat = _read_key_with_repeats(
                        text_mode=True,
                        batch_text=True,
                    )
                except KeyboardInterrupt:
                    if self._handle_prompt_dismiss():
                        break
                    continue

                if key == "ENTER":
                    query = str(self.editor.get("buffer", "")).strip()
                    initialize_text_editor(self.editor, "")
                    self._exit_armed = False
                    if self._handle_query(query) == "exit":
                        break
                elif key == "ESC":
                    if self._handle_prompt_dismiss():
                        break
                elif key == "PAGEUP":
                    self.scroll_offset += 5 * repeat
                elif key == "PAGEDOWN":
                    self.scroll_offset = max(0, self.scroll_offset - 5 * repeat)
                else:
                    changed = apply_text_editor_key(
                        self.editor,
                        key,
                        repeat,
                        content_width=1,
                        allow_newlines=False,
                    )
                    if changed or isinstance(key, TextInput):
                        self.scroll_offset = 0
                        self._exit_armed = False

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

        while len(visible) < transcript_rows:
            visible.insert(0, Text(""))

        footer = Text(
            clip_text(
                "Enter send   Esc/Ctrl+C clear/exit/cancel   PgUp/PgDn scroll   /exit quit",
                inner_width,
            ),
            style="dim italic",
            no_wrap=True,
            overflow="crop",
        )
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
        left = "jarv \u25b8 heads up"
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

    def read_answer(self, label: str) -> str:
        previous = dict(self.editor)
        initialize_text_editor(self.editor, "")
        self._pause_esc_listener()
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
                    self.add_notice(Text(f"{label}{answer}", style="dim"))
                    return answer
                if key == "ESC":
                    if self._cancel_token is not None:
                        self._cancel_token.cancel()
                        raise TurnCancelled
                    return "[no response]"
                apply_text_editor_key(
                    self.editor,
                    key,
                    repeat,
                    content_width=1,
                    allow_newlines=False,
                )
        finally:
            self._resume_esc_listener()
            self.editor = previous

    def bind_cancel_token(self, token: CancellationToken) -> None:
        self.unbind_cancel_token()
        self._cancel_token = token
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
            self.add_notice(Text("Draft cleared.", style="dim"))
            return False
        if self._exit_armed:
            return True
        self._exit_armed = True
        self.add_notice(Text("Press Esc or Ctrl+C again to exit.", style="yellow"))
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
        name = first_word.lower()
        takes_rest = _COMMAND_TAKES_REST.get(name)
        if takes_rest is None:
            return None
        if not takes_rest and rest:
            return None
        return f"/{name}", rest

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
                ui=ui,
            )
            if getattr(result, "cancelled", False) is True:
                initialize_text_editor(self.editor, result.prompt or query)
                self.add_notice(Text("Cancelled.", style="yellow"))
            elif isinstance(getattr(result, "error", None), str):
                self.add_notice(Text("Turn failed.", style="red"))
        except KeyboardInterrupt:
            initialize_text_editor(self.editor, query)
            self.add_notice(Text("Cancelled.", style="yellow"))

    @contextmanager
    def _captured_console_output(self):
        live = self.live
        self._refresh_suspended += 1
        try:
            with self._preserve_alt_screen():
                if live is not None:
                    live.stop()
                try:
                    with self.console.capture() as capture:
                        yield capture
                finally:
                    if live is not None:
                        live.start(refresh=True)
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

    def _sync_after_slash(
        self,
        command: str,
        notice: RenderableType | None,
    ) -> bool:
        if command in _HISTORY_SYNC_SLASH_COMMANDS:
            self._refresh_session_context()
            self._sync_transcript_from_history(notice)
            return True
        if command in _SESSION_SWITCHING_SLASH_COMMANDS:
            changed = self._refresh_session_context()
            if changed:
                self._sync_transcript_from_history(notice)
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
        self.refresh()

    def _append(self, kind: str, renderable: RenderableType, *, spacer_before: bool = False) -> None:
        with self.lock:
            self.entries.append(TranscriptEntry(kind, renderable, spacer_before=spacer_before))
            self.scroll_offset = 0
        self.refresh()

    def _upsert(self, index: int | None, kind: str, renderable: RenderableType) -> int:
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
        label = "jarv> "
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
        if isinstance(last_root, dict):
            model = str(last_root.get("model") or self.config.get("model") or "")
            context_window = known_context_window(model, config=self.config)
            if context_window:
                input_tokens = int(last_root.get("input_tokens") or 0)
                context_percent = min(max((input_tokens / context_window) * 100, 0.0), 999.9)

        if context_percent is None:
            status.append("context unknown", style="dim")
        else:
            status.append(
                f"{context_percent:.1f}% full",
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
