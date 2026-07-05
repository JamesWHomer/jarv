"""Rich terminal UI helpers and tool dispatch wrappers for the root agent."""

import os
import platform
import sys
import threading
import time
from contextlib import contextmanager

from rich.console import Group
from rich.control import Control, ControlType
from rich.live import Live
from rich.live_render import LiveRender
from rich.markdown import Markdown
from rich.segment import Segment
from rich.text import Text

from .cancellation import CancellationToken, TurnCancelled
from .command_input import read_editable_line
from .config import DEFAULT_CONFIG, get_setting
from .display import (
    console,
    flatten_headings,
    output_display_split,
    output_renderable,
    rendered_text_lines,
    terminal_size,
    tool_card,
    track_live_display,
)
from .history import get_shell_name
from .orchestrator import (
    SpawnObserver,
    parse_spawn_children,
    spawn_tool_output,
)
from .retained_outputs import RetainedOutputStore
from .usage import format_cost, format_int, load_usage, usage_cost_summary


def _ui_call(ui, method: str, *args, **kwargs):
    if ui is None:
        return None
    handler = getattr(ui, method, None)
    if handler is None:
        return None
    return handler(*args, **kwargs)


def print_mode_spacer(config: dict, *, mode: str | None = None) -> None:
    """Emit the one trailing blank line that separates cards in print mode.

    The inline ("print") tool-call display draws each card into normal
    scrollback and relies on a single blank line as the separator. Routing
    every separator through here keeps the spacing identical across tool
    cards, ask_user, spawn, and run_command instead of each site re-deciding.

    ``mode`` lets a caller that has already resolved the effective display
    mode (e.g. ``auto`` -> ``print``) pass it through instead of re-reading
    the raw config value.
    """
    effective = mode if mode is not None else get_setting(config, "tool_call_display")
    if effective == "print":
        console.print()


def _print_tool_card(renderable, config: dict, ui=None) -> None:
    """Print a tool card with the spacing required by its display mode."""
    if ui is not None:
        _ui_call(ui, "show_tool_card", renderable)
        return
    console.print(renderable)
    print_mode_spacer(config)


def _replace_terminal_rows(row_count: int) -> bool:
    """Clear the preceding terminal rows and return the cursor to their start."""
    if row_count <= 0 or not console.is_terminal:
        return False
    output = console.file
    output.write("\x1b[?25l")
    output.write(f"\x1b[{row_count}A")
    for index in range(row_count):
        output.write("\r\x1b[2K")
        if index < row_count - 1:
            output.write("\x1b[1B")
    if row_count > 1:
        output.write(f"\x1b[{row_count - 1}A")
    output.write("\r\x1b[?25h")
    output.flush()
    return True


_THINKING_FRAMES = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]
STREAM_PREVIEW_REFRESH_INTERVAL = 1 / 12


def response_wait_label(has_reasoning: bool) -> str:
    """Return the live wait label for the response stream.

    Interactive ``run_command`` continuations never reach this label: their
    "deciding next input" footer is owned by :class:`InteractiveCommandCard`,
    which stands in for the response-wait spinner for the whole session.
    """
    return "Thinking" if has_reasoning else "Waiting"


_TOOL_ACTIVITY_LABELS = {
    "run_command": ("Writing command", "Wrote command"),
    "spawn": ("Planning parallel tasks", "Planned parallel tasks"),
    "read": ("Selecting content", "Selected content"),
    "edit": ("Writing edit", "Wrote edit"),
    "ask_user": ("Writing question", "Wrote question"),
    "web_search": ("Writing web search", "Wrote web search"),
}


def tool_activity_label(tool_names: tuple[str, ...]) -> str:
    """Return the live activity label for tool-call serialization."""
    if len(tool_names) != 1:
        return f"Preparing {len(tool_names)} actions"
    return _TOOL_ACTIVITY_LABELS.get(
        tool_names[0],
        ("Preparing action", "Prepared action"),
    )[0]


class ResponseWaitIndicator:
    """Animated response wait line with live elapsed timer."""

    def __init__(self, start_time: float):
        self._start = start_time
        self.has_reasoning = False

    def __rich_console__(self, console, options):
        now = time.perf_counter()
        elapsed = now - self._start
        frame = _THINKING_FRAMES[int(now * 10) % len(_THINKING_FRAMES)]
        label = response_wait_label(self.has_reasoning)
        yield Text(f"{frame}  {label}\u2026  {int(elapsed)}s")


class ToolActivityIndicator:
    """Animated tool-call serialization line with its own elapsed timer."""

    def __init__(self, start_time: float):
        self._start = start_time
        self._tool_names: dict[str, str] = {}

    def start_tool_call(self, key: str, name: str) -> None:
        self._tool_names[key] = name

    def __rich_console__(self, console, options):
        now = time.perf_counter()
        elapsed = now - self._start
        frame = _THINKING_FRAMES[int(now * 10) % len(_THINKING_FRAMES)]
        label = tool_activity_label(tuple(self._tool_names.values()))
        yield Text(f"{frame}  {label}\u2026  {int(elapsed)}s")


class RunningCommandCard:
    """Render a command immediately with a live elapsed timer."""

    def __init__(
        self,
        command: str,
        metadata: str,
        display_mode: str,
        start_time: float,
    ):
        self._command = command
        self._metadata = metadata
        self._display_mode = display_mode
        self._start = start_time

    def __rich_console__(self, console, options):
        elapsed = int(max(0.0, time.perf_counter() - self._start))
        command_line = Text("> ", style="bold yellow")
        command_line.append(self._command)
        body = command_line
        if self._display_mode != "fullscreen":
            body = Group(
                command_line,
                Text(f"Running\u2026 {elapsed}s", style="dim"),
            )
        yield tool_card(
            "run_command",
            body,
            metadata=self._metadata,
            display_mode=self._display_mode,
            status=f"running {elapsed}s",
            status_style="blue",
        )


class InteractiveCommandCard:
    """One growing transcript box for a whole interactive ``run_command`` session.

    Replaces the old "one fresh box per model step" rendering: the ``> command``
    header and ``model window`` metadata are drawn once and each model step appends
    a compact ``stdin> …`` marker (with the model's per-step decision time) plus the
    new delta output. The footer animates while the model decides the next input and
    resolves to ``exit N`` on completion. One instance is shared between the inline
    Rich ``Live`` (which auto-refreshes the footer) and the heads-up live-tool slot.
    """

    def __init__(self, command: str, metadata: str, display_mode: str, start_time: float):
        self.command = command
        self.metadata = metadata
        self.display_mode = display_mode
        self._start = start_time
        # Each segment: {"marker": Text | None, "output": str, "seconds": float | None}.
        self._segments: list[dict] = []
        self._state = "running"  # running | thinking | waiting | done
        self._think_start: float | None = None
        self._check_in = False
        self._exit_code: int | None = None
        self._lock = threading.Lock()

    @property
    def exited(self) -> bool:
        return self._state == "done"

    def seed_initial(self, snapshot) -> None:
        """Record the command's first batch of output once it goes idle/exits."""
        result = snapshot.to_delta_command_result()
        with self._lock:
            self._segments.append(
                {"marker": None, "output": result.full_model_output(), "seconds": None}
            )
            if snapshot.exited:
                self._state = "done"
                self._exit_code = snapshot.exit_code
            else:
                self._state = "waiting"
                self._check_in = bool(getattr(snapshot, "check_in", False))

    def set_thinking(self, start_time: float) -> None:
        with self._lock:
            self._state = "thinking"
            self._think_start = start_time

    def add_step(
        self,
        marker: Text,
        output: str,
        seconds: float | None,
        *,
        exited: bool,
        exit_code: int | None = None,
        check_in: bool = False,
    ) -> None:
        with self._lock:
            self._segments.append(
                {"marker": marker, "output": output, "seconds": seconds}
            )
            if exited:
                self._state = "done"
                self._exit_code = exit_code
            else:
                self._state = "waiting"
                self._check_in = check_in

    def _footer(self) -> Text | None:
        if self._state == "running":
            elapsed = int(max(0.0, time.perf_counter() - self._start))
            return Text(f"Running… {elapsed}s", style="dim")
        if self._state == "thinking":
            now = time.perf_counter()
            frame = _THINKING_FRAMES[int(now * 10) % len(_THINKING_FRAMES)]
            elapsed = int(max(0.0, now - (self._think_start or now)))
            return Text(f"{frame}  deciding next input…  {elapsed}s", style="yellow")
        if self._state == "waiting":
            if self._check_in:
                return Text(
                    "Still running — model deciding whether to wait or step in",
                    style="dim",
                )
            return Text(
                "Idle on stdin — model deciding the next input", style="dim"
            )
        # done
        if self._exit_code in (None, 0):
            return Text("exit 0", style="dim")
        return Text(f"exit {self._exit_code}", style="bold red")

    def _body_parts(self):
        command_line = Text("> ", style="bold yellow")
        command_line.append(self.command)
        parts = [command_line]
        # Fullscreen heads-up renders into a scrollable transcript, so each delta
        # is kept whole and scrolling reaches it. Inline pins the card inside a
        # Rich ``Live`` that cannot scroll, so long deltas collapse to head/tail.
        full = self.display_mode == "fullscreen"
        for segment in self._segments:
            marker = segment["marker"]
            if marker is not None:
                line = marker.copy()
                if segment["seconds"] is not None:
                    line.append(f"   ·{segment['seconds']:.1f}s", style="dim")
                parts.append(line)
            output = segment["output"]
            if output:
                parts.append(
                    Text(output, style="dim") if full else output_renderable(output)
                )
        return parts

    def __rich_console__(self, console, options):
        with self._lock:
            parts = self._body_parts()
            footer = self._footer()
            state = self._state
            exit_code = self._exit_code
        inner_width = max(
            10,
            options.max_width - (4 if self.display_mode == "fullscreen" else 2),
        )
        lines = rendered_text_lines(Group(*parts), inner_width)
        # Inline pins the card in a Rich ``Live`` that cannot scroll past the
        # terminal, so collapse the middle to a head/tail window. Fullscreen
        # heads-up renders into a scrollable transcript, so keep the whole
        # session (every stdin marker + delta output) and let scrolling reach it.
        if self.display_mode != "fullscreen":
            # Crop against the height Rich's ``Live`` itself uses for overflow
            # (``options.size.height``), not ``os.get_terminal_size()``. When those
            # two diverge the card builds itself taller than the Live's crop window,
            # so Rich replaces the newest rows with its own "…" ellipsis and the
            # back-and-forth appears frozen. The reserve leaves room for the card
            # header and footer, which are added outside ``lines``.
            term_h = options.size.height
            budget = max(8, term_h - 8)
            if len(lines) > budget:
                head_n, tail_n = output_display_split(budget)
                hidden = len(lines) - head_n - tail_n
                lines = (
                    lines[:head_n]
                    + [Text(f"… {hidden} earlier lines hidden …", style="dim italic")]
                    + lines[-tail_n:]
                )
        body_items = list(lines)
        if footer is not None:
            body_items.append(footer)
        if state == "done":
            status, status_style = "done", (
                "green" if exit_code in (None, 0) else "red"
            )
        else:
            status, status_style = "waiting", "blue"
        yield tool_card(
            "run_command",
            Group(*body_items),
            metadata=self.metadata,
            display_mode=self.display_mode,
            status=status,
            status_style=status_style,
        )


def _start_response_wait_indicator(
    interactive: bool,
    start_time: float,
) -> tuple[ResponseWaitIndicator | None, Live | None]:
    if not interactive:
        return None, None
    wait_indicator = ResponseWaitIndicator(start_time)
    spinner_live = Live(
        wait_indicator,
        refresh_per_second=4,
        console=console,
        auto_refresh=True,
        transient=True,
    )
    spinner_live.start()
    return wait_indicator, spinner_live


def _markdown_tail_source(text: str, max_chars: int) -> str:
    """Cap the markdown handed to the live preview so rendering stays cheap.

    This is a *silent* performance ceiling only — it carries no visible banner.
    The single "earlier lines hidden" hint is owned by :class:`TailMarkdown`'s
    line-level crop, so the preview never stacks two hints on top of each other.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    newline = tail.find("\n")
    if newline >= 0:
        tail = tail[newline + 1:]
    return tail


class TailMarkdown:
    """Renders Markdown but keeps only the last few rendered rows.

    Live can't move the cursor above the top of the terminal viewport, so if
    the rendered content ever exceeds the visible height the redraw lands at
    row 0 and the prior frame stays in scrollback — producing duplicates. By
    pre-cropping to the viewport from the top we guarantee the live region
    never overflows, while still showing the most recent (streaming) tail.

    ``max_lines`` is the row budget. Pass ``None`` (the streaming default) to
    derive it from the *current* terminal height at paint time, so the crop
    tracks a mid-stream resize instead of a value frozen at the first token.
    A pinned integer is still accepted for deterministic tests.
    """

    def __init__(
        self,
        text: str,
        max_lines: int | None = None,
        max_source_chars: int = 8000,
        *,
        reserve_rows: int = 2,
    ):
        self._text = _markdown_tail_source(text, max_source_chars)
        self._max_lines = max(1, max_lines) if max_lines is not None else None
        self._reserve_rows = max(1, reserve_rows)

    def _row_budget(self, console) -> int:
        if self._max_lines is not None:
            return self._max_lines
        _, term_h = terminal_size(console=console)
        return max(1, term_h - self._reserve_rows)

    def __rich_console__(self, console, options):
        max_lines = self._row_budget(console)
        # Always reserve exactly one top row for the hint/spacer so the live
        # block grows to `max_lines` and then holds steady — no one-row jump
        # when the hint first appears at the overflow threshold.
        content_budget = max(0, max_lines - 1)
        md = Markdown(self._text)
        lines = console.render_lines(md, options, pad=False)
        hidden = max(0, len(lines) - content_budget)
        if hidden:
            lines = lines[-content_budget:] if content_budget else []
            hint = Text(
                f"↑ {hidden} earlier line{'s' if hidden != 1 else ''} hidden — full reply will print when done",
                style="dim italic",
            )
            yield from console.render(hint, options)
        else:
            yield Segment.line()
        for line in lines:
            yield from line
            yield Segment.line()


class InPlaceLiveRender(LiveRender):
    """Overwrite live rows in place instead of blanking the entire block first."""

    _erase_to_end = Control((ControlType.ERASE_IN_LINE, 0)).segment

    def position_cursor(self) -> Control:
        if self._shape is None:
            return Control()
        _, height = self._shape
        return Control(
            ControlType.CARRIAGE_RETURN,
            *((ControlType.CURSOR_UP, 1),) * max(0, height - 1),
        )

    def __rich_console__(self, console, options):
        previous_height = self._shape[1] if self._shape is not None else 0
        rendered = list(super().__rich_console__(console, options))
        current_width, current_height = self._shape or (0, 0)

        for segment in rendered:
            if segment.text == "\n" and not segment.control:
                yield self._erase_to_end
            yield segment
        if current_height:
            yield self._erase_to_end

        stable_height = max(previous_height, current_height)
        for _ in range(stable_height - current_height):
            yield Segment.line()
            yield self._erase_to_end
        self._shape = (current_width, stable_height)


class InPlaceLive(Live):
    """Rich Live variant that avoids a visible clear-then-redraw flash."""

    def __init__(self, renderable=None, **kwargs):
        super().__init__(renderable, **kwargs)
        self._live_render = InPlaceLiveRender(
            self.get_renderable(),
            vertical_overflow=self.vertical_overflow,
        )


class StreamingMarkdownPreview:
    """Coalesce text deltas into bounded, manually refreshed preview frames."""

    def __init__(
        self,
        live: Live,
        max_lines: int | None = None,
        *,
        refresh_interval: float = STREAM_PREVIEW_REFRESH_INTERVAL,
        clock=time.perf_counter,
    ):
        self._live = live
        self._max_lines = max_lines
        self._refresh_interval = max(0.0, refresh_interval)
        self._clock = clock
        self._chunks: list[str] = []
        self._last_refresh_at: float | None = None
        self._dirty = False

    @property
    def text(self) -> str:
        return "".join(self._chunks)

    def append(self, delta: str) -> None:
        self._chunks.append(delta)
        self._dirty = True
        now = self._clock()
        if (
            self._last_refresh_at is None
            or now - self._last_refresh_at >= self._refresh_interval
        ):
            self._render(now, refresh=True)

    def replace(self, text: str) -> None:
        self._chunks = [text]
        self._dirty = True

    def flush(self, *, refresh: bool = True) -> None:
        if self._dirty:
            self._render(self._clock(), refresh=refresh)

    def _render(self, now: float, *, refresh: bool) -> None:
        source = flatten_headings(self.text)
        self._live.update(
            TailMarkdown(source, self._max_lines),
            refresh=refresh,
        )
        self._last_refresh_at = now
        self._dirty = False


def thought_complete_indicator(text: str) -> Text:
    """Return the static completed-thinking bubble."""
    return Text(f"\u2726 {text}", style="dim")


def tool_complete_indicator(text: str) -> Text:
    """Return the static completed-tool bubble."""
    return Text(f"\u2713 {text}", style="dim")



def format_thought_duration(seconds: float) -> str:
    """Return a compact human-readable duration for the thinking timer."""
    rounded = round(max(0.0, seconds), 1)
    unit = "second" if rounded == 1 else "seconds"
    return f"{rounded:.1f} {unit}"


def response_start_status(seconds: float, has_reasoning: bool) -> str:
    """Return the completed wait-status text for the first visible response.

    Interactive ``run_command`` continuations don't emit this status — their
    per-step decision time is folded into the card's "·N.Ns" marker instead.
    """
    duration = format_thought_duration(seconds)
    if has_reasoning:
        return f"Thought for {duration}."
    return f"Started responding in {duration}."


def tool_activity_complete_status(seconds: float, tool_names: tuple[str, ...]) -> str:
    """Return the completed status text for tool-call serialization."""
    duration = format_thought_duration(seconds)
    if len(tool_names) != 1:
        return f"Prepared {len(tool_names)} actions in {duration}."
    completed = _TOOL_ACTIVITY_LABELS.get(
        tool_names[0],
        ("Preparing action", "Prepared action"),
    )[1]
    return f"{completed} in {duration}."


def _format_agent_usage_line(usage: dict) -> Text | None:
    totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else {}
    last_root = usage.get("last_root_request") if isinstance(usage.get("last_root_request"), dict) else None
    if not isinstance(last_root, dict):
        return None

    session_total = int(totals.get("total_tokens") or 0)
    last_total = int(last_root.get("total_tokens") or 0)
    if session_total <= 0 and last_total <= 0:
        return None

    input_tokens = int(last_root.get("input_tokens") or 0)
    cached_input = int(last_root.get("cached_input_tokens") or 0)
    output_tokens = int(last_root.get("output_tokens") or 0)

    line = Text("Usage: ", style="dim")
    line.append(format_int(input_tokens), style="bold")
    line.append(" in", style="dim")
    if cached_input:
        line.append(" (", style="dim")
        line.append(format_int(cached_input), style="cyan")
        line.append(" cached)", style="dim")
    line.append(" · ", style="dim")
    line.append(format_int(output_tokens), style="bold")
    line.append(" out · ", style="dim")
    line.append(format_int(last_total), style="bold")
    line.append(" last · ", style="dim")
    line.append(format_int(session_total), style="bold")
    line.append(" session", style="dim")

    cost = usage_cost_summary(totals)
    known_cost_requests = cost["exact_requests"] + cost["estimated_requests"]
    if known_cost_requests or cost["has_tracked_cost"]:
        label = "cost " if cost["exact_requests"] and not cost["estimated_requests"] else "est. "
        line.append(f" · {label}", style="dim")
        line.append(format_cost(cost["total_usd"]), style="green")
    if cost["unknown_requests"] or cost["contract_requests"]:
        line.append(" · cost incomplete", style="yellow")
    if last_root.get("estimated"):
        line.append(" · usage estimated", style="yellow")
    return line


def _print_agent_usage_if_enabled(
    config: dict,
    usage_path,
    session_id: str | None,
    ui=None,
    *,
    heads_up: bool = False,
) -> None:
    if heads_up:
        return
    if not get_setting(config, "print_usage_after_agent"):
        return
    usage_line = _format_agent_usage_line(load_usage(usage_path, session_id, warn=False))
    if usage_line is not None:
        if ui is not None:
            _ui_call(ui, "show_usage_line", usage_line)
        else:
            console.print(usage_line)


def get_system_info() -> str:
    shell = get_shell_name()
    parts = [
        f"OS: {platform.system()} {platform.release()}",
        f"CWD: {os.getcwd()}",
        f"Shell: {shell}",
    ]
    if platform.system() == "Windows" and "PowerShell 5.1" in shell:
        parts.append("Shell syntax: Windows PowerShell 5.1; `&&` is not supported. Use `;` or `if ($?) { ... }`.")
    user = os.environ.get("USERNAME") or os.environ.get("USER")
    if user:
        parts.append(f"User: {user}")
    return "\n".join(parts)


@contextmanager
def _ask_user_terminal_input():
    """Temporarily route ask_user input to the controlling terminal when possible."""
    def is_tty(stream) -> bool:
        isatty = getattr(stream, "isatty", None)
        return callable(isatty) and bool(isatty())

    if is_tty(sys.stdin):
        yield True
        return

    if not is_tty(sys.stdout):
        yield False
        return

    if sys.platform == "win32":
        yield True
        return

    encoding = getattr(sys.stdin, "encoding", None)
    if not isinstance(encoding, str) or not encoding:
        encoding = "utf-8"
    try:
        tty = open("/dev/tty", "r", encoding=encoding)
    except OSError:
        yield False
        return

    original_stdin = sys.stdin
    try:
        sys.stdin = tty
        yield True
    finally:
        sys.stdin = original_stdin
        tty.close()


def _dispatch_ask_user(args: dict, config: dict | None = None, ui=None) -> str:
    question = args.get("question")
    if not isinstance(question, str) or not question.strip():
        msg = "[tool argument error: question must be a non-empty string]"
        if ui is not None:
            _ui_call(ui, "show_error", msg)
        else:
            console.print(f"[red]{msg}[/red]")
        return msg
    if ui is not None:
        answer = _ui_call(ui, "ask_user", question, config or DEFAULT_CONFIG)
        return str(answer) if answer is not None else "[no response]"
    with _ask_user_terminal_input() as can_prompt:
        if not can_prompt:
            return "[non-interactive session; user unavailable]"
        config = config or DEFAULT_CONFIG
        display_mode = get_setting(config, "tool_call_display")
        if display_mode == "auto":
            display_mode = "print"
        question_renderable = Markdown(flatten_headings(question))
        if display_mode == "print":
            console.print(
                tool_card(
                    "ask_user",
                    question_renderable,
                    status="waiting",
                    status_style="blue",
                    display_mode="print",
                )
            )
        else:
            waiting_card = tool_card(
                "ask_user",
                question_renderable,
                status="waiting",
                status_style="blue",
                display_mode="fullscreen",
            )
            waiting_height = len(
                console.render_lines(waiting_card, console.options, pad=False)
            )
            console.print(
                waiting_card
            )
        try:
            prompt = (
                "\x1b[34m\u258e\x1b[0m \x1b[1;36m>\x1b[0m "
                if display_mode == "print"
                else "\x1b[1;36m>\x1b[0m "
            )
            answer = read_editable_line(prompt, text_style="\x1b[97m").strip()
        except KeyboardInterrupt:
            raise
        except EOFError:
            answer = "[no response]"
            if display_mode == "print":
                console.print(f"\n[dim]{answer}[/dim]")
            else:
                console.print()
        if display_mode == "fullscreen":
            answer_line = Text("> ", style="bold cyan")
            answer_line.append(answer, style="bright_white")
            _replace_terminal_rows(waiting_height + 1)
            console.print(
                tool_card(
                    "ask_user",
                    Group(question_renderable, answer_line),
                    status="done",
                    display_mode="fullscreen",
                )
            )
        print_mode_spacer(config, mode=display_mode)
        return answer


def _dispatch_spawn_with_ui(
    args: dict,
    root_node,
    store,
    client,
    config,
    cancellation_token: CancellationToken | None = None,
    retained_store: RetainedOutputStore | None = None,
    ui=None,
) -> str:
    children_raw = parse_spawn_children(args)
    if isinstance(children_raw, str):
        if ui is not None:
            _ui_call(ui, "show_error", children_raw)
        else:
            console.print(f"[red]{children_raw}[/red]")
        return children_raw

    top_labels = [c.get("label", "?") for c in children_raw if isinstance(c, dict)]
    # state per label: {status, depth, tldr?, reason?}
    states: dict[str, dict] = {}
    # parent_label -> ordered child labels. Top-level entries live under root_node.label.
    children_of: dict[str, list[str]] = {root_node.label: list(top_labels)}
    for lbl in top_labels:
        states[lbl] = {"status": "running", "depth": 0}
    lock = threading.Lock()

    def _push_spawn_panel() -> None:
        if ui is not None:
            _ui_call(ui, "show_tool_card", SpawnPanel())

    class PanelObserver(SpawnObserver):
        def on_spawn_start(self, parent_label: str, child_labels: list[str]) -> None:
            with lock:
                if parent_label == root_node.label:
                    parent_depth = -1
                else:
                    parent_depth = states.get(parent_label, {}).get("depth", 0)
                bucket = children_of.setdefault(parent_label, [])
                for cl in child_labels:
                    if cl not in states:
                        states[cl] = {"status": "running", "depth": parent_depth + 1}
                        bucket.append(cl)
            _push_spawn_panel()

        def on_child_done(self, parent_label: str, label: str, result: dict) -> None:
            with lock:
                existing = states.get(label, {"depth": 0})
                states[label] = {**existing, **result}
            _push_spawn_panel()

    observer = PanelObserver()

    class SpawnPanel:
        def __rich_console__(self, con, options):
            now = time.perf_counter()
            frame = _THINKING_FRAMES[int(now * 10) % len(_THINKING_FRAMES)]
            with lock:
                # DFS in insertion order to display children directly under
                # their parent, indented one level per depth.
                ordered: list[str] = []

                def walk(parent: str) -> None:
                    for cl in children_of.get(parent, []):
                        ordered.append(cl)
                        walk(cl)

                walk(root_node.label)
                snap = {lbl: dict(states[lbl]) for lbl in ordered}
            lines = []
            for lbl in ordered:
                state = snap[lbl]
                status = state["status"]
                indent = "  " * state.get("depth", 0)
                line = Text()
                line.append(indent)
                if status == "running":
                    line.append(f" {frame} ", style="yellow")
                    line.append(lbl, style="bold")
                elif status == "done":
                    line.append(" ✓ ", style="bold green")
                    line.append(lbl, style="bold cyan")
                    line.append(f"  {state.get('tldr', '')}", style="dim")
                else:
                    line.append(" ✗ ", style="bold red")
                    line.append(lbl, style="bold cyan")
                    line.append(f"  {state.get('reason', '')}", style="dim red")
                lines.append(line)
            total = len(snap)
            done = sum(1 for s in snap.values() if s["status"] != "running")
            yield tool_card(
                "spawn",
                Group(*lines),
                status=f"{done}/{total} done",
                status_style="green" if done == total else "magenta",
                display_mode=get_setting(config, "tool_call_display"),
            )

    def _run_spawn() -> str:
        return spawn_tool_output(
            root_node,
            children_raw,
            store,
            client,
            config,
            spawn_observer=observer,
            cancellation_token=cancellation_token,
            retained_store=retained_store,
        )

    if ui is not None:
        _push_spawn_panel()
        try:
            output = _run_spawn()
        except (KeyboardInterrupt, TurnCancelled):
            with lock:
                for state in states.values():
                    if state["status"] == "running":
                        state["status"] = "cancelled"
                        state["reason"] = "cancelled"
            _push_spawn_panel()
            raise
        if output.startswith("[error:") or output.startswith("[tool argument error:"):
            _ui_call(ui, "show_error", output)
            return output
        _push_spawn_panel()
        return output

    with track_live_display(), Live(
        SpawnPanel(),
        refresh_per_second=10,
        console=console,
        auto_refresh=True,
        transient=False,
        vertical_overflow="visible",
    ) as live:
        try:
            output = _run_spawn()
        except (KeyboardInterrupt, TurnCancelled):
            with lock:
                for state in states.values():
                    if state["status"] == "running":
                        state["status"] = "cancelled"
                        state["reason"] = "cancelled"
            live.update(SpawnPanel())
            raise
        if output.startswith("[error:") or output.startswith("[tool argument error:"):
            live.update(Text(output, style="red"))
            return output
        live.update(SpawnPanel())

    print_mode_spacer(config)