import json
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass

from rich.console import Group
from rich.control import Control, ControlType
from rich.live import Live
from rich.live_render import LiveRender
from rich.markdown import Markdown
from rich.markup import escape
from rich.segment import Segment
from rich.text import Text

from .config import DEFAULT_CONFIG
from .context_budget import build_input, trim_turn_input
from .cancellation import CancellationToken, TurnCancelled, cancel_token_on_sigint
from .command_input import read_editable_line
from .display import (
    console,
    flatten_headings,
    terminal_size,
    tool_card,
    track_live_display,
)
from .history import (
    artifact_file_for,
    forget_current_session,
    get_shell_name,
    history_metadata,
    load_history,
    prepare_session_context,
    reads_file_for,
    redo_file_for,
    save_history,
)
from .artifacts import ArtifactStore, load_artifact_store, save_artifact_store
from .provider import (
    create_client,
    ProviderError,
    RetryableStreamError,
    ReasoningDone,
    ReasoningStarted,
    StreamDone,
    TextDelta,
    ToolCallDone,
    ToolCallStarted,
    response_output_text,
    responses_input_id,
    stream_response,
)
from .orchestrator import (
    ASK_USER_TOOL,
    RUN_COMMAND_TOOL,
    SPAWN_TOOL,
    AgentNode,
    DepthExceeded,
    SpawnObserver,
    append_web_search_read_nudge,
    dispatch_parallel_safe_tool_batch,
    filter_enabled_tools,
    history_has_web_search_read_nudge,
    spawn_batch,
    tool_enabled,
    tool_call_is_parallel_safe,
)
from .safety import check_command
from .shell import (
    COMMAND_OUTPUT_UNSET,
    command_result_renderable,
    execute_command,
    resolve_command_output_window,
    truncate_model_output,
)
from .read_tool import READ_TOOL, retain_command_output
from .retained_outputs import (
    RetainedOutputStore,
    load_retained_output_store,
    save_retained_output_store,
)
from .tool_outputs import ToolOutput
from .usage import (
    estimate_context_breakdown,
    format_cost,
    format_int,
    load_usage,
    record_response_usage,
    usage_cost_summary,
    usage_file_for,
)
from .provider_catalog import configured_service_tier
from .web import WEB_SEARCH_TOOL

# Responses API tool format (flat, no "function" wrapper key)
TOOLS = [
    RUN_COMMAND_TOOL,
    WEB_SEARCH_TOOL,
    SPAWN_TOOL,
    READ_TOOL,
    ASK_USER_TOOL,
]


def build_agent_tools(config: dict) -> list[dict]:
    return filter_enabled_tools(TOOLS, config)


def resolve_tool_call_display(config: dict, *, heads_up: bool) -> str:
    mode = config.get("tool_call_display", DEFAULT_CONFIG["tool_call_display"])
    if mode == "auto":
        return "fullscreen" if heads_up else "print"
    return str(mode)


def _print_tool_card(renderable, config: dict) -> None:
    """Print a tool card with the spacing required by its display mode."""
    console.print(renderable)
    if config.get(
        "tool_call_display",
        DEFAULT_CONFIG["tool_call_display"],
    ) == "print":
        console.print()


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


@dataclass(frozen=True)
class AgentRunResult:
    cancelled: bool = False
    prompt: str | None = None
    error: str | None = None


def response_wait_label(has_reasoning: bool) -> str:
    """Return the live wait label for the response stream."""
    return "Thinking" if has_reasoning else "Waiting"


_TOOL_ACTIVITY_LABELS = {
    "run_command": ("Writing command", "Wrote command"),
    "spawn": ("Planning parallel tasks", "Planned parallel tasks"),
    "read": ("Selecting content", "Selected content"),
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


def _start_response_wait_indicator(interactive: bool, start_time: float) -> tuple[ResponseWaitIndicator | None, Live | None]:
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
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    newline = tail.find("\n")
    if newline >= 0:
        tail = tail[newline + 1:]
    return f"_Earlier streaming content hidden; full reply prints when done._\n\n{tail}"


class TailMarkdown:
    """Renders Markdown but keeps only the last `max_lines` rendered rows.

    Live can't move the cursor above the top of the terminal viewport, so if
    the rendered content ever exceeds the visible height the redraw lands at
    row 0 and the prior frame stays in scrollback — producing duplicates. By
    pre-cropping to the viewport from the top we guarantee the live region
    never overflows, while still showing the most recent (streaming) tail.
    """

    def __init__(self, text: str, max_lines: int, max_source_chars: int = 8000):
        self._text = _markdown_tail_source(text, max_source_chars)
        self._max_lines = max(1, max_lines)

    def __rich_console__(self, console, options):
        md = Markdown(self._text)
        lines = console.render_lines(md, options, pad=False)
        hidden = max(0, len(lines) - self._max_lines)
        # Always emit exactly one top row (hint or blank spacer) so the live
        # block has a fixed height from the first token to the last — no jump
        # when the hint crosses the overflow threshold.
        if hidden:
            lines = lines[-(self._max_lines - 1):] if self._max_lines > 1 else []
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
        max_lines: int,
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
    """Return the completed wait-status text for the first visible response."""
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


def _print_agent_usage_if_enabled(config: dict, usage_path, session_id: str | None) -> None:
    if not config.get("print_usage_after_agent", DEFAULT_CONFIG["print_usage_after_agent"]):
        return
    usage_line = _format_agent_usage_line(load_usage(usage_path, session_id, warn=False))
    if usage_line is not None:
        console.print(usage_line)


def to_response_input_item(item: dict) -> dict | None:
    """Convert one stored history item to a Responses API input item."""
    role = item.get("role")
    typ = item.get("type")
    try:
        if role == "user":
            return {"role": "user", "content": str(item.get("content", ""))}
        if role == "assistant":
            return {"role": "assistant", "content": str(item.get("content") or "")}
        if typ == "reasoning" and "id" in item:
            result = {
                "type": "reasoning",
                "id": responses_input_id(str(item["id"]), "rs"),
                "summary": item.get("summary", []),
            }
            if item.get("provider_content"):
                result["provider_content"] = item["provider_content"]
            return result
        if typ == "function_call":
            result = {
                "type": "function_call",
                "id": responses_input_id(str(item["id"]), "fc"),
                "call_id": item["call_id"],
                "name": item["name"],
                "arguments": item["arguments"],
            }
            if item.get("provider_content"):
                result["provider_content"] = item["provider_content"]
            return result
        if typ == "function_call_output":
            return {
                "type": "function_call_output",
                "call_id": item["call_id"],
                "output": item["output"],
            }
    except KeyError:
        return None
    return None


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


def _dispatch_run_command_with_ui(
    args: dict,
    config: dict,
    history: list | None = None,
    usage_path=None,
    session_id: str | None = None,
    cancellation_token: CancellationToken | None = None,
    retained_store: RetainedOutputStore | None = None,
) -> str:
    cmd = args.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        msg = "[tool argument error: command must be a non-empty string]"
        console.print(f"[red]{msg}[/red]")
        return msg

    try:
        head_chars, tail_chars = resolve_command_output_window(
            (
                COMMAND_OUTPUT_UNSET
                if args.get("head_chars", COMMAND_OUTPUT_UNSET) is None
                else args.get("head_chars", COMMAND_OUTPUT_UNSET)
            ),
            (
                COMMAND_OUTPUT_UNSET
                if args.get("tail_chars", COMMAND_OUTPUT_UNSET) is None
                else args.get("tail_chars", COMMAND_OUTPUT_UNSET)
            ),
            config.get("max_tool_output_chars", DEFAULT_CONFIG["max_tool_output_chars"]),
        )
    except ValueError as e:
        msg = f"[tool argument error: {e}]"
        console.print(f"[red]{msg}[/red]")
        return msg

    safety_level = config.get("command_safety", "risky")
    audit = config.get("audit", True)
    allowed, denial = check_command(
        cmd,
        safety_level,
        audit=audit,
        config=config,
        history=history,
        usage_path=usage_path,
        session_id=session_id,
        cancellation_token=cancellation_token,
    )
    if not allowed:
        console.print(f"[dim]{denial}[/dim]")
        return denial

    display_mode = config.get(
        "tool_call_display",
        DEFAULT_CONFIG["tool_call_display"],
    )
    metadata = f"model window {head_chars:,} / {tail_chars:,} chars"
    running_card = RunningCommandCard(
        cmd,
        metadata,
        display_mode,
        time.perf_counter(),
    )
    live = Live(
        running_card,
        refresh_per_second=4,
        console=console,
        auto_refresh=True,
        transient=False,
    )
    with track_live_display(), live:
        try:
            result = execute_command(
                cmd,
                config.get("command_timeout", 60),
                cancellation_token=cancellation_token,
            )
        except (KeyboardInterrupt, TurnCancelled):
            cancelled_line = Text("> ", style="bold yellow")
            cancelled_line.append(cmd)
            live.update(
                tool_card(
                    "run_command",
                    Group(cancelled_line, Text("Cancelled", style="bold red")),
                    metadata=metadata,
                    display_mode=display_mode,
                    status="cancelled",
                    status_style="red",
                ),
                refresh=True,
            )
            raise

        command_line = Text("> ", style="bold yellow")
        command_line.append(cmd)
        body_parts = [command_line, command_result_renderable(result)]
        output, output_id = retain_command_output(
            result.full_model_output(),
            head_chars,
            tail_chars,
            retained_store,
            int(
                config.get(
                    "max_tool_output_chars",
                    DEFAULT_CONFIG["max_tool_output_chars"],
                )
            ),
        )
        if output_id is not None:
            retained_line = Text("Retained command output: ", style="dim")
            retained_line.append(output_id, style="cyan")
            body_parts.append(retained_line)
        live.update(
            tool_card(
                "run_command",
                Group(*body_parts),
                metadata=metadata,
                display_mode=display_mode,
                status=(
                    "done"
                    if not result.timed_out and result.exit_code in (None, 0)
                    else "failed"
                ),
                status_style=(
                    "green"
                    if not result.timed_out and result.exit_code in (None, 0)
                    else "red"
                ),
            ),
            refresh=True,
        )
    if display_mode == "print":
        console.print()
    return output


def _dispatch_ask_user(args: dict, config: dict | None = None) -> str:
    question = args.get("question")
    if not isinstance(question, str) or not question.strip():
        msg = "[tool argument error: question must be a non-empty string]"
        console.print(f"[red]{msg}[/red]")
        return msg
    if not sys.stdin.isatty():
        return "[non-interactive session; user unavailable]"
    config = config or DEFAULT_CONFIG
    display_mode = config.get(
        "tool_call_display",
        DEFAULT_CONFIG["tool_call_display"],
    )
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
    if display_mode == "print":
        console.print()
    return answer


def _dispatch_spawn_with_ui(
    args: dict,
    root_node,
    store,
    client,
    config,
    cancellation_token: CancellationToken | None = None,
    retained_store: RetainedOutputStore | None = None,
) -> str:
    children_raw = args.get("children")
    if not isinstance(children_raw, list) or not children_raw:
        msg = "[tool argument error: children must be a non-empty list]"
        console.print(f"[red]{msg}[/red]")
        return msg

    top_labels = [c.get("label", "?") for c in children_raw if isinstance(c, dict)]
    # state per label: {status, depth, tldr?, reason?}
    states: dict[str, dict] = {}
    # parent_label -> ordered child labels. Top-level entries live under root_node.label.
    children_of: dict[str, list[str]] = {root_node.label: list(top_labels)}
    for lbl in top_labels:
        states[lbl] = {"status": "running", "depth": 0}
    lock = threading.Lock()

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

        def on_child_done(self, parent_label: str, label: str, result: dict) -> None:
            with lock:
                existing = states.get(label, {"depth": 0})
                states[label] = {**existing, **result}

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
                display_mode=config.get(
                    "tool_call_display",
                    DEFAULT_CONFIG["tool_call_display"],
                ),
            )

    with track_live_display(), Live(
        SpawnPanel(),
        refresh_per_second=10,
        console=console,
        auto_refresh=True,
        transient=False,
        vertical_overflow="visible",
    ) as live:
        try:
            results = spawn_batch(
                root_node,
                children_raw,
                store,
                client,
                config,
                observer=observer,
                usage_path=root_node.usage_path,
                session_id=root_node.session_id,
                cancellation_token=cancellation_token,
                retained_store=retained_store,
            )
        except (KeyboardInterrupt, TurnCancelled):
            with lock:
                for state in states.values():
                    if state["status"] == "running":
                        state["status"] = "cancelled"
                        state["reason"] = "cancelled"
            live.update(SpawnPanel())
            raise
        except DepthExceeded as e:
            output = f"[error: {e}]"
            live.update(Text(output, style="red"))
            return output
        except ValueError as e:
            output = f"[tool argument error: {e}]"
            console.print(f"[red]{output}[/red]")
            return output
        live.update(SpawnPanel())
        output = json.dumps(results)

    if config.get(
        "tool_call_display",
        DEFAULT_CONFIG["tool_call_display"],
    ) == "print":
        console.print()
    return output


def run_agent(
    query: str,
    config: dict,
    client=None,
    propagate_keyboard_interrupt: bool = False,
    new_session: bool = False,
    incognito: bool = False,
    heads_up: bool = False,
) -> AgentRunResult:
    config = dict(config)
    config["tool_call_display"] = resolve_tool_call_display(
        config,
        heads_up=heads_up,
    )
    interactive = sys.stdout.isatty()
    reply_text = ""
    tool_calls = []
    history: list = []
    metadata: dict = {}
    session_context = None
    artifact_file = None
    artifact_store = None
    reads_file = None
    retained_store = None
    usage_path = None
    wait_indicator: ResponseWaitIndicator | None = None
    tool_indicator: ToolActivityIndicator | None = None
    spinner_live: Live | None = None
    stream_live: Live | None = None
    reasoning_items = []
    active_tool_call = None
    cancellation_token = CancellationToken()
    thought_started = time.perf_counter()
    wait_indicator, spinner_live = _start_response_wait_indicator(interactive, thought_started)

    def _refresh_wait_indicator() -> None:
        if wait_indicator is not None and spinner_live is not None:
            spinner_live.update(wait_indicator, refresh=True)

    def _refresh_tool_indicator() -> None:
        if tool_indicator is not None and spinner_live is not None:
            spinner_live.update(tool_indicator, refresh=True)

    def _stop_live_displays() -> None:
        nonlocal spinner_live, stream_live
        if spinner_live is not None:
            spinner_live.stop()
            spinner_live = None
        if stream_live is not None:
            stream_live.stop()
            stream_live = None

    def _save_turn_state() -> None:
        if incognito:
            return
        if session_context is not None:
            redo_path = redo_file_for(session_context.history_file)
            if redo_path.exists():
                redo_path.unlink()
            save_history(history, session_context.history_file)
        if artifact_store is not None and artifact_file is not None:
            save_artifact_store(artifact_store, artifact_file)
        if retained_store is not None and reads_file is not None:
            save_retained_output_store(retained_store, reads_file)

    def _checkpoint_cancelled_turn() -> None:
        recorded_reasoning_ids = {
            str(item.get("id"))
            for item in history
            if isinstance(item, dict) and item.get("type") == "reasoning"
        }
        for item in reasoning_items:
            if str(item.id) not in recorded_reasoning_ids:
                stored_reasoning = {
                    "type": "reasoning",
                    "id": item.id,
                    "summary": [],
                    **metadata,
                }
                if item.provider_content:
                    stored_reasoning["provider_content"] = item.provider_content
                history.append(stored_reasoning)

        if reply_text:
            history.append({"role": "assistant", "content": reply_text, **metadata})

        recorded_call_ids = {
            str(item.get("call_id"))
            for item in history
            if isinstance(item, dict) and item.get("type") == "function_call"
        }
        active_call_id = (
            str(active_tool_call.call_id)
            if active_tool_call is not None else None
        )
        for item in tool_calls:
            if str(item.call_id) in recorded_call_ids:
                continue
            if str(item.call_id) == active_call_id:
                output = "[cancelled by user; execution may have made partial changes]"
            else:
                output = "[cancelled by user before execution]"
            stored_call = {
                "type": "function_call",
                "id": item.id,
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
                **metadata,
            }
            if item.provider_content:
                stored_call["provider_content"] = item.provider_content
            history.extend([
                stored_call,
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": output,
                    **metadata,
                },
            ])

        history.append({
            "role": "assistant",
            "content": "[Turn cancelled by user.]",
            **metadata,
        })
        _save_turn_state()

    sigint_cancel_scope = cancel_token_on_sigint(cancellation_token)
    try:
        sigint_cancel_scope.__enter__()
        if client is None:
            client = create_client(config)

        if new_session:
            forget_current_session()
        session_context = prepare_session_context(
            mark_message=not incognito,
            persist_metadata=not incognito,
        )
        history = [] if (new_session or incognito) else load_history(session_context.history_file)
        web_search_read_nudge_sent = history_has_web_search_read_nudge(history)
        metadata = history_metadata(session_context)

        artifact_file = artifact_file_for(session_context.history_file)
        artifact_store = load_artifact_store(artifact_file)
        reads_file = reads_file_for(session_context.history_file)
        retained_store = load_retained_output_store(reads_file)
        usage_path = usage_file_for(session_context.history_file)
        root_node = AgentNode(
            label="root",
            depth=0,
            parent_label=None,
            task=query,
            sterile=False,
            visible_labels=artifact_store.all_labels(),
            usage_path=usage_path,
            session_id=session_context.session_id,
            incognito=incognito,
        )

        history.append({"role": "user", "content": query, **metadata})

        instructions = (
            config["system_prompt"]
            + f"\n\nSystem info:\n{get_system_info()}"
        )
        tools = build_agent_tools(config)
        input_items = build_input(
            history,
            model=config["model"],
            config=config,
            instructions=instructions,
            tools=tools,
        )

        kwargs = dict(
            model=config["model"],
            instructions=instructions,
            tools=tools,
            input=input_items,
        )
        effort = config.get("reasoning_effort")
        if effort:
            kwargs["reasoning"] = {"effort": effort}

        while True:
            reply_text = ""
            tool_calls = []
            reasoning_items = []
            saw_reasoning = False
            got_text = False
            started_tool_positions: dict[str, int] = {}
            started_tool_names: list[str] = []
            tool_started_at: float | None = None
            tool_completed_at: float | None = None
            response_phase_completed = False
            tool_phase_completed = False
            stream_preview: StreamingMarkdownPreview | None = None

            def complete_response_phase() -> None:
                nonlocal spinner_live, wait_indicator, response_phase_completed
                if response_phase_completed:
                    return
                if spinner_live is not None:
                    spinner_live.stop()
                    spinner_live = None
                wait_indicator = None
                response_phase_completed = True
                if interactive:
                    console.print(
                        thought_complete_indicator(
                            response_start_status(
                                time.perf_counter() - thought_started,
                                has_reasoning=saw_reasoning,
                            )
                        )
                    )

            def complete_tool_phase() -> None:
                nonlocal spinner_live, tool_phase_completed
                if tool_started_at is None or tool_phase_completed:
                    return
                if spinner_live is not None:
                    spinner_live.stop()
                    spinner_live = None
                tool_phase_completed = True
                if interactive:
                    console.print(
                        tool_complete_indicator(
                            tool_activity_complete_status(
                                (tool_completed_at or time.perf_counter())
                                - tool_started_at,
                                tuple(started_tool_names),
                            )
                        )
                    )

            def note_tool_call_started(
                item_id: str,
                call_id: str,
                name: str,
            ) -> None:
                nonlocal spinner_live, stream_live, tool_indicator, tool_started_at
                keys = {value for value in (item_id, call_id) if value}
                positions = {
                    started_tool_positions[key]
                    for key in keys
                    if key in started_tool_positions
                }
                if positions:
                    position = min(positions)
                    for key in keys:
                        started_tool_positions[key] = position
                    if name and not started_tool_names[position]:
                        started_tool_names[position] = name
                        if tool_indicator is not None:
                            tool_indicator.start_tool_call(str(position), name)
                            _refresh_tool_indicator()
                    return

                if tool_started_at is None:
                    complete_response_phase()
                    if stream_preview is not None:
                        stream_preview.flush(refresh=False)
                    if stream_live is not None:
                        stream_live.stop()
                        stream_live = None
                    tool_started_at = time.perf_counter()
                    if interactive:
                        tool_indicator = ToolActivityIndicator(tool_started_at)
                        spinner_live = Live(
                            tool_indicator,
                            refresh_per_second=4,
                            console=console,
                            auto_refresh=True,
                            transient=True,
                        )

                if not keys:
                    keys = {f"tool_{len(started_tool_names)}"}
                position = len(started_tool_names)
                for key in keys:
                    started_tool_positions[key] = position
                started_tool_names.append(name)
                if tool_indicator is not None:
                    tool_indicator.start_tool_call(str(position), name)
                    if spinner_live is not None:
                        spinner_live.start()
                    _refresh_tool_indicator()

            if spinner_live is None:
                thought_started = time.perf_counter()
                wait_indicator, spinner_live = _start_response_wait_indicator(interactive, thought_started)

            _ctx_breakdown = estimate_context_breakdown(
                config["model"],
                kwargs.get("instructions", ""),
                kwargs.get("tools", []),
                kwargs.get("input", []),
            )

            stream_replays = 0
            while True:
                retry_stream = False
                try:
                    final_response = None
                    for event in stream_response(
                        client, config,
                        kwargs["model"], kwargs["instructions"],
                        kwargs["tools"], kwargs["input"],
                        reasoning=kwargs.get("reasoning"),
                        prompt_cache_key=f"jarv:{session_context.session_id}",
                        cancellation_token=cancellation_token,
                    ):
                        if isinstance(event, TextDelta):
                            if not got_text:
                                got_text = True
                                complete_tool_phase()
                                complete_response_phase()
                                if interactive:
                                    _, term_h = terminal_size(console=console)
                                    stream_max_lines = term_h - 2
                                    stream_live = InPlaceLive(
                                        TailMarkdown("", stream_max_lines),
                                        console=console,
                                        auto_refresh=False,
                                        transient=True,
                                        vertical_overflow="crop",
                                    )
                                    stream_live.start()
                                    stream_preview = StreamingMarkdownPreview(
                                        stream_live,
                                        stream_max_lines,
                                    )
                            if stream_preview is not None:
                                stream_preview.append(event.delta)
                            else:
                                reply_text += event.delta
                        elif isinstance(event, ToolCallStarted):
                            note_tool_call_started(
                                event.id,
                                event.call_id,
                                event.name,
                            )
                        elif isinstance(event, ToolCallDone):
                            note_tool_call_started(
                                event.id,
                                event.call_id,
                                event.name,
                            )
                            tool_completed_at = time.perf_counter()
                            tool_calls.append(event)
                        elif isinstance(event, ReasoningStarted):
                            saw_reasoning = True
                            if wait_indicator is not None:
                                wait_indicator.has_reasoning = True
                                _refresh_wait_indicator()
                        elif isinstance(event, ReasoningDone):
                            saw_reasoning = True
                            reasoning_items.append(event)
                            if wait_indicator is not None:
                                wait_indicator.has_reasoning = True
                                _refresh_wait_indicator()
                        elif isinstance(event, StreamDone):
                            final_response = event.response
                    if stream_preview is not None:
                        reply_text = stream_preview.text
                    final_text = response_output_text(final_response)
                    if final_text and len(final_text) >= len(reply_text):
                        reply_text = final_text
                        got_text = True
                        if stream_preview is not None:
                            stream_preview.replace(final_text)
                    if not incognito:
                        record_response_usage(
                            usage_path,
                            session_context.session_id,
                            config["model"],
                            final_response,
                            "root",
                            provider=str(config.get("provider") or "openai"),
                            requested_service_tier=configured_service_tier(config),
                            context_breakdown=_ctx_breakdown,
                            output_text=reply_text or "\n".join(
                                f"{item.name} {item.arguments}" for item in tool_calls
                            ),
                        )
                except RetryableStreamError:
                    retry_stream = stream_replays == 0
                    reply_text = ""
                    tool_calls = []
                    reasoning_items = []
                    saw_reasoning = False
                    got_text = False
                    started_tool_positions = {}
                    started_tool_names = []
                    tool_started_at = None
                    tool_completed_at = None
                    response_phase_completed = False
                    tool_phase_completed = False
                    stream_preview = None
                    tool_indicator = None
                    wait_indicator = None
                    if not retry_stream:
                        raise
                    stream_replays += 1
                finally:
                    if stream_preview is not None:
                        reply_text = stream_preview.text
                        stream_preview.flush(refresh=False)
                    if spinner_live is not None:
                        spinner_live.stop()
                        spinner_live = None
                    if stream_live is not None:
                        stream_live.stop()
                        stream_live = None
                if not retry_stream:
                    break
                thought_started = time.perf_counter()
                wait_indicator, spinner_live = _start_response_wait_indicator(
                    interactive,
                    thought_started,
                )
            complete_tool_phase()
            complete_response_phase()
            if got_text:
                if interactive:
                    console.print(Markdown(flatten_headings(reply_text)))
                else:
                    print(reply_text)

            if tool_calls:
                if config.get(
                    "tool_call_display",
                    DEFAULT_CONFIG["tool_call_display"],
                ) == "print":
                    console.print()
                new_input_items = []
                for ri in reasoning_items:
                    rd = {"type": "reasoning", "id": ri.id, "summary": [], **metadata}
                    if ri.provider_content:
                        rd["provider_content"] = ri.provider_content
                    history.append(rd)
                    api_item = to_response_input_item(rd)
                    if api_item is not None:
                        new_input_items.append(api_item)
                def append_tool_result(item, output: ToolOutput) -> None:
                    nonlocal active_tool_call
                    fc = {
                        "type": "function_call",
                        "id": item.id,
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                        **metadata,
                    }
                    if item.provider_content:
                        fc["provider_content"] = item.provider_content
                    fco = {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": output,
                        **metadata,
                    }
                    history.extend([fc, fco])
                    active_tool_call = None
                    for stored_item in (fc, fco):
                        api_item = to_response_input_item(stored_item)
                        if api_item is not None:
                            new_input_items.append(api_item)

                item_index = 0
                while item_index < len(tool_calls):
                    item = tool_calls[item_index]
                    if tool_call_is_parallel_safe(item.name):
                        group_end = item_index
                        while (
                            group_end < len(tool_calls)
                            and tool_call_is_parallel_safe(tool_calls[group_end].name)
                        ):
                            group_end += 1
                        group = tool_calls[item_index:group_end]
                        active_tool_call = group[0]
                        results = dispatch_parallel_safe_tool_batch(
                            group,
                            node=root_node,
                            store=artifact_store,
                            client=client,
                            config=config,
                            cancellation_token=cancellation_token,
                            retained_store=retained_store,
                        )
                        for safe_item, result in zip(group, results):
                            output = result.output
                            if safe_item.name != "read":
                                output = truncate_model_output(
                                    output,
                                    config.get(
                                        "max_tool_output_chars",
                                        DEFAULT_CONFIG["max_tool_output_chars"],
                                    ),
                                )
                            if safe_item.name == "read" and result.args is not None:
                                read_path = Text(
                                    str(result.args.get("input", "")),
                                    no_wrap=True,
                                    overflow="ellipsis",
                                )
                                read_meta = Text(
                                    f"offset {result.args.get('offset', 0)!r}  \u2022  "
                                    f"size {result.args.get('size', 'default')!r}",
                                    style="dim",
                                )
                                _print_tool_card(
                                    tool_card(
                                        "read",
                                        Group(read_path, read_meta),
                                        display_mode=config.get(
                                            "tool_call_display",
                                            DEFAULT_CONFIG["tool_call_display"],
                                        ),
                                    ),
                                    config,
                                )
                            elif (
                                safe_item.name == "web_search"
                                and result.args is not None
                            ):
                                query = Text(str(result.args.get("query", "")))
                                _print_tool_card(
                                    tool_card(
                                        "web_search",
                                        query,
                                        metadata="DuckDuckGo",
                                        display_mode=config.get(
                                            "tool_call_display",
                                            DEFAULT_CONFIG["tool_call_display"],
                                        ),
                                    ),
                                    config,
                                )
                                if not web_search_read_nudge_sent:
                                    output = append_web_search_read_nudge(output)
                                    web_search_read_nudge_sent = True
                            append_tool_result(safe_item, output)
                        active_tool_call = None
                        item_index = group_end
                        continue

                    if not tool_enabled(config, item.name):
                        append_tool_result(
                            item,
                            f"[tool disabled: {item.name}]",
                        )
                        item_index += 1
                        continue
                    active_tool_call = item
                    try:
                        args = json.loads(item.arguments or "{}")
                    except json.JSONDecodeError as e:
                        output = f"[tool argument error: invalid JSON: {e}]"
                        console.print(f"[red]{output}[/red]")
                    else:
                        if item.name == "run_command":
                            output = _dispatch_run_command_with_ui(
                                args,
                                config,
                                history,
                                usage_path=usage_path,
                                session_id=session_context.session_id,
                                cancellation_token=cancellation_token,
                                retained_store=retained_store,
                            )
                        elif item.name == "spawn":
                            output = _dispatch_spawn_with_ui(
                                args,
                                root_node,
                                artifact_store,
                                client,
                                config,
                                cancellation_token=cancellation_token,
                                retained_store=retained_store,
                            )
                        elif item.name == "ask_user":
                            output = _dispatch_ask_user(args, config)
                        else:
                            output = f"[unknown tool: {item.name}]"
                            console.print(f"[red]{output}[/red]")

                    if item.name not in {"run_command", "read"}:
                        output = truncate_model_output(
                            output,
                            config.get(
                                "max_tool_output_chars",
                                DEFAULT_CONFIG["max_tool_output_chars"],
                            ),
                        )
                    append_tool_result(item, output)
                    item_index += 1
                kwargs["input"] = trim_turn_input(
                    kwargs["input"] + new_input_items,
                    model=config["model"],
                    config=config,
                    instructions=kwargs["instructions"],
                    tools=kwargs["tools"],
                )
            else:
                history.append({"role": "assistant", "content": reply_text, **metadata})
                _save_turn_state()
                _print_agent_usage_if_enabled(config, usage_path, session_context.session_id)
                break
        return AgentRunResult()
    except (KeyboardInterrupt, TurnCancelled):
        cancellation_token.cancel()
        if session_context is not None:
            _checkpoint_cancelled_turn()
        return AgentRunResult(cancelled=True, prompt=query)
    except ProviderError as e:
        console.print(f"[red]API error:[/red] {escape(str(e))}")
        if reply_text and not tool_calls:
            history.append({"role": "assistant", "content": reply_text, **metadata})
        if not incognito and session_context is not None:
            save_history(history, session_context.history_file)
        if not incognito and artifact_store is not None and artifact_file is not None:
            save_artifact_store(artifact_store, artifact_file)
        if not incognito and retained_store is not None and reads_file is not None:
            save_retained_output_store(retained_store, reads_file)
        return AgentRunResult(error=str(e))
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        if not incognito and session_context is not None:
            save_history(history, session_context.history_file)
        if not incognito and artifact_store is not None and artifact_file is not None:
            save_artifact_store(artifact_store, artifact_file)
        if not incognito and retained_store is not None and reads_file is not None:
            save_retained_output_store(retained_store, reads_file)
        return AgentRunResult(error=str(e))
    finally:
        sigint_cancel_scope.__exit__(None, None, None)
        _stop_live_displays()



