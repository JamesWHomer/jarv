import json
import os
import platform
import sys
import threading
import time

from openai import OpenAI, OpenAIError
from rich import box
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.segment import Segment
from rich.text import Text

from .config import DEFAULT_CONFIG
from .display import console, flatten_headings
from .history import (
    artifact_file_for,
    get_shell_name,
    history_metadata,
    load_history,
    prepare_session_context,
    redo_file_for,
    save_history,
)
from .artifacts import ArtifactStore, load_artifact_store, save_artifact_store
from .orchestrator import (
    READ_ARTIFACT_TOOL,
    RUN_COMMAND_TOOL,
    SPAWN_TOOL,
    AgentNode,
    DepthExceeded,
    SpawnObserver,
    dispatch_tool,
    spawn_batch,
)
from .shell import display_command_result, execute_command
from .usage import record_response_usage, usage_file_for

# Responses API tool format (flat, no "function" wrapper key)
TOOLS = [RUN_COMMAND_TOOL, SPAWN_TOOL, READ_ARTIFACT_TOOL]


_THINKING_FRAMES = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]


class ThinkingIndicator:
    """Animated thinking bubble with live elapsed timer; re-renders on each Live refresh."""

    def __init__(self, start_time: float):
        self._start = start_time

    def __rich_console__(self, console, options):
        now = time.perf_counter()
        elapsed = now - self._start
        frame = _THINKING_FRAMES[int(now * 10) % len(_THINKING_FRAMES)]
        yield Text(f"{frame}  Thinking\u2026  {int(elapsed)}s")


class TailMarkdown:
    """Renders Markdown but keeps only the last `max_lines` rendered rows.

    Live can't move the cursor above the top of the terminal viewport, so if
    the rendered content ever exceeds the visible height the redraw lands at
    row 0 and the prior frame stays in scrollback — producing duplicates. By
    pre-cropping to the viewport from the top we guarantee the live region
    never overflows, while still showing the most recent (streaming) tail.
    """

    def __init__(self, text: str, max_lines: int):
        self._text = text
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


def thought_complete_indicator(text: str) -> Text:
    """Return the static completed-thinking bubble."""
    return Text(f"\u2726 {text}", style="dim")



def format_thought_duration(seconds: float) -> str:
    """Return a compact human-readable duration for the thinking timer."""
    rounded = round(max(0.0, seconds), 1)
    unit = "second" if rounded == 1 else "seconds"
    return f"{rounded:.1f} {unit}"


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
            return {"type": "reasoning", "id": item["id"], "summary": item.get("summary", [])}
        if typ == "function_call":
            return {
                "type": "function_call",
                "id": item["id"],
                "call_id": item["call_id"],
                "name": item["name"],
                "arguments": item["arguments"],
            }
        if typ == "function_call_output":
            return {
                "type": "function_call_output",
                "call_id": item["call_id"],
                "output": item["output"],
            }
    except KeyError:
        return None
    return None


def build_input(history: list, max_history: int) -> list:
    """Convert stored history to Responses API input format."""
    slice_ = history[-max_history:]
    # Drop leading non-user items to avoid orphaned tool call pairs after truncation.
    for i, m in enumerate(slice_):
        if isinstance(m, dict) and m.get("role") == "user":
            slice_ = slice_[i:]
            break
    else:
        slice_ = []
    items = []
    for m in slice_:
        if not isinstance(m, dict):
            continue
        api_item = to_response_input_item(m)
        if api_item is not None:
            items.append(api_item)
    return items


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


def _dispatch_run_command_with_ui(args: dict, config: dict) -> str:
    cmd = args.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        msg = "[tool argument error: command must be a non-empty string]"
        console.print(f"[red]{msg}[/red]")
        return msg
    console.print()
    console.print(Rule(f"[bold yellow]$ {escape(cmd)}[/bold yellow]", style="yellow", align="left"))
    # Avoid a constantly repainting spinner while a child process is running;
    # on Windows this can cause focus annoyances in heads-up mode.
    console.print("[dim]Running command...[/dim]")
    result = execute_command(cmd, config.get("command_timeout", 60))
    display_command_result(result)
    console.print(Rule(style="bright_black"))
    return result.to_model_output()


def _dispatch_spawn_with_ui(args: dict, root_node, store, client, config) -> str:
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
            yield Panel(
                Group(*lines),
                title=f"[bold magenta]spawn[/bold magenta] [dim]{done}/{total}[/dim]",
                title_align="left",
                border_style="magenta",
                box=box.ROUNDED,
                padding=(0, 1),
            )

    console.print()
    with Live(
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
            )
        except DepthExceeded as e:
            output = f"[error: {e}]"
            live.update(Text(output, style="red"))
            return output
        live.update(SpawnPanel())
        output = json.dumps(results)

    console.print()
    return output


def run_agent(
    query: str,
    config: dict,
    client: OpenAI,
    propagate_keyboard_interrupt: bool = False,
) -> None:
    interactive = sys.stdout.isatty()
    session_context = prepare_session_context(mark_message=True)
    history = load_history(session_context.history_file)
    max_history = config.get("max_history", DEFAULT_CONFIG["max_history"])
    metadata = history_metadata(session_context)

    artifact_file = artifact_file_for(session_context.history_file)
    artifact_store = load_artifact_store(artifact_file)
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
    )

    history.append({"role": "user", "content": query, **metadata})

    redo_path = redo_file_for(session_context.history_file)
    if redo_path.exists():
        redo_path.unlink()

    input_items = build_input(history, max_history)

    kwargs = dict(
        model=config["model"],
        instructions=(
            config["system_prompt"]
            + f"\n\nSystem info:\n{get_system_info()}"
        ),
        tools=TOOLS,
        input=input_items,
    )
    effort = config.get("reasoning_effort")
    if effort:
        kwargs["reasoning"] = {"effort": effort}

    try:
        while True:
            reply_text = ""
            tool_calls = []
            reasoning_items = []
            got_text = False

            thought_started = time.perf_counter()
            spinner_live: Live | None = None
            stream_live: Live | None = None
            if interactive:
                # Spinner runs at a low refresh rate to reduce Windows focus
                # annoyances; once text starts streaming we swap to a faster
                # Live that progressively renders the Markdown reply.
                spinner_live = Live(
                    ThinkingIndicator(thought_started),
                    refresh_per_second=4,
                    console=console,
                    auto_refresh=True,
                    transient=True,
                )
                spinner_live.start()
            try:
                final_response = None
                with client.responses.stream(**kwargs) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            if not got_text:
                                got_text = True
                                if spinner_live is not None:
                                    spinner_live.stop()
                                    spinner_live = None
                                if interactive:
                                    thought_elapsed = time.perf_counter() - thought_started
                                    console.print(
                                        thought_complete_indicator(
                                            f"Thought for {format_thought_duration(thought_elapsed)}."
                                        )
                                    )
                                    stream_max_lines = console.size.height - 2
                                    stream_live = Live(
                                        TailMarkdown("", stream_max_lines),
                                        refresh_per_second=12,
                                        console=console,
                                        auto_refresh=True,
                                        transient=True,
                                        vertical_overflow="crop",
                                    )
                                    stream_live.start()
                            reply_text += event.delta
                            if stream_live is not None:
                                stream_live.update(
                                    TailMarkdown(
                                        flatten_headings(reply_text),
                                        stream_max_lines,
                                    )
                                )
                        elif event.type == "response.output_item.done":
                            if event.item.type == "function_call":
                                tool_calls.append(event.item)
                            elif event.item.type == "reasoning":
                                reasoning_items.append(event.item)
                    try:
                        final_response = stream.get_final_response()
                    except Exception:
                        final_response = None
                record_response_usage(
                    usage_path,
                    session_context.session_id,
                    config["model"],
                    final_response,
                    "root",
                )
            finally:
                if spinner_live is not None:
                    spinner_live.stop()
                if stream_live is not None:
                    stream_live.stop()
            if got_text:
                if interactive:
                    console.print(Markdown(flatten_headings(reply_text)))
                else:
                    print(reply_text)
            if not got_text and interactive:
                thought_elapsed = time.perf_counter() - thought_started
                console.print(
                    thought_complete_indicator(
                        f"Thought for {format_thought_duration(thought_elapsed)}."
                    )
                )

            if tool_calls:
                new_input_items = []
                for ri in reasoning_items:
                    rd = {"type": "reasoning", "id": ri.id, "summary": [], **metadata}
                    history.append(rd)
                    api_item = to_response_input_item(rd)
                    if api_item is not None:
                        new_input_items.append(api_item)
                for item in tool_calls:
                    try:
                        args = json.loads(item.arguments or "{}")
                    except json.JSONDecodeError as e:
                        output = f"[tool argument error: invalid JSON: {e}]"
                        console.print(f"[red]{output}[/red]")
                    else:
                        if item.name == "run_command":
                            output = _dispatch_run_command_with_ui(args, config)
                        elif item.name == "spawn":
                            output = _dispatch_spawn_with_ui(args, root_node, artifact_store, client, config)
                        elif item.name == "read_artifact":
                            output = dispatch_tool(item.name, args, root_node, artifact_store, client, config)
                            console.print(f"[dim]read_artifact({args.get('label')!r})[/dim]")
                        else:
                            output = f"[unknown tool: {item.name}]"
                            console.print(f"[red]{output}[/red]")

                    fc = {
                        "type": "function_call",
                        "id": item.id,
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                        **metadata,
                    }
                    fco = {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": output,
                        **metadata,
                    }
                    history.extend([fc, fco])
                    for stored_item in (fc, fco):
                        api_item = to_response_input_item(stored_item)
                        if api_item is not None:
                            new_input_items.append(api_item)
                kwargs["input"] = kwargs["input"] + new_input_items
            else:
                history.append({"role": "assistant", "content": reply_text, **metadata})
                save_history(history[-max_history:], session_context.history_file)
                save_artifact_store(artifact_store, artifact_file)
                break
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        save_history(history[-max_history:], session_context.history_file)
        save_artifact_store(artifact_store, artifact_file)
        if propagate_keyboard_interrupt:
            raise
    except OpenAIError as e:
        console.print(f"[red]OpenAI API error:[/red] {e}")
        save_history(history[-max_history:], session_context.history_file)
        save_artifact_store(artifact_store, artifact_file)
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        save_history(history[-max_history:], session_context.history_file)
        save_artifact_store(artifact_store, artifact_file)
        raise SystemExit(1)



