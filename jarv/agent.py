import json
import os
import platform
import time

from openai import OpenAI, OpenAIError
from rich import box
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from .config import DEFAULT_CONFIG
from .display import console, flatten_headings
from .history import (
    SessionContext,
    artifact_file_for,
    get_shell_name,
    history_metadata,
    load_history,
    prepare_session_context,
    save_history,
)
from .artifacts import ArtifactStore, load_artifact_store, save_artifact_store
from .orchestrator import (
    READ_ARTIFACT_TOOL,
    RUN_COMMAND_TOOL,
    SPAWN_TOOL,
    AgentNode,
    dispatch_tool,
)
from .shell import display_command_result, execute_command

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


def thought_complete_indicator(text: str) -> Text:
    """Return the static completed-thinking bubble."""
    return Text(f"\u2726 {text}", style="dim")


def safe_flush_index(text: str) -> int:
    """Return the largest index `i` such that `text[:i]` ends in a paragraph
    break and contains a balanced number of ``` fences. Content up to `i`
    can be committed to scrollback as standalone Markdown without breaking
    a code block mid-render."""
    fence_count = 0
    last_safe = 0
    i = 0
    n = len(text)
    while i < n - 1:
        if text.startswith("```", i):
            fence_count += 1
            i += 3
            continue
        if fence_count % 2 == 0 and text[i] == "\n" and text[i + 1] == "\n":
            last_safe = i + 2
            i += 2
            continue
        i += 1
    return last_safe


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


def new_terminal_context_input(context: SessionContext) -> dict | None:
    if context.scope != "global" or not context.previous_global_session_changed:
        return None
    return {
        "role": "user",
        "content": "\n".join(
            [
                "<new_terminal>",
                f"Terminal session: {context.session_label}",
                "</new_terminal>",
            ]
        ),
    }


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
    children = args.get("children") or []
    labels = [c.get("label", "?") for c in children if isinstance(c, dict)]
    console.print()
    console.print(Rule(f"[bold magenta]spawn → {', '.join(labels) or '(none)'}[/bold magenta]", style="magenta", align="left"))
    output = dispatch_tool("spawn", args, root_node, store, client, config)
    try:
        parsed = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        console.print(f"[red]{output}[/red]")
    else:
        if isinstance(parsed, list):
            for entry in parsed:
                lbl = entry.get("label", "?")
                status = entry.get("status", "?")
                if status == "done":
                    tldr = entry.get("tldr", "")
                    console.print(f"[green]✓[/green] [bold]{escape(lbl)}[/bold]: {escape(tldr)}")
                else:
                    reason = entry.get("reason", "")
                    console.print(f"[red]✗[/red] [bold]{escape(lbl)}[/bold]: {escape(reason)}")
        else:
            console.print(f"[yellow]{escape(output)}[/yellow]")
    console.print(Rule(style="bright_black"))
    return output


def run_agent(
    query: str,
    config: dict,
    client: OpenAI,
    session_override: tuple[str, str, str] | None = None,
    independent: bool = False,
    propagate_keyboard_interrupt: bool = False,
) -> None:
    session_context = prepare_session_context(
        config,
        independent=independent,
        session_override=session_override,
        mark_message=True,
    )
    history = load_history(session_context.history_file)
    max_history = config.get("max_history", DEFAULT_CONFIG["max_history"])
    metadata = history_metadata(session_context)

    artifact_file = artifact_file_for(session_context.history_file)
    artifact_store = load_artifact_store(artifact_file)
    root_node = AgentNode(
        label="root",
        depth=0,
        parent_label=None,
        task=query,
        sterile=False,
        visible_labels=artifact_store.all_labels(),
    )

    history.append({"role": "user", "content": query, **metadata})

    input_items = build_input(history, max_history)
    terminal_context = new_terminal_context_input(session_context)
    if terminal_context is not None and input_items:
        input_items.insert(len(input_items) - 1, terminal_context)

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

            # Spinner runs at a low refresh rate to reduce Windows focus
            # annoyances; once text starts streaming we swap to a faster
            # Live that progressively renders the Markdown reply.
            thought_started = time.perf_counter()
            spinner_live = Live(
                ThinkingIndicator(thought_started),
                refresh_per_second=4,
                console=console,
                auto_refresh=True,
                transient=True,
            )
            spinner_live.start()
            stream_live: Live | None = None
            flushed_to = 0
            try:
                with client.responses.stream(**kwargs) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            if not got_text:
                                got_text = True
                                spinner_live.stop()
                                spinner_live = None
                                thought_elapsed = time.perf_counter() - thought_started
                                console.print(
                                    thought_complete_indicator(
                                        f"Thought for {format_thought_duration(thought_elapsed)}."
                                    )
                                )
                                stream_live = Live(
                                    Markdown(""),
                                    refresh_per_second=12,
                                    console=console,
                                    auto_refresh=True,
                                    transient=False,
                                    vertical_overflow="visible",
                                )
                                stream_live.start()
                            reply_text += event.delta
                            # Commit settled paragraphs above the Live region so it
                            # only has to redraw the trailing in-progress chunk.
                            new_safe = safe_flush_index(reply_text)
                            if new_safe > flushed_to:
                                chunk = reply_text[flushed_to:new_safe]
                                console.print(Markdown(flatten_headings(chunk)))
                                flushed_to = new_safe
                            if stream_live is not None:
                                stream_live.update(
                                    Markdown(flatten_headings(reply_text[flushed_to:]))
                                )
                        elif event.type == "response.output_item.done":
                            if event.item.type == "function_call":
                                tool_calls.append(event.item)
                            elif event.item.type == "reasoning":
                                reasoning_items.append(event.item)
            finally:
                if spinner_live is not None:
                    spinner_live.stop()
                if stream_live is not None:
                    stream_live.stop()
            if not got_text:
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



