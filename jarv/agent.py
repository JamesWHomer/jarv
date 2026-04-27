import json
import os
import platform

from openai import OpenAI, OpenAIError
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.rule import Rule
from rich.spinner import Spinner

from .config import DEFAULT_CONFIG
from .display import console, flatten_headings
from .history import (
    SessionContext,
    get_shell_name,
    history_metadata,
    load_history,
    prepare_session_context,
    save_history,
)
from .shell import display_command_result, execute_command

# Responses API tool format (flat, no "function" wrapper key)
TOOLS = [
    {
        "type": "function",
        "name": "run_command",
        "description": "Run a shell command and return its output. Use this to interact with the filesystem, run scripts, check system info, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                }
            },
            "required": ["command"],
        },
    }
]


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
    parts = [
        f"OS: {platform.system()} {platform.release()}",
        f"CWD: {os.getcwd()}",
    ]
    parts.append(f"Shell: {get_shell_name()}")
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


def run_agent(
    query: str,
    config: dict,
    client: OpenAI,
    session_override: tuple[str, str, str] | None = None,
    independent: bool = False,
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

            with Live(
                Spinner("dots", text=" Thinking..."),
                refresh_per_second=15,
                console=console,
            ) as live:
                with client.responses.stream(**kwargs) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            if not got_text:
                                got_text = True
                            reply_text += event.delta
                            live.update(Markdown(flatten_headings(reply_text)))
                        elif event.type == "response.output_item.done":
                            if event.item.type == "function_call":
                                tool_calls.append(event.item)
                            elif event.item.type == "reasoning":
                                reasoning_items.append(event.item)

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
                        cmd = args["command"]
                        if not isinstance(cmd, str) or not cmd.strip():
                            raise ValueError("command must be a non-empty string")
                    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                        output = f"[tool argument error: {e}]"
                        console.print(f"[red]{output}[/red]")
                    else:
                        console.print()
                        console.print(Rule(f"[bold yellow]$ {escape(cmd)}[/bold yellow]", style="yellow", align="left"))
                        with Live(
                            Spinner("dots", text=" Running command..."),
                            refresh_per_second=15,
                            console=console,
                        ):
                            result = execute_command(cmd, config.get("command_timeout", 60))
                        display_command_result(result)
                        output = result.to_model_output()
                        console.print(Rule(style="bright_black"))

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
                break
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        save_history(history[-max_history:], session_context.history_file)
    except OpenAIError as e:
        console.print(f"[red]OpenAI API error:[/red] {e}")
        save_history(history[-max_history:], session_context.history_file)
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        save_history(history[-max_history:], session_context.history_file)
        raise SystemExit(1)
