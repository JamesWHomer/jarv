#!/usr/bin/env python3
"""jarv - a simple CLI agent powered by OpenAI"""

import sys
import json
import os
import platform
import subprocess
from pathlib import Path
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

console = Console()

CONFIG_DIR = Path.home() / ".jarv"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history.json"

DEFAULT_CONFIG = {
    "api_key": "",
    "model": "gpt-5.4-mini",
    "reasoning_effort": "medium",
    "max_history": 40,
    "system_prompt": (
        "You are Jarv, a helpful CLI assistant. "
        "You can run shell commands when needed to answer questions or complete tasks. "
        "Be concise and direct."
    ),
}

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


def load_config() -> dict:
    CONFIG_DIR.mkdir(exist_ok=True)
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
        print(f"Config created at {CONFIG_FILE}")
        print("Set your OpenAI API key there or via the OPENAI_API_KEY env var.")
        sys.exit(0)
    config = json.loads(CONFIG_FILE.read_text())
    for k, v in DEFAULT_CONFIG.items():
        config.setdefault(k, v)
    return config


def save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception:
        return []


def save_history(history: list) -> None:
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def run_command(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60
        )
        parts = []
        if result.stdout:
            parts.append(result.stdout.rstrip())
        if result.stderr:
            parts.append(f"[stderr] {result.stderr.rstrip()}")
        if result.returncode != 0:
            parts.append(f"[exit code {result.returncode}]")
        return "\n".join(parts) if parts else "(no output)"
    except subprocess.TimeoutExpired:
        return "[timed out after 60 seconds]"
    except Exception as e:
        return f"[error: {e}]"


def build_input(history: list, max_history: int) -> list:
    """Convert stored history to Responses API input format."""
    items = []
    for m in history[-max_history:]:
        role = m.get("role")
        typ = m.get("type")
        if role == "user":
            items.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            items.append({"role": "assistant", "content": m.get("content") or ""})
        elif typ == "reasoning":
            items.append({"type": "reasoning", "id": m["id"], "summary": m.get("summary", [])})
        elif typ == "function_call":
            items.append({
                "type": "function_call",
                "id": m["id"],
                "call_id": m["call_id"],
                "name": m["name"],
                "arguments": m["arguments"],
            })
        elif typ == "function_call_output":
            items.append({
                "type": "function_call_output",
                "call_id": m["call_id"],
                "output": m["output"],
            })
    return items


def get_system_info() -> str:
    parts = [
        f"OS: {platform.system()} {platform.release()}",
        f"CWD: {os.getcwd()}",
    ]
    shell = os.environ.get("SHELL")
    if not shell:
        shell = "PowerShell" if os.environ.get("PSModulePath") else os.environ.get("ComSpec", "cmd.exe")
    parts.append(f"Shell: {shell}")
    user = os.environ.get("USERNAME") or os.environ.get("USER")
    if user:
        parts.append(f"User: {user}")
    return "\n".join(parts)


def run_agent(query: str, config: dict, client: OpenAI) -> None:
    history = load_history()
    max_history = config.get("max_history", DEFAULT_CONFIG["max_history"])

    history.append({"role": "user", "content": query})

    input_items = build_input(history, max_history)

    kwargs = dict(
        model=config["model"],
        instructions=config["system_prompt"] + f"\n\nSystem info:\n{get_system_info()}",
        tools=TOOLS,
        input=input_items,
    )
    if effort := config.get("reasoning_effort"):
        kwargs["reasoning"] = {"effort": effort}

    while True:
        reply_text = ""
        tool_calls = []
        reasoning_items = []

        with client.responses.stream(**kwargs) as stream:
            with Live(Markdown(""), refresh_per_second=15, console=console, vertical_overflow="visible") as live:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        reply_text += event.delta
                        live.update(Markdown(reply_text))
                    elif event.type == "response.output_item.done":
                        if event.item.type == "function_call":
                            tool_calls.append(event.item)
                        elif event.item.type == "reasoning":
                            reasoning_items.append(event.item)

        if tool_calls:
            new_items = []
            for ri in reasoning_items:
                rd = {"type": "reasoning", "id": ri.id, "summary": []}
                history.append(rd)
                new_items.append(rd)
            for item in tool_calls:
                cmd = json.loads(item.arguments)["command"]
                console.print(f"\n[bold]$ {cmd}[/bold]")
                output = run_command(cmd)
                console.print(output)

                fc = {
                    "type": "function_call",
                    "id": item.id,
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                }
                fco = {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": output,
                }
                history.extend([fc, fco])
                new_items.extend([fc, fco])
            kwargs["input"] = kwargs["input"] + new_items
        else:
            history.append({"role": "assistant", "content": reply_text})
            save_history(history[-max_history:])
            break


def coerce_value(value: str):
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def cmd_set(args: list) -> None:
    if len(args) < 2:
        print("Usage: jarv set <key> <value>")
        print(f"Keys: {', '.join(DEFAULT_CONFIG.keys())}")
        return
    key, raw = args[0], " ".join(args[1:])
    config = load_config()
    value = coerce_value(raw)
    config[key] = value
    save_config(config)
    display = "***" if key == "api_key" else repr(value)
    print(f"{key} = {display}")


def cmd_unset(args: list) -> None:
    if not args:
        print("Usage: jarv unset <key>")
        return
    key = args[0]
    config = load_config()
    if key not in config:
        print(f"'{key}' is not set.")
        return
    if key in DEFAULT_CONFIG:
        config[key] = DEFAULT_CONFIG[key]
        save_config(config)
        print(f"{key} reset to default: {repr(DEFAULT_CONFIG[key])}")
    else:
        del config[key]
        save_config(config)
        print(f"{key} removed.")


def print_help() -> None:
    print(
        "Usage:\n"
        "  jarv <question or task>      Ask jarv anything\n"
        "  jarv set <key> <value>       Set a config value\n"
        "  jarv unset <key>             Reset a config key to its default\n"
        "  jarv clear                   Clear conversation history\n"
        "  jarv history                 Show recent conversation history\n"
        "  jarv config                  Show config file and current settings\n"
        "  jarv help                    Show this help\n\n"
        f"Config:  {CONFIG_FILE}\n"
        f"History: {HISTORY_FILE}\n\n"
        "Config keys:\n"
        "  api_key           OpenAI API key\n"
        "  model             Model name (e.g. gpt-5.4-mini, gpt-4o)\n"
        "  reasoning_effort  Reasoning effort: low, medium, high (unset to disable)\n"
        "  max_history       Number of messages to keep as context\n"
        "  system_prompt     System prompt sent to the model"
    )


def main() -> None:
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    args = sys.argv[1:]
    command = args[0].lower()

    if command == "help":
        print_help()
        return

    if command == "clear":
        save_history([])
        print("History cleared.")
        return

    if command == "history":
        history = load_history()
        if not history:
            print("No history yet.")
            return
        for m in history:
            role = m.get("role")
            if role == "user":
                print(f"\nYou: {m.get('content', '')}")
            elif role == "assistant":
                print(f"\nJarv: {m.get('content', '')}")
        return

    if command == "set":
        cmd_set(args[1:])
        return

    if command == "unset":
        cmd_unset(args[1:])
        return

    if command == "config":
        config = load_config()
        display = {k: ("***" if k == "api_key" and v else v) for k, v in config.items()}
        print(f"Config file: {CONFIG_FILE}")
        print(json.dumps(display, indent=2))
        return

    query = " ".join(args)
    config = load_config()

    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print(f"No API key found. Edit {CONFIG_FILE} or set OPENAI_API_KEY.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    run_agent(query, config, client)


if __name__ == "__main__":
    main()
