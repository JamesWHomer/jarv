#!/usr/bin/env python3
"""jarv - a simple CLI agent powered by OpenAI"""

import sys
import json
import os
import platform
import subprocess
import threading
import urllib.request
from pathlib import Path
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

console = Console()

CONFIG_DIR = Path.home() / ".jarv"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history.json"
SHA_FILE = CONFIG_DIR / "last_sha.txt"

GITHUB_REPO = "JamesWHomer/jarv"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/commits/main"
INSTALL_URL = f"https://github.com/{GITHUB_REPO}.git"

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
        console.print(f"[green]Config created at[/green] {CONFIG_FILE}")
        console.print("[dim]Set your OpenAI API key there or via the OPENAI_API_KEY env var.[/dim]")
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


DISPLAY_LINE_LIElastic License 2.0 = 30


def flatten_headings(text: str) -> str:
    import re
    return re.sub(r"^#{1,6}\s+(.+)$", r"**\1**", text, flags=re.MULTILINE)


def display_output(output: str) -> None:
    lines = output.splitlines()
    if len(lines) > DISPLAY_LINE_LIElastic License 2.0:
        console.print("\n".join(lines[:DISPLAY_LINE_LIElastic License 2.0]), style="dim")
        hidden = len(lines) - DISPLAY_LINE_LIElastic License 2.0
        console.print(f"[dim italic]... {hidden} more lines hidden (full output sent to model)[/dim italic]")
    else:
        console.print(output, style="dim")


def run_command(command: str) -> str:
    try:
        if platform.system() == "Windows":
            # Match the shell we advertise to the model in get_system_info().
            # subprocess with shell=True uses cmd.exe on Windows, which breaks
            # PowerShell commands like Get-ChildItem.
            shell_command = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ]
            proc = subprocess.Popen(
                shell_command,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        else:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        try:
            stdout, stderr = proc.communicate(timeout=60)
        except KeyboardInterrupt:
            proc.kill()
            proc.wait()
            raise
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return "[timed out after 60 seconds]"
        parts = []
        if stdout:
            parts.append(stdout.rstrip())
        if stderr:
            parts.append(f"[stderr] {stderr.rstrip()}")
        if proc.returncode != 0:
            parts.append(f"[exit code {proc.returncode}]")
        return "\n".join(parts) if parts else "(no output)"
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return f"[error: {e}]"


def build_input(history: list, max_history: int) -> list:
    """Convert stored history to Responses API input format."""
    slice_ = history[-max_history:]
    # Drop leading non-user items to avoid orphaned tool call pairs after truncation.
    for i, m in enumerate(slice_):
        if m.get("role") == "user":
            slice_ = slice_[i:]
            break
    else:
        slice_ = []
    items = []
    for m in slice_:
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
                new_items = []
                for ri in reasoning_items:
                    rd = {"type": "reasoning", "id": ri.id, "summary": []}
                    history.append(rd)
                    new_items.append(rd)
                for item in tool_calls:
                    cmd = json.loads(item.arguments)["command"]
                    console.print()
                    console.print(Rule(f"[bold yellow]$ {cmd}[/bold yellow]", style="yellow", align="left"))
                    with Live(
                        Spinner("dots", text=" Running command..."),
                        refresh_per_second=15,
                        console=console,
                    ):
                        output = run_command(cmd)
                    display_output(output)
                    console.print(Rule(style="bright_black"))

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
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        save_history(history[-max_history:])


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
        console.print("[red]Usage:[/red] jarv set <key> <value>")
        console.print(f"[dim]Keys: {', '.join(DEFAULT_CONFIG.keys())}[/dim]")
        return
    key, raw = args[0], " ".join(args[1:])
    config = load_config()
    value = coerce_value(raw)
    config[key] = value
    save_config(config)
    display = "[dim]***[/dim]" if key == "api_key" else f"[green]{repr(value)}[/green]"
    console.print(f"[bold cyan]{key}[/bold cyan] = {display}")


def cmd_unset(args: list) -> None:
    if not args:
        console.print("[red]Usage:[/red] jarv unset <key>")
        return
    key = args[0]
    config = load_config()
    if key not in config:
        console.print(f"[yellow]'{key}'[/yellow] is not set.")
        return
    if key in DEFAULT_CONFIG:
        config[key] = DEFAULT_CONFIG[key]
        save_config(config)
        console.print(f"[bold cyan]{key}[/bold cyan] reset to default: [dim]{repr(DEFAULT_CONFIG[key])}[/dim]")
    else:
        del config[key]
        save_config(config)
        console.print(f"[bold cyan]{key}[/bold cyan] removed.")


def print_help() -> None:
    cmd_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    cmd_table.add_column(style="bold cyan", no_wrap=True)
    cmd_table.add_column(style="dim")
    cmd_table.add_row("jarv <question>", "Ask jarv anything")
    cmd_table.add_row("jarv set <key> <value>", "Set a config value")
    cmd_table.add_row("jarv unset <key>", "Reset a config key to its default")
    cmd_table.add_row("jarv clear", "Clear conversation history")
    cmd_table.add_row("jarv history", "Show recent conversation history")
    cmd_table.add_row("jarv config", "Show current settings")
    cmd_table.add_row("jarv update", "Update jarv to the latest version")
    cmd_table.add_row("jarv help", "Show this help")

    key_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    key_table.add_column(style="bold yellow", no_wrap=True)
    key_table.add_column(style="dim")
    key_table.add_row("api_key", "OpenAI API key")
    key_table.add_row("model", "Model name (e.g. gpt-5.4-mini, gpt-4o)")
    key_table.add_row("reasoning_effort", "low, medium, high — or unset to disable")
    key_table.add_row("max_history", "Number of messages to keep as context")
    key_table.add_row("system_prompt", "System prompt sent to the model")

    console.print(Panel(cmd_table, title="[bold]jarv[/bold]", border_style="bright_black", padding=(1, 2)))
    console.print()
    console.print("[bold]Config keys[/bold]")
    console.print(key_table)
    console.print(f"\n[dim]Config:  {CONFIG_FILE}[/dim]")
    console.print(f"[dim]History: {HISTORY_FILE}[/dim]")


def _fetch_latest_sha() -> str | None:
    try:
        req = urllib.request.Request(GITHUB_API, headers={"User-Agent": "jarv-updater"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return data["sha"]
    except Exception:
        return None


def _load_known_sha() -> str:
    if SHA_FILE.exists():
        return SHA_FILE.read_text().strip()
    return ""


def _save_sha(sha: str) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    SHA_FILE.write_text(sha)


_update_available: list[str] = []


def _check_update_background() -> None:
    latest = _fetch_latest_sha()
    if latest and latest != _load_known_sha():
        _update_available.append(latest)


def cmd_update() -> None:
    console.print("[dim]Checking for updates...[/dim]")
    latest = _fetch_latest_sha()
    if latest is None:
        console.print("[red]Could not reach GitHub.[/red]")
        return
    known = _load_known_sha()
    if latest == known:
        console.print("[green]Already up to date.[/green]")
        return
    console.print("[cyan]Update found. Installing...[/cyan]")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", f"git+https://github.com/{GITHUB_REPO}.git"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        _save_sha(latest)
        console.print("[green]Updated successfully! Restart jarv to use the new version.[/green]")
    else:
        console.print("[red]Update failed:[/red]")
        console.print(result.stderr.strip(), style="dim")


def main() -> None:
    update_thread = threading.Thread(target=_check_update_background, daemon=True)
    update_thread.start()

    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    args = sys.argv[1:]
    command = args[0].lower()

    if command == "help":
        print_help()
        return

    if command == "update":
        cmd_update()
        return

    if command == "clear":
        save_history([])
        console.print("[dim]History cleared.[/dim]")
        return

    if command == "history":
        history = load_history()
        if not history:
            console.print("[dim]No history yet.[/dim]")
            return
        for m in history:
            role = m.get("role")
            if role == "user":
                console.print(f"\n[bold cyan]You[/bold cyan]  {m.get('content', '')}")
            elif role == "assistant":
                content = m.get("content", "")
                if content:
                    console.print(f"\n[bold green]Jarv[/bold green]")
                    console.print(Markdown(flatten_headings(content)))
        return

    if command == "set":
        cmd_set(args[1:])
        return

    if command == "unset":
        cmd_unset(args[1:])
        return

    if command == "config":
        config = load_config()
        table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
        table.add_column("Key", style="bold cyan")
        table.add_column("Value")
        for k, v in config.items():
            val = "[dim]***[/dim]" if k == "api_key" and v else repr(v)
            table.add_row(k, val)
        console.print(f"[dim]{CONFIG_FILE}[/dim]")
        console.print(table)
        return

    query = " ".join(args)
    config = load_config()

    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print(f"[red]No API key found.[/red] Edit {CONFIG_FILE} or set OPENAI_API_KEY.")
        sys.exit(1)

    update_thread.join(timeout=3)
    if _update_available:
        sha = _update_available[0]
        if not _load_known_sha():
            _save_sha(sha)
        else:
            console.print("[yellow]Update available![/yellow] Run [bold]jarv update[/bold] to install.")
    elif not _load_known_sha():
        latest = _fetch_latest_sha()
        if latest:
            _save_sha(latest)

    client = OpenAI(api_key=api_key)
    try:
        run_agent(query, config, client)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)


if __name__ == "__main__":
    main()
