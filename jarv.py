#!/usr/bin/env python3
"""jarv - a simple CLI agent powered by OpenAI"""

import sys
import json
import os
import platform
import subprocess
import threading
import urllib.request
import signal
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI, OpenAIError
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.panel import Panel
from rich.markup import escape
from rich.rule import Rule
from rich import box

console = Console()

__version__ = "0.1.0"
VALID_REASONING_EFFORTS = {"", "low", "medium", "high"}
REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")

CONFIG_DIR = Path.home() / ".jarv"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history.json"
SHA_FILE = CONFIG_DIR / "last_sha.txt"

GITHUB_REPO = "JamesWHomer/jarv"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/commits/main"
INSTALL_URL = f"https://github.com/{GITHUB_REPO}.git"

DEFAULT_CONFIG = {
    "api_key": "",
    "model": "gpt-4o-mini",
    "reasoning_effort": "",
    "max_history": 40,
    "command_timeout": 60,
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
        CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        console.print(f"[green]Config created at[/green] {CONFIG_FILE}")
        console.print("[dim]Set your OpenAI API key there or via the OPENAI_API_KEY env var.[/dim]")
        sys.exit(0)
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        backup = CONFIG_FILE.with_suffix(".json.bak")
        CONFIG_FILE.replace(backup)
        CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        console.print(f"[red]Config file was invalid JSON:[/red] {e}")
        console.print(f"[yellow]Backed it up to[/yellow] {backup}")
        console.print(f"[green]Created a fresh config at[/green] {CONFIG_FILE}")
        sys.exit(1)
    except (OSError, UnicodeDecodeError) as e:
        console.print(f"[red]Could not read config:[/red] {e}")
        sys.exit(1)
    if not isinstance(config, dict):
        console.print(f"[red]Config must be a JSON object:[/red] {CONFIG_FILE}")
        sys.exit(1)
    for k, v in DEFAULT_CONFIG.items():
        config.setdefault(k, v)
    return config


def save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    try:
        CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except OSError as e:
        console.print(f"[red]Could not save config:[/red] {e}")
        sys.exit(1)


def load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(history, list):
            return history
        console.print(f"[yellow]Ignoring invalid history format:[/yellow] {HISTORY_FILE}")
    except json.JSONDecodeError as e:
        console.print(f"[yellow]Ignoring malformed history:[/yellow] {e}")
    except (OSError, UnicodeDecodeError) as e:
        console.print(f"[yellow]Could not read history:[/yellow] {e}")
    return []


def save_history(history: list) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    try:
        HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")
    except OSError as e:
        console.print(f"[yellow]Could not save history:[/yellow] {e}")


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


@dataclass
class CommandResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False
    timeout: int | float = 60

    def to_model_output(self) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout.rstrip())
        if self.stderr:
            parts.append(f"[stderr] {self.stderr.rstrip()}")
        if self.timed_out:
            parts.append(f"[timed out after {self.timeout:g} seconds]")
        elif self.exit_code not in (None, 0):
            parts.append(f"[exit code {self.exit_code}]")
        return "\n".join(parts) if parts else "(no output)"


def _kill_process_tree(proc: subprocess.Popen) -> None:
    if platform.system() == "Windows":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.kill()


def execute_command(command: str, timeout: int | float = 60) -> CommandResult:
    try:
        timeout = float(timeout)
        if timeout <= 0:
            timeout = 60
    except (TypeError, ValueError):
        timeout = 60

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
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        else:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid,
            )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return CommandResult(command, stdout or "", stderr or "", proc.returncode, timeout=timeout)
        except KeyboardInterrupt:
            _kill_process_tree(proc)
            proc.wait()
            raise
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            stdout, stderr = proc.communicate()
            return CommandResult(command, stdout or "", stderr or "", proc.returncode, timed_out=True, timeout=timeout)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return CommandResult(command, "", f"[error: {e}]", None, timeout=timeout)


def display_command_result(result: CommandResult) -> None:
    if result.stdout:
        display_output(result.stdout.rstrip())
    if result.stderr:
        if result.stdout:
            console.print()
        console.print("stderr:", style="bold red")
        display_output(result.stderr.rstrip())
    if result.timed_out:
        console.print(f"[bold red]Timed out after {result.timeout:g}s[/bold red]")
    elif result.exit_code not in (None, 0):
        console.print(f"[bold red]Exit code:[/bold red] {result.exit_code}")
    else:
        console.print("[dim]Exit code: 0[/dim]")
    if not result.stdout and not result.stderr:
        console.print("(no output)", style="dim")


def run_command(command: str) -> str:
    return execute_command(command).to_model_output()

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
        role = m.get("role")
        typ = m.get("type")
        try:
            if role == "user":
                items.append({"role": "user", "content": str(m.get("content", ""))})
            elif role == "assistant":
                items.append({"role": "assistant", "content": str(m.get("content") or "")})
            elif typ == "reasoning" and "id" in m:
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
        except KeyError:
            continue
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


def model_supports_reasoning(model: str) -> bool:
    return model.lower().startswith(REASONING_MODEL_PREFIXES)


def validate_config(config: dict) -> bool:
    ok = True
    model = config.get("model")
    if not isinstance(model, str) or not model.strip():
        console.print("[red]Config 'model' must be a non-empty string.[/red]")
        ok = False

    effort = config.get("reasoning_effort", "")
    if effort is None:
        config["reasoning_effort"] = ""
    elif effort not in VALID_REASONING_EFFORTS:
        console.print("[red]Config 'reasoning_effort' must be one of: low, medium, high, or empty.[/red]")
        ok = False

    for key in ("max_history", "command_timeout"):
        try:
            value = int(config.get(key, DEFAULT_CONFIG[key]))
            if value <= 0:
                raise ValueError
            config[key] = value
        except (TypeError, ValueError):
            console.print(f"[red]Config '{key}' must be a positive integer.[/red]")
            ok = False

    return ok


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
    effort = config.get("reasoning_effort")
    if effort and model_supports_reasoning(config["model"]):
        kwargs["reasoning"] = {"effort": effort}
    elif effort:
        console.print(
            f"[dim]Ignoring reasoning_effort for model '{config['model']}' "
            "(not a reasoning model).[/dim]"
        )

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
    except OpenAIError as e:
        console.print(f"[red]OpenAI API error:[/red] {e}")
        save_history(history[-max_history:])
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        save_history(history[-max_history:])
        raise SystemExit(1)


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
    key_table.add_row("model", "Model name (default: gpt-4o-mini)")
    key_table.add_row("reasoning_effort", "low, medium, high, or empty to disable")
    key_table.add_row("max_history", "Number of messages to keep as context")
    key_table.add_row("command_timeout", "Seconds before a shell command is killed")
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
    if not validate_config(config):
        sys.exit(1)

    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print(f"[red]No API key found.[/red] Edit {CONFIG_FILE} or set OPENAI_API_KEY.")
        sys.exit(1)

    update_thread = threading.Thread(target=_check_update_background, daemon=True)
    update_thread.start()
    update_thread.join(timeout=0.2)
    if _update_available:
        sha = _update_available[0]
        if not _load_known_sha():
            _save_sha(sha)
        else:
            console.print("[yellow]Update available![/yellow] Run [bold]jarv update[/bold] to install.")

    client = OpenAI(api_key=api_key)
    try:
        run_agent(query, config, client)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)


if __name__ == "__main__":
    main()
