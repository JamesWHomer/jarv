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
import hashlib
from datetime import datetime, timezone
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

CONFIG_DIR = Path.home() / ".jarv"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history.json"
SESSIONS_FILE = CONFIG_DIR / "sessions.json"
SHA_FILE = CONFIG_DIR / "last_sha.txt"

GITHUB_REPO = "JamesWHomer/jarv"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/commits/main"
INSTALL_URL = f"https://github.com/{GITHUB_REPO}.git"

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarv, a helpful CLI assistant. "
    "You can run shell commands when needed to answer questions or complete tasks. "
    "Be concise and direct. "
    "When the user asks about jarv commands, behavior, config, updating, or usage, "
    "run `jarv help` before answering. Do not invent unsupported commands."
)

DEFAULT_CONFIG = {
    "api_key": "",
    "model": "gpt-5.4-mini",
    "reasoning_effort": "",
    "max_history": 40,
    "command_timeout": 60,
    "history_scope": "global",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
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
    changed = False
    for k, v in DEFAULT_CONFIG.items():
        if k not in config:
            config[k] = v
            changed = True

    if changed:
        save_config(config)

    return config


def save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    try:
        CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except OSError as e:
        console.print(f"[red]Could not save config:[/red] {e}")
        sys.exit(1)


def load_history(path: Path = HISTORY_FILE) -> list:
    if not path.exists():
        return []
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(history, list):
            return history
        console.print(f"[yellow]Ignoring invalid history format:[/yellow] {path}")
    except json.JSONDecodeError as e:
        console.print(f"[yellow]Ignoring malformed history:[/yellow] {e}")
    except (OSError, UnicodeDecodeError) as e:
        console.print(f"[yellow]Could not read history:[/yellow] {e}")
    return []


def save_history(history: list, path: Path = HISTORY_FILE) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    try:
        path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    except OSError as e:
        console.print(f"[yellow]Could not save history:[/yellow] {e}")


def load_sessions() -> dict:
    if not SESSIONS_FILE.exists():
        return {"sessions": {}, "last_global_session_id": ""}
    try:
        data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("sessions", {})
            data.setdefault("last_global_session_id", "")
            if isinstance(data["sessions"], dict):
                return data
    except json.JSONDecodeError as e:
        console.print(f"[yellow]Ignoring malformed sessions metadata:[/yellow] {e}")
    except (OSError, UnicodeDecodeError) as e:
        console.print(f"[yellow]Could not read sessions metadata:[/yellow] {e}")
    return {"sessions": {}, "last_global_session_id": ""}


def save_sessions(data: dict) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    try:
        SESSIONS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        console.print(f"[yellow]Could not save sessions metadata:[/yellow] {e}")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def format_elapsed(since: str | None, now: datetime) -> str:
    then = parse_timestamp(since)
    if then is None:
        return "unknown"
    seconds = max(0, int((now - then).total_seconds()))
    if seconds < 60:
        return f"{seconds} seconds"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minutes"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} hours"
    days = hours // 24
    return f"{days} days"


def get_shell_name() -> str:
    shell = os.environ.get("SHELL")
    if not shell:
        shell = "PowerShell" if os.environ.get("PSModulePath") else os.environ.get("ComSpec", "cmd.exe")
    return shell


def short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:length]


def detect_terminal_session() -> tuple[str, str, str]:
    candidates = [
        ("windows-terminal", os.environ.get("WT_SESSION")),
        ("term-session", os.environ.get("TERM_SESSION_ID")),
        ("tmux", os.environ.get("TMUX")),
        ("screen", os.environ.get("STY")),
    ]
    for source, value in candidates:
        if value:
            session_id = f"{source}-{short_hash(value)}"
            return session_id, f"{source} {session_id[-6:]}", source

    user = os.environ.get("USERNAME") or os.environ.get("USER") or "unknown-user"
    raw = "|".join([str(os.getppid()), os.getcwd(), user, get_shell_name()])
    session_id = f"parent-{short_hash(raw)}"
    return session_id, f"parent process {os.getppid()}", "parent-process"


def independent_session() -> tuple[str, str, str]:
    now = utc_now()
    raw = f"{now.isoformat()}|{os.getpid()}|{os.getppid()}|{os.getcwd()}"
    session_id = f"independent-{short_hash(raw)}"
    return session_id, f"independent {session_id[-6:]}", "independent"


def history_file_for_session(session_id: str) -> Path:
    return CONFIG_DIR / f"history-{short_hash(session_id)}.json"


def last_user_message(history: list) -> dict | None:
    for item in reversed(history):
        if isinstance(item, dict) and item.get("role") == "user":
            return item
    return None


@dataclass
class SessionContext:
    scope: str
    session_id: str
    session_label: str
    session_source: str
    history_file: Path
    is_new_session: bool
    previous_user_at: str | None
    previous_user_session_id: str
    previous_user_session_label: str
    previous_global_session_changed: bool
    now: datetime

    @property
    def elapsed_since_previous_user(self) -> str:
        return format_elapsed(self.previous_user_at, self.now)


def prepare_session_context(
    config: dict,
    *,
    independent: bool = False,
    session_override: tuple[str, str, str] | None = None,
    mark_message: bool = False,
) -> SessionContext:
    now = utc_now()
    if session_override:
        session_id, session_label, session_source = session_override
        scope = "independent" if independent else config.get("history_scope", DEFAULT_CONFIG["history_scope"])
        history_path = history_file_for_session(session_id) if scope != "global" else HISTORY_FILE
    elif independent:
        session_id, session_label, session_source = independent_session()
        scope = "independent"
        history_path = history_file_for_session(session_id)
    else:
        session_id, session_label, session_source = detect_terminal_session()
        scope = config.get("history_scope", DEFAULT_CONFIG["history_scope"])
        history_path = HISTORY_FILE if scope == "global" else history_file_for_session(session_id)

    sessions = load_sessions()
    last_global_session_id = str(sessions.get("last_global_session_id") or "")
    session_map = sessions.setdefault("sessions", {})
    is_new_session = session_id not in session_map
    meta = session_map.setdefault(
        session_id,
        {
            "label": session_label,
            "source": session_source,
            "first_seen_at": isoformat_utc(now),
        },
    )
    meta.update(
        {
            "label": session_label,
            "source": session_source,
            "last_seen_at": isoformat_utc(now),
            "history_file": str(history_path),
        }
    )

    history = load_history(history_path)
    previous_user = last_user_message(history)
    previous_user_at = previous_user.get("created_at") if previous_user else None
    previous_user_session_id = str(previous_user.get("session_id") or "") if previous_user else ""
    previous_user_session_label = str(previous_user.get("session_label") or "") if previous_user else ""
    comparison_session_id = previous_user_session_id or last_global_session_id
    previous_global_session_changed = scope == "global" and bool(comparison_session_id) and comparison_session_id != session_id

    if scope == "global":
        sessions["last_global_session_id"] = session_id
    if mark_message:
        meta["last_message_at"] = isoformat_utc(now)
    save_sessions(sessions)

    return SessionContext(
        scope=scope,
        session_id=session_id,
        session_label=session_label,
        session_source=session_source,
        history_file=history_path,
        is_new_session=is_new_session,
        previous_user_at=previous_user_at,
        previous_user_session_id=previous_user_session_id,
        previous_user_session_label=previous_user_session_label,
        previous_global_session_changed=previous_global_session_changed,
        now=now,
    )


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


def validate_config(config: dict) -> bool:
    ok = True
    model = config.get("model")
    if not isinstance(model, str) or not model.strip():
        console.print("[red]Config 'model' must be a non-empty string.[/red]")
        ok = False

    effort = config.get("reasoning_effort", "")
    if effort is None:
        config["reasoning_effort"] = ""

    history_scope = config.get("history_scope", DEFAULT_CONFIG["history_scope"])
    if history_scope not in {"global", "terminal"}:
        console.print("[red]Config 'history_scope' must be 'global' or 'terminal'.[/red]")
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


def history_metadata(context: SessionContext) -> dict:
    return {
        "created_at": isoformat_utc(context.now),
        "session_id": context.session_id,
        "session_label": context.session_label,
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
    cmd_table.add_row("jarv", "Start heads-up mode for repeated prompts")
    cmd_table.add_row("jarv <question>", "Ask jarv anything")
    cmd_table.add_row("jarv session", "Start an independent heads-up session with separate history")
    cmd_table.add_row("jarv set <key> <value>", "Set a config value")
    cmd_table.add_row("jarv unset <key>", "Reset a config key to its default")
    cmd_table.add_row("jarv clear", "Clear conversation history")
    cmd_table.add_row("jarv history", "Show recent conversation history")
    cmd_table.add_row("jarv config", "Show current settings")
    cmd_table.add_row("jarv update", "Update jarv to the latest version")
    cmd_table.add_row("jarv about", "Show detailed information about jarv")
    cmd_table.add_row("jarv help", "Show this help")

    key_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    key_table.add_column(style="bold yellow", no_wrap=True)
    key_table.add_column(style="dim")
    key_table.add_row("api_key", "OpenAI API key")
    key_table.add_row("model", "Model name (default: gpt-5.4-mini)")
    key_table.add_row("reasoning_effort", "Reasoning effort value (empty to disable)")
    key_table.add_row("max_history", "Number of messages to keep as context")
    key_table.add_row("command_timeout", "Seconds before a shell command is killed")
    key_table.add_row("history_scope", "History mode: global or terminal")
    key_table.add_row("system_prompt", "System prompt sent to the model")

    console.print(Panel(cmd_table, title="[bold]jarv[/bold]", border_style="bright_black", padding=(1, 2)))
    console.print()
    console.print("[bold]Config keys[/bold]")
    console.print(key_table)
    console.print(f"\n[dim]Config:  {CONFIG_FILE}[/dim]")
    console.print(f"[dim]History: {HISTORY_FILE}[/dim]")
    console.print(f"[dim]Sessions: {SESSIONS_FILE}[/dim]")


def print_about() -> None:
    about = f"""# jarv

jarv is a command-line AI assistant powered by OpenAI.

## Basic usage

- `jarv` - Start heads-up mode so you can keep sending prompts without rerunning the command.
- `jarv <question>` - Ask jarv anything. Your words after `jarv` are sent as the user message.
- `jarv session` - Start heads-up mode with an independent history for this terminal run.
- `jarv help` - Show the short command overview.
- `jarv about` - Show this detailed overview.
- `jarv config` - Show current settings. The API key is masked.
- `jarv set <key> <value>` - Set a config value. Values like `true`, `false`, integers, and floats are coerced.
- `jarv unset <key>` - Reset a default config key, or remove a custom key.
- `jarv history` - Show recent user and assistant messages.
- `jarv clear` - Clear saved conversation history.
- `jarv update` - Check GitHub for the latest main commit and install it with pip.

## Heads-up mode

Run `jarv` with no prompt to start an interactive session. Type a prompt and press Enter to send it. Type `exit` or `quit`, or press Ctrl+C, to leave.

Run `jarv session` to start an independent interactive session. It uses a separate history file and does not change your configured default history mode.

## How jarv works

1. Loads config from `{CONFIG_FILE}`.
2. Detects the current terminal/session and chooses the configured history scope.
3. Loads recent conversation history from the active history file.
4. Sends your query, recent history, the configured system prompt, and system info to the OpenAI Responses API. If global history moved to a different terminal, jarv inserts a small `<new_terminal>` marker before the new message.
5. Streams the assistant response in the terminal.
6. If the model calls the shell tool, jarv displays the command, runs it, shows stdout/stderr/exit status, and sends the full command result back to the model.
7. Saves the final assistant response back to history, trimmed to `max_history` items.

## Shell command behavior

- jarv exposes one tool to the model: `run_command`.
- Commands are run only when the model chooses to call that tool.
- On Windows, commands run through PowerShell.
- On other platforms, commands run through the system shell.
- Command output shown in the terminal is shortened after 30 lines, but the full output is sent back to the model.
- Commands are killed after `command_timeout` seconds.
- Interrupted commands/process trees are terminated when possible.

## Config

Config file: `{CONFIG_FILE}`

Keys:

- `api_key` - OpenAI API key. Can also be provided with the `OPENAI_API_KEY` environment variable.
- `model` - OpenAI model name. Default: `{DEFAULT_CONFIG['model']}`.
- `reasoning_effort` - Optional reasoning effort value. Empty disables this setting.
- `max_history` - Number of history items kept as context. Default: `{DEFAULT_CONFIG['max_history']}`.
- `command_timeout` - Seconds before a shell command is killed. Default: `{DEFAULT_CONFIG['command_timeout']}`.
- `history_scope` - History mode. Use `global` for shared history or `terminal` for one history per detected terminal. Default: `{DEFAULT_CONFIG['history_scope']}`.
- `system_prompt` - Instructions sent to the model before each request.

If the config file does not exist, jarv creates it and exits so you can add an API key.
If the config file is invalid JSON, jarv backs it up and creates a fresh default config.

## History and context

Global history file: `{HISTORY_FILE}`
Session metadata file: `{SESSIONS_FILE}`

jarv stores recent conversation items locally, including user messages, assistant messages, and tool-call context needed by the Responses API. `jarv clear` empties the active history file. `jarv history` displays only readable user and assistant messages from the active history.

When `history_scope` is `global`, all terminals share `{HISTORY_FILE}`. jarv still tells the model when a message appears to come from a new or different terminal, and how much time has passed since the previous user message.

When `history_scope` is `terminal`, jarv stores history in `history-<session-id>.json` files under `{CONFIG_DIR}`. `jarv session` always uses an independent `history-<session-id>.json` file for that interactive run, regardless of `history_scope`.

## Updates

- `jarv update` checks `{GITHUB_REPO}` on GitHub and installs the latest version from `{INSTALL_URL}`.
- Normal question runs also do a quick background update check and tell you if an update is available.
- After updating, run `jarv` again to use the new version.

## Files

- Config directory: `{CONFIG_DIR}`
- Config file: `{CONFIG_FILE}`
- History file: `{HISTORY_FILE}`
- Session metadata file: `{SESSIONS_FILE}`
- Last known update SHA: `{SHA_FILE}`

## Version

jarv {__version__}
"""
    console.print(Panel(Markdown(flatten_headings(about)), title="[bold]about jarv[/bold]", border_style="bright_black", padding=(1, 2)))


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
        console.print("[green]Updated successfully! Run jarv again to use the new version.[/green]")
    else:
        console.print("[red]Update failed:[/red]")
        console.print(result.stderr.strip(), style="dim")


def main() -> None:
    if len(sys.argv) < 2:
        config = load_config()
        if not validate_config(config):
            sys.exit(1)
        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            console.print(f"[red]No API key found.[/red] Edit {CONFIG_FILE} or set OPENAI_API_KEY.")
            sys.exit(1)
        client = OpenAI(api_key=api_key)
        run_heads_up_mode(config, client)
        return

    args = sys.argv[1:]
    command = args[0].lower()

    if command == "help":
        print_help()
        return

    if command == "about":
        print_about()
        return

    if command == "session":
        config = load_config()
        if not validate_config(config):
            sys.exit(1)
        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            console.print(f"[red]No API key found.[/red] Edit {CONFIG_FILE} or set OPENAI_API_KEY.")
            sys.exit(1)
        session = independent_session()
        console.print(f"[dim]Independent session: {session[1]}[/dim]")
        client = OpenAI(api_key=api_key)
        run_heads_up_mode(config, client, session_override=session, independent=True)
        return

    if command == "update":
        cmd_update()
        return

    if command == "clear":
        config = load_config()
        if not validate_config(config):
            sys.exit(1)
        session_context = prepare_session_context(config)
        save_history([], session_context.history_file)
        console.print("[dim]History cleared.[/dim]")
        return

    if command == "history":
        config = load_config()
        if not validate_config(config):
            sys.exit(1)
        session_context = prepare_session_context(config)
        history = load_history(session_context.history_file)
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


def run_heads_up_mode(
    config: dict,
    client: OpenAI,
    session_override: tuple[str, str, str] | None = None,
    independent: bool = False,
) -> None:
    title = "jarv independent session" if independent else "jarv heads-up mode"
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print("[dim]Type a prompt and press Enter. Type 'exit' or press Ctrl+C to leave.[/dim]")
    while True:
        try:
            query = console.input("\n[bold cyan]jarv>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            return

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            console.print("[dim]Goodbye.[/dim]")
            return

        run_agent(
            query,
            config,
            client,
            session_override=session_override,
            independent=independent,
        )


if __name__ == "__main__":
    main()
