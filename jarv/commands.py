import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from . import __version__
from .config import CONFIG_DIR, CONFIG_FILE, DEFAULT_CONFIG, load_config, save_config, validate_config
from .display import console, flatten_headings, jarv_panel, section_rule, status_line
from .history import (
    SESSIONS_DIR,
    SESSIONS_FILE,
    artifact_file_for,
    detect_terminal,
    forget_current_session,
    history_file_for_session,
    isoformat_utc,
    load_history,
    load_redo_stack,
    load_sessions,
    parse_timestamp,
    prepare_session_context,
    redo_file_for,
    save_history,
    save_redo_stack,
    save_sessions,
    set_terminal_session,
    split_last_exchange,
    utc_now,
)
from .usage import (
    estimate_token_cost_usd,
    format_cost,
    format_int,
    known_context_window,
    load_usage,
    usage_file_for,
)

ARCHIVE_DIR = CONFIG_DIR / "archive"

GITHUB_REPO = "JamesWHomer/jarv"
PYPI_VERSION_URL = "https://pypi.org/pypi/jarv/json"
UPDATE_FLAG_FILE = CONFIG_DIR / "update_available.txt"
LAST_CHECK_FILE = CONFIG_DIR / "last_update_check.txt"
UPDATE_CHECK_INTERVAL_HOURS = 24


def _read_key(text_mode: bool = False) -> str:
    """Read a single keypress and return a normalised token.

    Returns one of: UP, DOWN, LEFT, RIGHT, HOME, END, PAGEUP, PAGEDOWN,
    ENTER, ESC, TAB, CTRL_F, BACKSPACE, or the raw character.  Raises KeyboardInterrupt on
    Ctrl-C.  When ``text_mode`` is True, the convenience q/Q → ESC mapping is
    disabled so a search query can include those letters.
    """
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            second = msvcrt.getwch()
            return {
                "H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT",
                "G": "HOME", "O": "END",
                "I": "PAGEUP", "Q": "PAGEDOWN",
            }.get(second, "OTHER")
        if ch == "\r":
            return "ENTER"
        if ch == "\t":
            return "TAB"
        if ch == "\x1b":
            return "ESC"
        if not text_mode and ch in ("q", "Q"):
            return "ESC"
        if ch == "\x06":
            return "CTRL_F"
        if ch in ("\x08", "\x7f"):
            return "BACKSPACE"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 in ("5", "6"):
                        sys.stdin.read(1)  # consume trailing ~
                    return {
                        "A": "UP", "B": "DOWN", "D": "LEFT", "C": "RIGHT",
                        "H": "HOME", "F": "END",
                        "5": "PAGEUP", "6": "PAGEDOWN",
                    }.get(ch3, "OTHER")
                return "ESC"
            if ch in ("\r", "\n"):
                return "ENTER"
            if ch == "\t":
                return "TAB"
            if not text_mode and ch in ("q", "Q"):
                return "ESC"
            if ch == "\x06":
                return "CTRL_F"
            if ch in ("\x7f", "\x08"):
                return "BACKSPACE"
            if ch == "\x03":
                raise KeyboardInterrupt
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


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
        console.print(status_line("✗", "jarv /set <key> <value>", prefix_style="bold red", message_style="dim"))
        console.print(f"  [dim]Keys: {', '.join(DEFAULT_CONFIG.keys())}[/dim]")
        return
    key, raw = args[0], " ".join(args[1:])
    config = load_config()
    value = coerce_value(raw)
    config[key] = value
    save_config(config)
    display = "[dim]***[/dim]" if key == "api_key" else f"[green]{repr(value)}[/green]"
    console.print(f"[bold cyan]✓[/bold cyan] [bold cyan]{key}[/bold cyan] [dim]=[/dim] {display}")


def cmd_unset(args: list) -> None:
    if not args:
        console.print(status_line("✗", "jarv /unset <key>", prefix_style="bold red", message_style="dim"))
        return
    key = args[0]
    config = load_config()
    if key not in config:
        console.print(f"[yellow]○[/yellow] [bold]{key}[/bold] [dim]is not set.[/dim]")
        return
    if key in DEFAULT_CONFIG:
        config[key] = DEFAULT_CONFIG[key]
        save_config(config)
        console.print(f"[bold cyan]↺[/bold cyan] [bold cyan]{key}[/bold cyan] [dim]reset to default →[/dim] [green]{repr(DEFAULT_CONFIG[key])}[/green]")
    else:
        del config[key]
        save_config(config)
        console.print(f"[bold cyan]✓[/bold cyan] [bold cyan]{key}[/bold cyan] [dim]removed.[/dim]")


def print_help() -> None:
    cmd_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    cmd_table.add_column(style="bold cyan", no_wrap=True)
    cmd_table.add_column(style="white")
    cmd_table.add_row("jarv", "Start heads-up mode for repeated prompts")
    cmd_table.add_row("jarv <question>", "Ask jarv anything")
    cmd_table.add_row("jarv /set <key> <value>", "Set a config value")
    cmd_table.add_row("jarv /unset <key>", "Reset a config key to its default")
    cmd_table.add_row("jarv /new", "Start a fresh session on the next message")
    cmd_table.add_row("jarv /archive", "Archive this terminal's session and start a fresh one")
    cmd_table.add_row("jarv /sessions, /session", "List sessions (all in a TTY; 5 most recent when piped/non-TTY)")
    cmd_table.add_row("jarv /sessions <id>", "Load a specific session into this terminal by id prefix")
    cmd_table.add_row("jarv /history", "Show recent conversation history")
    cmd_table.add_row("jarv /usage", "Show token usage for this session")
    cmd_table.add_row("jarv /undo [n]", "Unsend the last n exchanges (default 1)")
    cmd_table.add_row("jarv /redo [n]", "Restore the last n undone exchanges (default 1)")
    cmd_table.add_row("jarv /config", "Show current settings")
    cmd_table.add_row("jarv /update", "Update jarv to the latest version")
    cmd_table.add_row("jarv /about", "Show detailed information about jarv")
    cmd_table.add_row("jarv /setup", "Run the setup wizard")
    cmd_table.add_row("jarv /help", "Show this help")

    key_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    key_table.add_column(style="bold yellow", no_wrap=True)
    key_table.add_column(style="white")
    key_table.add_row("provider", "API provider (openai, openrouter, anthropic, gemini, groq, deepseek, ...)")
    key_table.add_row("api_key", "API key (or use provider-specific env var)")
    key_table.add_row("base_url", "Custom API base URL (overrides provider default)")
    key_table.add_row("model", "Model name (default: gpt-5.4-mini)")
    key_table.add_row("reasoning_effort", "Reasoning effort value (empty to disable)")
    key_table.add_row("max_history", "Number of messages to keep as context")
    key_table.add_row("command_timeout", "Seconds before a shell command is killed")
    key_table.add_row("command_safety", "Command confirmation level (all, risky, none)")
    key_table.add_row("audit", "LLM auditor for flagged commands (true/false)")
    key_table.add_row("auditor_auto_approve", "Let auditor auto-approve safe commands (true/false)")
    key_table.add_row("auditor_model", "Model for auditor (empty = auto-select fast model)")
    key_table.add_row("system_prompt", "System prompt sent to the model")
    key_table.add_row("max_subagent_depth", "Max spawn depth for nested subagents")
    key_table.add_row("subagent_thread_pool_max_workers", "Parallel subagents per spawn call")
    key_table.add_row("check_updates", "Non-blocking background update check on one-shot runs (true/false)")

    paths_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    paths_table.add_column(style="dim", no_wrap=True)
    paths_table.add_column(style="dim")
    paths_table.add_row("Config", str(CONFIG_FILE))
    paths_table.add_row("Sessions index", str(SESSIONS_FILE))
    paths_table.add_row("Session data", str(SESSIONS_DIR))

    body = Group(
        section_rule("commands"),
        Text(""),
        cmd_table,
        Text(""),
        section_rule("config keys"),
        Text(""),
        key_table,
        Text(""),
        section_rule("paths"),
        Text(""),
        paths_table,
    )
    console.print(jarv_panel(body, title="help"))


def print_about() -> None:
    about = f"""# jarv

jarv is a command-line AI assistant that supports multiple AI providers including OpenAI, Anthropic, Google Gemini, OpenRouter, Groq, DeepSeek, and more.

## Basic usage

- `jarv` - Start heads-up mode so you can keep sending prompts without rerunning the command.
- `jarv <question>` - Ask jarv anything. Your words after `jarv` are sent as the user message.
- `jarv /help` - Show the short command overview. (`jarv help` also works as a permanent alias.)
- `jarv /about` - Show this detailed overview.
- `jarv /config` - Show current settings. The API key is masked.
- `jarv /set <key> <value>` - Set a config value. Values like `true`, `false`, integers, and floats are coerced.
- `jarv /unset <key>` - Reset a default config key, or remove a custom key.
- `jarv /history` - Show recent user and assistant messages.
- `jarv /usage` - Show token usage for the current session.
- `jarv /undo [n]` - Unsend the last n exchanges (default 1). The removed exchange is pushed onto a redo stack.
- `jarv /redo [n]` - Restore the last n undone exchanges (default 1). Sending a new message clears the redo stack.
- `jarv /new` - Start a fresh session on the next message.
- `jarv /archive` - Archive this terminal's session history and start a fresh one on the next message.
- `jarv /sessions` / `jarv /session` - List sessions by recency. In an interactive terminal you can scroll through all of them; when stdout is not a TTY (e.g. piped), only the 5 most recent are listed.
- `jarv /sessions <id>` - Bind this terminal to a specific session id (prefix match).
- `jarv /update` - Check PyPI for the latest version and install it with pip.

## Heads-up mode

Run `jarv` with no prompt to start an interactive session. Type a prompt and press Enter to send it. Commands start with `/` (e.g. `/new`, `/history`). Type `exit`, `quit`, or `/exit`, or press Ctrl+C, to leave.

## How jarv works

1. Loads config from `{CONFIG_FILE}`.
2. Detects the current terminal and resolves its active session (default: one session per terminal).
3. Loads recent conversation history from that session's history file.
4. Sends your query, recent history, the configured system prompt, and system info to the configured provider's API.
5. Streams the assistant response in the terminal.
6. When the model issues tool calls, jarv runs the matching handler and feeds results back into the model (for `run_command`, that means showing the command, running it, printing stdout/stderr/exit status, and returning the full output).
7. Saves the final assistant response back to history, trimmed to `max_history` items.

## Tools and shell commands

- The root model sees three tools: `run_command`, `spawn`, and `read_artifact`.
- Spawned subagents also get a mandatory `finish` tool (to return output) and may get `spawn` when the parent sets `sterile: false`.
- Shell commands run only when the model calls `run_command`.
- On Windows, `run_command` uses PowerShell.
- On other platforms, `run_command` uses the system shell.
- Command output shown in the terminal is shortened after 30 lines, but the full output is sent back to the model.
- Commands are killed after `command_timeout` seconds.
- Interrupted commands/process trees are terminated when possible.

## Config

Config file: `{CONFIG_FILE}`

Keys:

- `provider` - API provider. Options: openai, openrouter, anthropic, gemini, groq, deepseek, together, fireworks, ollama, lm_studio, vllm. Default: `openai`.
- `api_key` - API key. Can also be provided via provider-specific env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
- `base_url` - Custom API base URL. Overrides the provider's default endpoint.
- `model` - Model name. Default: `{DEFAULT_CONFIG['model']}`.
- `reasoning_effort` - Optional reasoning effort value. Empty disables this setting.
- `max_history` - Number of history items kept as context. Default: `{DEFAULT_CONFIG['max_history']}`.
- `command_timeout` - Seconds before a shell command is killed. Default: `{DEFAULT_CONFIG['command_timeout']}`.
- `command_safety` - Command confirmation level. `all` = confirm every command, `risky` = confirm only dangerous commands (destructive ops, privilege escalation, network exfil, etc.), `none` = no confirmation. Default: `risky`.
- `audit` - When `true`, flagged commands are sent to a fast LLM auditor (uses extra tokens). The auditor's verdict appears inside the safety panel. Works with both `risky` and `all` safety levels. Default: `false`.
- `auditor_auto_approve` - When `true`, the auditor auto-approves commands it deems safe. When `false`, the auditor only shows a recommendation and the user always decides. Default: `true`.
- `auditor_model` - Model used for the auditor. Empty = auto-select a fast/cheap model for the current provider. Default: empty.
- `system_prompt` - Instructions sent to the model before each request.
- `max_subagent_depth` - Maximum recursion depth for `spawn` (root is 0). Default: `{DEFAULT_CONFIG['max_subagent_depth']}`.
- `subagent_thread_pool_max_workers` - Max parallel children in one `spawn` batch. Default: `{DEFAULT_CONFIG['subagent_thread_pool_max_workers']}`.
- `check_updates` - When `true`, a one-shot `jarv <question>` run fires a non-blocking background check against GitHub. If a new version is found it is flagged locally and shown at the start of the next run. Default: `true`. Set to `false` to disable entirely. Heads-up mode (`jarv` with no args) and slash commands do not run this check.
- `/usage` model metadata comes from LiteLLM.

If the config file does not exist, jarv creates it and exits so you can add an API key.
If the config file is invalid JSON, jarv backs it up and creates a fresh default config.

## History and sessions

Session metadata file: `{SESSIONS_FILE}`

Each terminal is bound to exactly one session at a time. By default a fresh terminal gets its own session (id derived from terminal fingerprint). Per-session history and artifact sidecars live in `{SESSIONS_DIR}` as `history-<hash>.json` and `artifacts-<hash>.json`.

- `jarv /new` starts a fresh session by unmapping the current terminal. The next prompt creates a new session.
- `jarv /archive` archives the current session's history+artifacts and removes the terminal's mapping. The next prompt starts a fresh session.
- `jarv /sessions` / `jarv /session` lists sessions by recency (all in a TTY; 5 most recent when stdout is not a TTY).
- `jarv /sessions <id>` binds a specific session id (prefix match) to this terminal.

## Updates

- `jarv /update` checks PyPI for the latest version and installs it with pip.
- A one-shot `jarv <question>` (arguments on the command line, not heads-up mode) fires a fully non-blocking background check when `check_updates` is true. If an update is found it is saved locally; the next invocation shows the notification instantly with no network wait.
- The background check is throttled to at most once every {UPDATE_CHECK_INTERVAL_HOURS} hours.
- Set `check_updates` to `false` (`jarv /set check_updates false`) to disable the background check entirely.
- After updating, run `jarv` again to use the new version.

## Files

- Config directory: `{CONFIG_DIR}`
- Config file: `{CONFIG_FILE}`
- Session metadata file: `{SESSIONS_FILE}`
- Session history and artifacts: `{SESSIONS_DIR}`

## Version

jarv {__version__}
"""
    console.print(jarv_panel(Markdown(flatten_headings(about)), title="about", subtitle=f"v{__version__}"))


def _fetch_latest_pypi_version() -> str | None:
    try:
        req = urllib.request.Request(PYPI_VERSION_URL, headers={"User-Agent": "jarv-updater"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return data["info"]["version"]
    except Exception:
        return None


def _should_check_now() -> bool:
    """Return True if enough time has passed since the last update check."""
    import time
    if not LAST_CHECK_FILE.exists():
        return True
    try:
        last = float(LAST_CHECK_FILE.read_text().strip())
        return (time.time() - last) >= UPDATE_CHECK_INTERVAL_HOURS * 3600
    except Exception:
        return True


def _record_check_time() -> None:
    import time
    CONFIG_DIR.mkdir(exist_ok=True)
    LAST_CHECK_FILE.write_text(str(time.time()))


def _check_update_background() -> None:
    """Check PyPI for a newer version and write a flag file if one is available.

    Runs in a daemon thread — never blocks the main process. The flag is read
    (and cleared) on the *next* jarv invocation so there is zero network wait
    on the current run.
    """
    if not _should_check_now():
        return
    _record_check_time()
    latest = _fetch_latest_pypi_version()
    if not latest:
        return
    if latest != __version__:
        CONFIG_DIR.mkdir(exist_ok=True)
        UPDATE_FLAG_FILE.write_text(latest)


def maybe_print_update_available() -> None:
    """Show a pending update notification written by a previous run's background check."""
    if not UPDATE_FLAG_FILE.exists():
        return
    try:
        latest = UPDATE_FLAG_FILE.read_text().strip()
        UPDATE_FLAG_FILE.unlink(missing_ok=True)
        if latest and latest != __version__:
            console.print(f"[yellow]Update available![/yellow] [dim]v{__version__} → v{latest}[/dim]  Run [bold]jarv /update[/bold] to install.")
    except Exception:
        pass


def _is_pipx_env() -> bool:
    """Detect if jarv is running inside a pipx-managed virtualenv."""
    exe = Path(sys.executable).resolve()
    return "pipx" in exe.parts

def _run_pip_upgrade() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "jarv"],
        capture_output=True, text=True,
    )

def _is_externally_managed_error(result: subprocess.CompletedProcess) -> bool:
    return result.returncode != 0 and "externally-managed-environment" in (result.stderr or "")

def cmd_update() -> None:
    console.print("[dim]⟳ Checking for updates…[/dim]")
    latest = _fetch_latest_pypi_version()
    if latest is None:
        console.print("[bold red]✗[/bold red] [red]Could not reach PyPI.[/red]")
        return
    if latest == __version__:
        console.print(f"[bold green]✓[/bold green] [green]Already up to date.[/green] [dim](v{__version__})[/dim]")
        return
    console.print(f"[bold cyan]↓[/bold cyan] Update found [dim](v{__version__} → v{latest})[/dim]. Installing…")

    if _is_pipx_env():
        with console.status("[dim]Running pipx upgrade…[/dim]", spinner="dots"):
            result = subprocess.run(
                ["pipx", "upgrade", "jarv"],
                capture_output=True, text=True,
            )
    else:
        with console.status("[dim]Running pip install…[/dim]", spinner="dots"):
            result = _run_pip_upgrade()
        if _is_externally_managed_error(result):
            console.print("[dim]System Python detected — retrying with pipx…[/dim]")
            pipx_available = subprocess.run(
                ["pipx", "--version"], capture_output=True, text=True,
            ).returncode == 0
            if pipx_available:
                with console.status("[dim]Running pipx upgrade…[/dim]", spinner="dots"):
                    result = subprocess.run(
                        ["pipx", "upgrade", "jarv"],
                        capture_output=True, text=True,
                    )
            else:
                console.print("[bold red]✗[/bold red] [red]Update failed:[/red] pip is blocked by your system Python (PEP 668).")
                console.print("[dim]Install pipx and reinstall jarv with it:[/dim]")
                console.print("  [bold]brew install pipx && pipx install jarv[/bold]")
                return

    if result.returncode == 0:
        UPDATE_FLAG_FILE.unlink(missing_ok=True)
        console.print("[bold green]✓[/bold green] [green]Updated successfully.[/green] [dim]Run jarv again to use the new version.[/dim]")
    else:
        console.print("[bold red]✗[/bold red] [red]Update failed:[/red]")
        output = "\n".join(filter(None, [result.stdout.strip(), result.stderr.strip()]))
        if output:
            console.print(output, style="dim")


def cmd_new() -> None:
    session_context = prepare_session_context()
    history = load_history(session_context.history_file)
    if not history:
        console.print("[dim]○ Already on a new session.[/dim]")
        return
    forget_current_session()
    console.print("[bold green]✓[/bold green] [green]New session starts on your next message.[/green]")


def archive_session_files(history_path: Path) -> Path | None:
    """Move history/artifacts/usage for a session into ARCHIVE_DIR.

    Returns the new archived history path, or None if nothing was archived.
    """
    if not history_path.exists() or not load_history(history_path):
        return None
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    cleared_at = utc_now().strftime("%Y%m%dT%H%M%SZ")
    stem_suffix = history_path.stem[len("history"):]
    archived_history = ARCHIVE_DIR / f"history-{cleared_at}{stem_suffix}.json"
    history_path.rename(archived_history)

    artifact_path = artifact_file_for(history_path)
    if artifact_path.exists():
        artifact_path.rename(ARCHIVE_DIR / f"artifacts-{cleared_at}{stem_suffix}.json")

    usage_path = usage_file_for(history_path)
    if usage_path.exists():
        usage_path.rename(ARCHIVE_DIR / f"usage-{cleared_at}{stem_suffix}.json")

    redo_path = redo_file_for(history_path)
    if redo_path.exists():
        redo_path.unlink()

    return archived_history


def unarchive_session_files(archived_history_path: Path, session_id: str) -> Path | None:
    """Reverse archive_session_files for the given session id."""
    if not archived_history_path.exists():
        return None
    restored_history = history_file_for_session(session_id)
    archived_history_path.rename(restored_history)

    archived_dir = archived_history_path.parent
    archived_tail = archived_history_path.stem[len("history"):]  # "-{ts}-{hash}"
    restored_suffix = restored_history.stem[len("history"):]  # "-{hash}"
    for kind in ("artifacts", "usage"):
        sib = archived_dir / f"{kind}{archived_tail}.json"
        if sib.exists():
            sib.rename(SESSIONS_DIR / f"{kind}{restored_suffix}.json")
    return restored_history


def _history_content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text" and isinstance(c.get("text"), str):
                    chunks.append(c["text"])
                elif "content" in c and isinstance(c["content"], str):
                    chunks.append(c["content"])
                else:
                    chunks.append(f"[{c.get('type', 'item')}]")
            else:
                chunks.append(str(c))
        return "\n".join(chunks)
    return str(content)


def _rendered_text_lines(renderable, width: int) -> list[Text]:
    options = console.options.update(width=max(1, width))
    rendered = console.render_lines(renderable, options, pad=False)
    lines: list[Text] = []
    for rendered_line in rendered:
        line = Text(no_wrap=True, overflow="crop")
        for segment in rendered_line:
            if segment.text:
                line.append(segment.text, style=segment.style)
        lines.append(line)
    return lines


def _markdown_to_text_lines(content: str, width: int) -> list[Text]:
    return _rendered_text_lines(Markdown(flatten_headings(content)), width)


def _history_visual_lines(history: list, width: int) -> list[Text]:
    lines: list[Text] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).lower()
        if role == "system":
            continue
        body = _history_content_to_str(item.get("content", "")).strip()
        if not body:
            continue
        if role == "user":
            for j, raw in enumerate(body.splitlines() or [""]):
                t = Text(no_wrap=False, overflow="fold")
                if j == 0:
                    t.append("user: ", style="bold cyan")
                else:
                    t.append("  ")
                t.append(raw, style="bold")
                lines.extend(_rendered_text_lines(t, width))
        elif role == "assistant":
            lines.append(Text("jarv:", style="bold green", no_wrap=True, overflow="crop"))
            lines.extend(_markdown_to_text_lines(body, width))
        else:
            label = role or "?"
            for j, raw in enumerate(body.splitlines() or [""]):
                t = Text(no_wrap=False, overflow="fold")
                if j == 0:
                    t.append(f"{label}: ", style="dim")
                else:
                    t.append("  ")
                t.append(raw, style="dim")
                lines.extend(_rendered_text_lines(t, width))
        lines.append(Text(""))
    if lines and lines[-1].plain == "":
        lines.pop()
    return lines


def delete_session_files(history_path: Path) -> None:
    """Permanently remove history/artifacts/usage/redo files for a session."""
    for path in (
        history_path,
        artifact_file_for(history_path),
        usage_file_for(history_path),
        redo_file_for(history_path),
    ):
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


def cmd_archive() -> None:
    session_context = prepare_session_context()
    history_path = session_context.history_file

    archived_history = archive_session_files(history_path)
    if archived_history is not None:
        console.print(f"[bold cyan]▸[/bold cyan] [dim]Session archived to[/dim] [cyan]{archived_history}[/cyan]")
    else:
        console.print("[dim]○ No history to archive.[/dim]")

    forget_current_session()
    if archived_history is not None:
        console.print("[bold green]✓[/bold green] [green]New session starts on your next message.[/green]")



def _short_session_id(sid: str) -> str:
    """Return the shortest unambiguous prefix hint for display (type prefix + 6 hash chars)."""
    # IDs look like: parent-5d44fec1a0fe  or  windows-terminal-3dece1d0fac8
    # Keep the descriptive prefix and show only 6 chars of the trailing hash.
    parts = sid.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) >= 6:
        return f"{parts[0]}-{parts[1][:6]}"
    return sid[:16]


def _sessions_plain(sessions: dict, terminals: dict) -> None:
    """Non-interactive fallback session list (used when stdout is not a tty)."""
    terminal_id, _ = detect_terminal()
    current_session_id = terminals.get(terminal_id)
    now = utc_now()

    def sort_key(sid: str) -> str:
        meta = sessions[sid]
        return meta.get("last_message_at") or meta.get("last_used_at") or ""

    sorted_sessions = sorted(sessions.keys(), key=sort_key, reverse=True)[:5]

    table = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 2), header_style="bold cyan", pad_edge=False)
    table.add_column("", no_wrap=True, width=1)
    table.add_column("ID prefix", style="bold cyan", no_wrap=True)
    table.add_column("Last active", style="dim", no_wrap=True)
    table.add_column("First message")

    for sid in sorted_sessions:
        meta = sessions[sid]
        ts_str = meta.get("last_message_at") or meta.get("last_used_at")
        ts = parse_timestamp(ts_str)
        if ts:
            delta = now - ts
            secs = int(delta.total_seconds())
            if secs < 60:
                time_str = "just now"
            elif secs < 3600:
                time_str = f"{secs // 60}m ago"
            elif secs < 86400:
                time_str = f"{secs // 3600}h ago"
            elif secs < 7 * 86400:
                time_str = f"{secs // 86400}d ago"
            else:
                time_str = ts.strftime("%b %d")
        else:
            time_str = "—"

        snippet = ""
        history_path_str = meta.get("history_file")
        if history_path_str:
            history_path = Path(history_path_str)
            if history_path.exists():
                history = load_history(history_path)
                for item in history:
                    if isinstance(item, dict) and item.get("role") == "user":
                        content = str(item.get("content", "")).replace("\n", " ").strip()
                        if content:
                            snippet = content[:72] + ("…" if len(content) > 72 else "")
                            break

        marker = "[green]●[/green]" if sid == current_session_id else ""
        table.add_row(marker, _short_session_id(sid), time_str, snippet or "[dim]no messages[/dim]")

    total = len(sessions)
    shown = len(sorted_sessions)
    footer_parts: list = [table]
    if total > shown:
        footer_parts += [Text(""), Text(f"Showing {shown} most recent of {total} sessions.", style="dim")]
    footer_parts += [Text("Run jarv /sessions <id> to switch to a session.", style="dim italic")]
    console.print(jarv_panel(Group(*footer_parts), title="sessions", subtitle=f"{shown}/{total}"))


def _cmd_sessions_load(prefix: str) -> None:
    data = load_sessions()
    sessions = data["sessions"]
    if not sessions:
        console.print("[yellow]No sessions exist yet.[/yellow]")
        return
    if prefix in sessions:
        session_id = prefix
    else:
        matches = [sid for sid in sessions if sid.startswith(prefix)]
        if not matches:
            console.print(f"[bold red]✗[/bold red] [red]No session matches:[/red] [bold]{prefix}[/bold]")
            console.print("[dim]  Run [bold]jarv /sessions[/bold] to see available sessions.[/dim]")
            return
        if len(matches) > 1:
            console.print(f"[bold yellow]?[/bold yellow] [yellow]Ambiguous prefix[/yellow] [bold]{prefix}[/bold] [dim]matches {len(matches)} sessions:[/dim]")
            for m in matches:
                console.print(f"  [dim]•[/dim] [cyan]{m}[/cyan]")
            return
        session_id = matches[0]
    set_terminal_session(session_id)
    label = sessions[session_id].get("label", session_id)
    console.print(f"[bold green]✓[/bold green] [green]Loaded[/green] [bold cyan]{_short_session_id(session_id)}[/bold cyan] [dim]({label})[/dim]")


def cmd_sessions(args: list | None = None) -> None:
    if args:
        _cmd_sessions_load(args[0])
        return
    data = load_sessions()
    sessions = data["sessions"]
    terminals = data["terminals"]

    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        console.print("[dim]Sessions are created automatically when you start chatting.[/dim]")
        return

    if not sys.stdin.isatty() or not console.is_terminal:
        _sessions_plain(sessions, terminals)
        return

    terminal_id, _ = detect_terminal()
    current_session_id = terminals.get(terminal_id)
    now = utc_now()

    def sort_key(sid: str) -> str:
        meta = sessions[sid]
        return meta.get("last_message_at") or meta.get("last_used_at") or ""

    sorted_sessions = sorted(sessions.keys(), key=sort_key, reverse=True)

    # Precompute all display data so the live render never blocks on I/O.
    rows: list[dict] = []
    for sid in sorted_sessions:
        meta = sessions[sid]
        ts_str = meta.get("last_message_at") or meta.get("last_used_at")
        ts = parse_timestamp(ts_str)
        if ts:
            delta = now - ts
            secs = int(delta.total_seconds())
            if secs < 60:
                time_str = "just now"
            elif secs < 3600:
                time_str = f"{secs // 60}m ago"
            elif secs < 86400:
                time_str = f"{secs // 3600}h ago"
            elif secs < 7 * 86400:
                time_str = f"{secs // 86400}d ago"
            else:
                time_str = ts.strftime("%b %d")
        else:
            time_str = "—"

        snippet = ""
        hp_str = meta.get("history_file")
        if hp_str:
            hp = Path(hp_str)
            if hp.exists():
                history = load_history(hp)
                for item in history:
                    if isinstance(item, dict) and item.get("role") == "user":
                        content = str(item.get("content", "")).replace("\n", " ").strip()
                        if content:
                            snippet = content[:60] + ("…" if len(content) > 60 else "")
                            break

        rows.append({
            "sid": sid,
            "short_id": _short_session_id(sid),
            "time_str": time_str,
            "snippet": snippet,
            "is_current": sid == current_session_id,
            "archived": bool(meta.get("archived")),
        })

    view_mode = "active"  # "active" | "all" | "archived"
    arm_delete_sid: str | None = None
    flash: tuple[str, str] | None = None  # (message, style) shown above the footer
    search_query: str = ""
    search_active: bool = False  # input bar focused for typing
    search_text_cache: dict[str, str] = {}  # sid -> lowercased transcript text
    last_action: dict | None = None  # most recent undoable action (5s window)
    undo_lock = threading.Lock()
    UNDO_WINDOW = 5.0
    # When a row is archived/unarchived from a filtered view, keep it visible in
    # place (with its new aesthetic) until the cursor moves.
    ghost_sid: str | None = None
    selected_sid: str | None = next(
        (r["sid"] for r in rows if r["is_current"] and not r["archived"]),
        next((r["sid"] for r in rows if not r["archived"]), rows[0]["sid"] if rows else None),
    )
    offset = 0
    preview_sid: str | None = None
    preview_offset = 0
    preview_cache: dict[tuple[str, int], list[Text]] = {}  # (sid, width) -> pre-built visual lines

    def _truncate(value: str, width: int) -> str:
        if width <= 0:
            return ""
        if len(value) <= width:
            return value
        if width <= 3:
            return value[:width]
        return value[:width - 3] + "..."

    def _content_rows(term_h: int, has_status: bool, show_footer: bool, has_search: bool = False) -> int:
        # Panel border = 2 rows. Header consumes 1 row. Footer = 2 rows (blank + controls).
        content = max(1, term_h - 2 - 1)
        if show_footer:
            content -= 2
        if has_status:
            content -= 1
        if has_search:
            content -= 1
        return max(1, content)

    def _max_vis(has_status: bool = False) -> int:
        term_h = console.size.height
        has_search = bool(search_active or search_query)
        return _content_rows(term_h, has_status, show_footer=term_h >= 6, has_search=has_search)

    def _fast_search_text(r: dict) -> str:
        # Cheap fields available without disk I/O — exact short id, full sid,
        # the user's first-message snippet, and the session label.
        meta = sessions.get(r["sid"], {})
        label = meta.get("label", "") if isinstance(meta.get("label"), str) else ""
        return f"{r['short_id']} {r['sid']} {r.get('snippet', '')} {label}".lower()

    def _build_search_text(sid: str) -> str:
        meta = sessions.get(sid, {})
        hp_str = meta.get("history_file")
        chunks: list[str] = []
        label = meta.get("label")
        if isinstance(label, str):
            chunks.append(label)
        if hp_str:
            hp = Path(hp_str)
            if hp.exists():
                try:
                    history = load_history(hp)
                except Exception:
                    history = []
                for item in history:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content", "")
                    if isinstance(content, str):
                        chunks.append(content)
                    elif isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict):
                                t = c.get("text") if c.get("type") == "text" else c.get("content")
                                if isinstance(t, str):
                                    chunks.append(t)
        return "\n".join(chunks).lower()

    def _search_text(sid: str) -> str:
        cached = search_text_cache.get(sid)
        if cached is not None:
            return cached
        text = _build_search_text(sid)
        search_text_cache[sid] = text
        return text

    def _prefetch_worker() -> None:
        for r in rows:
            if prefetch_stop.is_set():
                return
            sid = r["sid"]
            if sid in search_text_cache:
                continue
            try:
                text = _build_search_text(sid)
            except Exception:
                text = ""
            search_text_cache[sid] = text

    prefetch_stop = threading.Event()
    prefetch_thread = threading.Thread(target=_prefetch_worker, daemon=True)
    prefetch_thread.start()

    def _visible_rows_list() -> list[dict]:
        q = search_query.lower().strip()
        def keep(r: dict) -> bool:
            if r["sid"] == ghost_sid:
                return True
            if view_mode == "active" and r["archived"]:
                return False
            if view_mode == "archived" and not r["archived"]:
                return False
            if q:
                if q in _fast_search_text(r):
                    return True
                if q not in _search_text(r["sid"]):
                    return False
            return True
        return [r for r in rows if keep(r)]

    def _selected_pos(visible: list[dict]) -> int:
        for i, r in enumerate(visible):
            if r["sid"] == selected_sid:
                return i
        return 0

    def _clamp_offset(sel: int, off: int, mv: int, n: int) -> int:
        if n == 0:
            return 0
        if sel < off:
            return sel
        if sel >= off + mv:
            return sel - mv + 1
        return max(0, min(off, max(0, n - mv)))

    def _subtitle() -> str:
        n_active = sum(1 for r in rows if not r["archived"])
        n_archived = len(rows) - n_active
        if view_mode == "active":
            return f"[dim]{n_active} active[/dim]"
        if view_mode == "archived":
            return f"[dim]{n_archived} archived[/dim]"
        return f"[dim]{n_active} active · {n_archived} archived[/dim]"

    def _footer_text() -> str:
        if search_active:
            return "type to filter   Enter apply   Esc cancel   Backspace delete"
        cur_visible = _visible_rows_list()
        cur = cur_visible[_selected_pos(cur_visible)] if cur_visible else None
        a_hint = "a unarchive" if (cur and cur["archived"]) else "a archive"
        find_hint = "^F edit search" if search_query else "^F find"
        parts = [
            "←→/↑↓ navigate", "Enter load", "p preview", "d delete",
            a_hint, f"Tab view: {view_mode}", find_hint,
        ]
        action = last_action
        if action is not None:
            remaining = action["deadline"] - time.time()
            if remaining > 0:
                parts.append(f"u undo ({int(remaining) + 1}s)")
        parts.append("q cancel")
        return "   ".join(parts)

    def _build_preview_lines(sid: str, width: int) -> list[Text]:
        meta = sessions.get(sid, {})
        hp_str = meta.get("history_file")
        if not hp_str:
            return [Text("(no history file)", style="dim")]
        hp = Path(hp_str)
        if not hp.exists():
            return [Text("(history file missing)", style="dim")]
        history = load_history(hp)
        if not history:
            return [Text("(empty conversation)", style="dim")]
        return _history_visual_lines(history, width) or [Text("(empty conversation)", style="dim")]

    def _preview_lines(sid: str, width: int) -> list[Text]:
        cache_key = (sid, width)
        if cache_key not in preview_cache:
            preview_cache[cache_key] = _build_preview_lines(sid, width)
        return preview_cache[cache_key]

    def _render_preview() -> Panel:
        nonlocal preview_offset
        term_h = console.size.height
        term_w = console.size.width
        panel_width = max(1, term_w)
        inner_width = max(1, panel_width - 4)
        show_footer = term_h >= 6
        # Header (1 row) + footer (2 rows: blank + controls) inside the panel border (2 rows).
        body_rows = max(1, term_h - 2 - 1 - (2 if show_footer else 0))

        sid = preview_sid or ""
        all_lines = _preview_lines(sid, inner_width)
        total = len(all_lines)
        max_off = max(0, total - body_rows)
        if preview_offset > max_off:
            preview_offset = max_off
        if preview_offset < 0:
            preview_offset = 0
        start = preview_offset
        end = min(total, start + body_rows)

        meta = sessions.get(sid, {})
        short_id = _short_session_id(sid) if sid else ""
        label = meta.get("label", "")

        parts: list = []
        parts.append(
            Text(
                _truncate(f"  {short_id}  {label}", inner_width),
                style="dim",
                no_wrap=True,
                overflow="crop",
            )
        )
        for i in range(start, end):
            parts.append(all_lines[i])
        if total == 0:
            parts.append(Text(_truncate("  (empty)", inner_width), style="dim"))

        if show_footer:
            position = f"{start + 1}–{end} of {total}" if total else "0"
            parts.append(Text(""))
            parts.append(
                Text(
                    _truncate(
                        f"↑↓ scroll   ←→ session   Enter load   p/Esc back   ·   {position}",
                        inner_width,
                    ),
                    style="dim italic",
                    no_wrap=True,
                    overflow="crop",
                )
            )

        return Panel(
            Group(*parts),
            title="[bold bright_white]jarv ▸ preview[/bold bright_white]",
            title_align="left",
            subtitle=f"[dim]{short_id}[/dim]" if short_id else None,
            subtitle_align="right",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            width=panel_width,
        )

    def _render() -> Panel:
        if preview_sid is not None:
            return _render_preview()
        nonlocal offset
        term_w = console.size.width
        term_h = console.size.height
        panel_width = max(1, term_w)
        inner_width = max(1, panel_width - 4)
        show_footer = term_h >= 6

        visible = _visible_rows_list()
        n = len(visible)
        sel = _selected_pos(visible)

        status: Text | None = None
        cur = visible[sel] if visible else None
        if arm_delete_sid and cur and cur["sid"] == arm_delete_sid:
            prompt = (
                f"Delete {cur['short_id']} permanently? "
                "Press d again to confirm · any other key cancels"
            )
            status = Text(_truncate(prompt, inner_width), style="bold red", no_wrap=True, overflow="crop")
        elif flash is not None:
            msg, style = flash
            status = Text(_truncate(msg, inner_width), style=style, no_wrap=True, overflow="crop")

        has_search = bool(search_active or search_query)
        mv = _content_rows(
            term_h,
            has_status=status is not None,
            show_footer=show_footer,
            has_search=has_search,
        )
        offset = _clamp_offset(sel, offset, mv, n)
        start = offset
        end = min(n, offset + mv)

        def _search_bar() -> Text:
            cursor = "▌" if search_active else ""
            shown = search_query + cursor
            prefix = " › " if search_active else "   "
            label_style = "bold cyan" if search_active else "cyan"
            value_style = "bold cyan" if search_active else "bold"
            placeholder_style = "bold cyan" if search_active else "dim italic"
            line = Text(no_wrap=True, overflow="crop")
            line.append(prefix, style="bold cyan" if search_active else "")
            line.append("find: ", style=label_style)
            avail = max(0, inner_width - len(prefix) - 6)
            if shown:
                line.append(_truncate(shown, avail), style=value_style)
            else:
                line.append(_truncate("(type to filter transcripts)", avail), style=placeholder_style)
            return line

        parts: list = []
        if n == 0:
            if has_search:
                parts.append(_search_bar())
            empty_msg = (
                f"  (no sessions match \"{search_query}\")"
                if search_query
                else "  (no sessions in this view)"
            )
            parts.append(Text(_truncate(empty_msg, inner_width), style="dim"))
            if status is not None:
                parts.append(status)
            if show_footer:
                parts.append(Text(""))
                parts.append(
                    Text(
                        _truncate(_footer_text(), inner_width),
                        style="dim italic",
                        no_wrap=True,
                        overflow="crop",
                    )
                )
            return Panel(
                Group(*parts),
                title="[bold bright_white]jarv ▸ sessions[/bold bright_white]",
                title_align="left",
                subtitle=_subtitle(),
                subtitle_align="right",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(0, 1),
                width=panel_width,
            )

        parts.append(
            Text(
                _truncate(f"  showing {start + 1}–{end} of {n}", inner_width),
                style="dim",
                no_wrap=True,
                overflow="crop",
            )
        )
        if has_search:
            parts.append(_search_bar())

        for i in range(start, end):
            r = visible[i]
            is_sel = (i == sel) and not search_active
            is_armed = is_sel and arm_delete_sid == r["sid"]
            t = Text(no_wrap=True, overflow="ellipsis")
            prefix = " › " if is_sel else "   "
            if r["is_current"]:
                marker = "●  "
            elif r["archived"]:
                marker = "⌫  "
            else:
                marker = "   "
            remaining = inner_width - len(prefix) - len(marker)
            id_width = max(0, min(24, remaining))
            remaining -= id_width
            time_width = max(0, min(12, remaining))
            remaining -= time_width
            snippet_width = max(0, remaining)

            if is_armed:
                prefix_style = "bold red"
            elif is_sel:
                prefix_style = "bold cyan"
            else:
                prefix_style = ""
            t.append(_truncate(prefix, inner_width), style=prefix_style)

            if inner_width > len(prefix):
                if is_armed:
                    marker_style = "bold red"
                elif r["is_current"]:
                    marker_style = "green"
                elif r["archived"]:
                    marker_style = "dim"
                else:
                    marker_style = ""
                t.append(_truncate(marker, inner_width - len(prefix)), style=marker_style)

            if id_width:
                short_id = _truncate(r["short_id"], id_width)
                if is_armed:
                    id_style = "bold red"
                elif is_sel:
                    id_style = "bold cyan"
                elif r["archived"]:
                    id_style = "dim cyan"
                else:
                    id_style = "cyan"
                t.append(f"{short_id:<{id_width}}", style=id_style)

            if time_width:
                time_str = _truncate(r["time_str"], time_width)
                if is_armed:
                    time_style = "bold red"
                elif is_sel:
                    time_style = "bold"
                else:
                    time_style = "dim"
                t.append(f"{time_str:<{time_width}}", style=time_style)

            snip = r["snippet"] or "no messages"
            if snippet_width:
                if is_armed:
                    snip_style = "bold red"
                elif is_sel:
                    snip_style = "bold" if not r["archived"] else "dim strike"
                elif r["archived"]:
                    snip_style = "dim strike"
                else:
                    snip_style = "dim"
                t.append(_truncate(snip, snippet_width), style=snip_style)
            parts.append(t)

        if status is not None:
            parts.append(status)

        if show_footer:
            parts.append(Text(""))
            parts.append(
                Text(
                    _truncate(_footer_text(), inner_width),
                    style="dim italic",
                    no_wrap=True,
                    overflow="crop",
                )
            )

        return Panel(
            Group(*parts),
            title="[bold bright_white]jarv \u25b8 sessions[/bold bright_white]",
            title_align="left",
            subtitle=_subtitle(),
            subtitle_align="right",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            width=panel_width,
        )

    def _finalize_action(action: dict) -> None:
        """Apply the irreversible part of an action that's leaving its window."""
        if action["kind"] == "did_delete":
            hp_str = action.get("history_path")
            if hp_str:
                delete_session_files(Path(hp_str))

    def _expire_action(action: dict) -> None:
        nonlocal last_action
        with undo_lock:
            if last_action is not action:
                return
            last_action = None
        _finalize_action(action)

    def _commit_pending() -> None:
        nonlocal last_action
        with undo_lock:
            action = last_action
            last_action = None
        if action is None:
            return
        t = action.get("timer")
        if t is not None:
            t.cancel()
        _finalize_action(action)

    def _start_undo(action: dict) -> None:
        nonlocal last_action
        _commit_pending()
        action["deadline"] = time.time() + UNDO_WINDOW
        timer = threading.Timer(UNDO_WINDOW, _expire_action, args=(action,))
        timer.daemon = True
        action["timer"] = timer
        with undo_lock:
            last_action = action
        timer.start()

    def _take_last_action() -> dict | None:
        """Atomically grab and clear the current undoable action if still valid."""
        nonlocal last_action
        with undo_lock:
            action = last_action
            if action is None:
                return None
            if time.time() >= action["deadline"]:
                # Already expired — let the timer handle finalization.
                return None
            last_action = None
        t = action.get("timer")
        if t is not None:
            t.cancel()
        return action

    def _do_undo() -> tuple[tuple[str, str], str | None] | None:
        """Returns ((flash_msg, flash_style), restored_sid) or None."""
        action = _take_last_action()
        if action is None:
            return None
        kind = action["kind"]
        if kind == "did_archive":
            sid = action["sid"]
            row = next((r for r in rows if r["sid"] == sid), None)
            if row is None:
                return (("○ session no longer exists", "dim"), None)
            meta = sessions.get(sid, {})
            hp_str = meta.get("history_file")
            hp = Path(hp_str) if hp_str else None
            restored = unarchive_session_files(hp, sid) if hp else None
            if restored is not None:
                meta["history_file"] = str(restored)
            meta.pop("archived", None)
            meta.pop("archived_at", None)
            row["archived"] = False
            save_sessions(data)
            return ((f"↺ restored {row['short_id']}", "green"), sid)
        if kind == "did_unarchive":
            sid = action["sid"]
            row = next((r for r in rows if r["sid"] == sid), None)
            if row is None:
                return (("○ session no longer exists", "dim"), None)
            meta = sessions.get(sid, {})
            hp_str = meta.get("history_file")
            hp = Path(hp_str) if hp_str else None
            archived_path = archive_session_files(hp) if hp else None
            if archived_path is None:
                return (("○ couldn't re-archive", "dim"), None)
            meta["history_file"] = str(archived_path)
            meta["archived"] = True
            meta["archived_at"] = isoformat_utc(utc_now())
            for term_id, mapped_sid in list(terminals.items()):
                if mapped_sid == sid:
                    terminals.pop(term_id)
            row["archived"] = True
            row["is_current"] = False
            save_sessions(data)
            return ((f"↺ archived {row['short_id']}", "cyan"), sid)
        if kind == "did_delete":
            sid = action["sid"]
            snapshot_row = action["row"]
            snapshot_meta = action["meta"]
            row_index = action.get("row_index", len(rows))
            removed_terminals = action.get("removed_terminals", [])
            sessions[sid] = snapshot_meta
            for term_id in removed_terminals:
                terminals[term_id] = sid
            if 0 <= row_index <= len(rows):
                rows.insert(row_index, snapshot_row)
            else:
                rows.append(snapshot_row)
            save_sessions(data)
            return ((f"↺ restored {snapshot_row['short_id']}", "green"), sid)
        return None

    loaded_row: dict | None = None
    auto_restored = False
    with Live(
        get_renderable=_render,
        console=console,
        screen=True,
        auto_refresh=False,
        transient=False,
        vertical_overflow="crop",
    ) as live:
        while True:
            live.refresh()
            try:
                key = _read_key(text_mode=search_active and preview_sid is None)
            except KeyboardInterrupt:
                break

            # Search-input mode intercepts most keys (only outside preview).
            if search_active and preview_sid is None:
                if key == "ESC":
                    search_active = False
                    search_query = ""
                    offset = 0
                elif key in ("ENTER", "DOWN"):
                    # Exit search mode, drop focus into the filtered list.
                    search_active = False
                    visible_now = _visible_rows_list()
                    if visible_now and not any(r["sid"] == selected_sid for r in visible_now):
                        selected_sid = visible_now[0]["sid"]
                    offset = 0
                elif key == "BACKSPACE":
                    if search_query:
                        search_query = search_query[:-1]
                        offset = 0
                elif key == "CTRL_F":
                    search_active = False
                elif isinstance(key, str) and len(key) == 1 and key.isprintable():
                    search_query += key
                    offset = 0
                    visible_now = _visible_rows_list()
                    if visible_now and not any(r["sid"] == selected_sid for r in visible_now):
                        selected_sid = visible_now[0]["sid"]
                # All other keys (UP, PAGEUP/DOWN, HOME/END, TAB, a, d, p, …)
                # are swallowed while typing the query.
                continue

            # Preview mode intercepts most keys.
            if preview_sid is not None:
                if key in ("p", "ESC"):
                    preview_sid = None
                    preview_offset = 0
                elif key in ("LEFT", "RIGHT"):
                    visible_now = _visible_rows_list()
                    if visible_now:
                        pos = next(
                            (i for i, r in enumerate(visible_now) if r["sid"] == preview_sid),
                            _selected_pos(visible_now),
                        )
                        delta = -1 if key == "LEFT" else 1
                        pos = max(0, min(len(visible_now) - 1, pos + delta))
                        preview_sid = visible_now[pos]["sid"]
                        selected_sid = preview_sid
                        preview_offset = 0
                elif key == "UP":
                    preview_offset = max(0, preview_offset - 1)
                elif key == "DOWN":
                    preview_offset += 1
                elif key == "PAGEUP":
                    preview_offset = max(0, preview_offset - _max_vis())
                elif key == "PAGEDOWN":
                    preview_offset += _max_vis()
                elif key == "HOME":
                    preview_offset = 0
                elif key == "END":
                    preview_offset = max(0, len(_preview_lines(preview_sid, max(1, console.size.width - 4))) - 1)
                elif key == "ENTER":
                    row = next((r for r in rows if r["sid"] == preview_sid), None)
                    if row is not None:
                        if row["archived"]:
                            meta = sessions.get(row["sid"], {})
                            hp_str = meta.get("history_file")
                            if hp_str:
                                restored = unarchive_session_files(Path(hp_str), row["sid"])
                                if restored is not None:
                                    meta["history_file"] = str(restored)
                                    meta.pop("archived", None)
                                    meta.pop("archived_at", None)
                                    row["archived"] = False
                                    save_sessions(data)
                                    auto_restored = True
                        set_terminal_session(row["sid"])
                        loaded_row = row
                        break
                continue

            if key != "d":
                arm_delete_sid = None
            flash = None

            visible = _visible_rows_list()
            n_vis = len(visible)
            sel = _selected_pos(visible) if visible else 0
            cur = visible[sel] if visible else None

            if key in ("UP", "LEFT"):
                if visible:
                    selected_sid = visible[max(0, sel - 1)]["sid"]
                ghost_sid = None
            elif key in ("DOWN", "RIGHT"):
                if visible:
                    selected_sid = visible[min(n_vis - 1, sel + 1)]["sid"]
                ghost_sid = None
            elif key == "HOME":
                if visible:
                    selected_sid = visible[0]["sid"]
                ghost_sid = None
            elif key == "END":
                if visible:
                    selected_sid = visible[n_vis - 1]["sid"]
                ghost_sid = None
            elif key == "PAGEUP":
                if visible:
                    selected_sid = visible[max(0, sel - _max_vis())]["sid"]
                ghost_sid = None
            elif key == "PAGEDOWN":
                if visible:
                    selected_sid = visible[min(n_vis - 1, sel + _max_vis())]["sid"]
                ghost_sid = None
            elif key == "ENTER":
                if cur is None:
                    continue
                if cur["archived"]:
                    meta = sessions.get(cur["sid"], {})
                    hp_str = meta.get("history_file")
                    if hp_str:
                        restored = unarchive_session_files(Path(hp_str), cur["sid"])
                        if restored is not None:
                            meta["history_file"] = str(restored)
                            meta.pop("archived", None)
                            meta.pop("archived_at", None)
                            cur["archived"] = False
                            save_sessions(data)
                            auto_restored = True
                set_terminal_session(cur["sid"])
                loaded_row = cur
                break
            elif key == "ESC":
                if search_query:
                    search_query = ""
                    offset = 0
                    continue
                break
            elif key == "CTRL_F":
                search_active = True
                continue
            elif key == "TAB":
                view_mode = {"active": "all", "all": "archived", "archived": "active"}[view_mode]
                offset = 0
                ghost_sid = None
            elif key == "p":
                if cur is not None:
                    preview_sid = cur["sid"]
                    preview_offset = 0
            elif key == "a":
                if cur is None:
                    continue
                sid = cur["sid"]
                meta = sessions.get(sid, {})
                hp_str = meta.get("history_file")
                hp = Path(hp_str) if hp_str else None
                if cur["archived"]:
                    restored = unarchive_session_files(hp, sid) if hp else None
                    if restored is not None:
                        meta["history_file"] = str(restored)
                        meta.pop("archived", None)
                        meta.pop("archived_at", None)
                        cur["archived"] = False
                        save_sessions(data)
                        flash = (f"✓ restored {cur['short_id']}", "green")
                        ghost_sid = sid if view_mode == "archived" else None
                        _start_undo({"kind": "did_unarchive", "sid": sid})
                    else:
                        meta.pop("archived", None)
                        meta.pop("archived_at", None)
                        cur["archived"] = False
                        save_sessions(data)
                        flash = (f"○ archive missing for {cur['short_id']} — marked active", "dim")
                        ghost_sid = sid if view_mode == "archived" else None
                else:
                    archived_path = archive_session_files(hp) if hp else None
                    if archived_path is not None:
                        meta["history_file"] = str(archived_path)
                        meta["archived"] = True
                        meta["archived_at"] = isoformat_utc(utc_now())
                        for term_id, mapped_sid in list(terminals.items()):
                            if mapped_sid == sid:
                                terminals.pop(term_id)
                        cur["archived"] = True
                        cur["is_current"] = False
                        save_sessions(data)
                        flash = (f"✓ archived {cur['short_id']}", "cyan")
                        ghost_sid = sid if view_mode == "active" else None
                        _start_undo({"kind": "did_archive", "sid": sid})
                    else:
                        flash = (f"○ nothing to archive for {cur['short_id']}", "dim")
            elif key == "d":
                if cur is None:
                    continue
                sid = cur["sid"]
                if arm_delete_sid == sid:
                    meta = sessions.get(sid, {})
                    hp_str = meta.get("history_file")
                    snapshot_meta = dict(meta)
                    snapshot_row = dict(cur)
                    row_index = next(
                        (i for i, r in enumerate(rows) if r["sid"] == sid), len(rows)
                    )
                    removed_terminals: list[str] = []
                    for term_id, mapped_sid in list(terminals.items()):
                        if mapped_sid == sid:
                            removed_terminals.append(term_id)
                            terminals.pop(term_id)
                    sessions.pop(sid, None)
                    rows[:] = [r for r in rows if r["sid"] != sid]
                    save_sessions(data)
                    new_visible = _visible_rows_list()
                    if new_visible:
                        new_sel = min(sel, len(new_visible) - 1)
                        selected_sid = new_visible[new_sel]["sid"]
                    else:
                        selected_sid = None
                    flash = (f"✓ deleted {cur['short_id']}", "red")
                    arm_delete_sid = None
                    _start_undo({
                        "kind": "did_delete",
                        "sid": sid,
                        "row": snapshot_row,
                        "meta": snapshot_meta,
                        "row_index": row_index,
                        "removed_terminals": removed_terminals,
                        "history_path": hp_str,
                    })
                else:
                    arm_delete_sid = sid
            elif key == "u":
                result = _do_undo()
                if result is not None:
                    flash, restored_sid = result
                    if restored_sid is not None:
                        selected_sid = restored_sid
                        ghost_sid = restored_sid

    prefetch_stop.set()
    _commit_pending()

    if loaded_row is not None:
        label = sessions.get(loaded_row["sid"], {}).get("label", loaded_row["sid"])
        prefix = "Restored & loaded" if auto_restored else "Loaded"
        console.print(
            f"[bold green]✓[/bold green] [green]{prefix}[/green] "
            f"[bold cyan]{loaded_row['short_id']}[/bold cyan] [dim]({label})[/dim]"
        )
        return
    console.print("[dim]○ Cancelled.[/dim]")



def cmd_history() -> None:
    session_context = prepare_session_context()
    history = load_history(session_context.history_file)
    if not history:
        console.print("[dim]○ No history yet.[/dim]")
        return

    exchanges = sum(1 for m in history if isinstance(m, dict) and m.get("role") == "user")

    if not sys.stdin.isatty() or not console.is_terminal:
        parts: list = [section_rule("conversation"), Text("")]
        for m in history:
            role = m.get("role")
            if role == "user":
                line = Text()
                line.append("▌ ", style="bold cyan")
                line.append("You", style="bold cyan")
                parts.append(line)
                parts.append(Text(f"  {m.get('content', '')}"))
                parts.append(Text(""))
            elif role == "assistant":
                content = m.get("content", "")
                if content:
                    line = Text()
                    line.append("▌ ", style="bold green")
                    line.append("Jarv", style="bold green")
                    parts.append(line)
                    parts.append(Markdown(flatten_headings(content)))
                    parts.append(Text(""))
        console.print(jarv_panel(Group(*parts), title="history", subtitle=f"{exchanges} exchange(s)"))
        return

    offset = 0

    def _body_rows() -> int:
        term_h = console.size.height
        return max(1, term_h - 2 - 1 - 2)  # panel border + header + footer

    def _render() -> Panel:
        nonlocal offset
        term_w = console.size.width
        panel_width = max(1, term_w)
        show_footer = console.size.height >= 6
        body = _body_rows()
        inner_width = max(1, panel_width - 4)
        lines = _history_visual_lines(history, inner_width)
        total = len(lines)
        max_off = max(0, total - body)
        offset = max(0, min(offset, max_off))
        start = offset
        end = min(total, start + body)

        parts: list = []
        for i in range(start, end):
            parts.append(lines[i])
        if not lines:
            parts.append(Text("  (empty)", style="dim"))

        if show_footer:
            position = f"{start + 1}–{end} of {total}" if total else "0"
            parts.append(Text(""))
            parts.append(
                Text(
                    f"↑↓ scroll   PgUp/PgDn   Home/End   q exit   ·   {position}",
                    style="dim italic",
                    no_wrap=True,
                    overflow="crop",
                )
            )

        return Panel(
            Group(*parts),
            title="[bold bright_white]jarv ▸ history[/bold bright_white]",
            title_align="left",
            subtitle=f"[dim]{exchanges} exchange(s)[/dim]",
            subtitle_align="right",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            width=panel_width,
        )

    with Live(
        get_renderable=_render,
        console=console,
        screen=True,
        auto_refresh=False,
        transient=False,
        vertical_overflow="crop",
    ) as live:
        while True:
            live.refresh()
            try:
                key = _read_key()
            except KeyboardInterrupt:
                break
            total = len(_history_visual_lines(history, max(1, console.size.width - 4)))
            page = max(1, _body_rows() - 1)
            max_off = max(0, total - _body_rows())
            if key == "ESC":
                break
            elif key == "UP":
                offset = max(0, offset - 1)
            elif key == "DOWN":
                offset = min(max_off, offset + 1)
            elif key == "PAGEUP":
                offset = max(0, offset - page)
            elif key == "PAGEDOWN":
                offset = min(max_off, offset + page)
            elif key == "HOME":
                offset = 0
            elif key == "END":
                offset = max_off


_BREAKDOWN_KEYS = ("system", "tools", "history", "tool_io", "reasoning")
_BREAKDOWN_LABELS = {
    "system": "System",
    "tools": "Tools",
    "history": "History",
    "tool_io": "Tool I/O",
    "reasoning": "Reasoning",
}
_BREAKDOWN_COLORS = {
    "system": "white",
    "tools": "yellow",
    "history": "cyan",
    "tool_io": "magenta",
    "reasoning": "green",
}


_BAR_FILL_CHARS = " ▏▎▍▌▋▊▉█"


def _smooth_bar(percent: float | None, width: int = 36, color: str = "cyan") -> Text:
    """Render a smooth horizontal bar with sub-cell precision."""
    bar = Text()
    if percent is None:
        bar.append("─" * width, style="bright_black")
        return bar
    pct = max(0.0, min(percent, 100.0)) / 100
    total_eighths = pct * width * 8
    full = int(total_eighths // 8)
    remainder = int(total_eighths - full * 8)
    if full > width:
        full = width
        remainder = 0
    bar.append("█" * full, style=color)
    if full < width:
        if remainder:
            bar.append(_BAR_FILL_CHARS[remainder], style=color)
            empty = width - full - 1
        else:
            empty = width - full
        if empty > 0:
            bar.append("─" * empty, style="bright_black")
    return bar


def _fill_color(percent: float | None) -> str:
    if percent is None:
        return "bright_black"
    if percent >= 90:
        return "bright_red"
    if percent >= 70:
        return "yellow"
    if percent >= 40:
        return "cyan"
    return "green"


def _breakdown_bar(breakdown: dict, width: int = 48) -> Text:
    total = sum(int(breakdown.get(k, 0)) for k in _BREAKDOWN_KEYS)
    if total == 0:
        return Text("─" * width, style="bright_black")
    bar = Text()
    used = 0
    non_zero = [k for k in _BREAKDOWN_KEYS if int(breakdown.get(k, 0)) > 0]
    for i, key in enumerate(non_zero):
        count = int(breakdown.get(key, 0))
        is_last = i == len(non_zero) - 1
        if is_last:
            chars = width - used
        else:
            chars = max(1, round((count / total) * width))
            chars = min(chars, width - used - (len(non_zero) - i - 1))
        if chars > 0:
            bar.append("█" * chars, style=_BREAKDOWN_COLORS[key])
            used += chars
    if used < width:
        bar.append("─" * (width - used), style="bright_black")
    return bar


def _breakdown_section(breakdown: dict) -> Group:
    total = sum(int(breakdown.get(k, 0)) for k in _BREAKDOWN_KEYS)
    bar = _breakdown_bar(breakdown)

    bd_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    bd_table.add_column(no_wrap=True, width=1)
    bd_table.add_column(no_wrap=True)
    bd_table.add_column(justify="right", no_wrap=True)
    bd_table.add_column(justify="right", style="dim", no_wrap=True, width=5)

    for key in _BREAKDOWN_KEYS:
        count = int(breakdown.get(key, 0))
        pct = f"{round(count / total * 100)}%" if total > 0 else "—"
        bd_table.add_row(
            Text("●", style=_BREAKDOWN_COLORS[key]),
            Text(_BREAKDOWN_LABELS[key]),
            Text(format_int(count), style="bold"),
            Text(pct, style="dim"),
        )
    return Group(bar, Text(""), bd_table)


def _plural(value: int, singular: str, plural: str | None = None) -> str:
    word = singular if value == 1 else (plural or f"{singular}s")
    return f"{value:,} {word}"


def _context_usage_renderable(last_root: dict | None) -> Text:
    if not isinstance(last_root, dict):
        return Text("Unknown until a root request is recorded", style="dim")
    model = str(last_root.get("model") or "")
    context_window = known_context_window(model)
    input_tokens = int(last_root.get("input_tokens") or 0)
    if context_window is None:
        return Text("Unknown for this model", style="dim")
    percent = (input_tokens / context_window) * 100
    color = _fill_color(percent)
    line = Text()
    line.append(f"{percent:5.1f}% full", style=f"bold {color}")
    line.append("  ")
    line.append_text(_smooth_bar(percent, width=32, color=color))
    line.append("  ")
    line.append(f"({format_int(input_tokens)} / {format_int(context_window)})", style="dim")
    return line


def _estimated_total_cost(usage: dict) -> float | None:
    models = usage.get("models") if isinstance(usage.get("models"), dict) else {}
    total = 0.0
    saw_model = False
    for model, bucket in models.items():
        if not isinstance(bucket, dict):
            continue
        if int(bucket.get("request_count") or 0) <= 0:
            continue
        saw_model = True
        estimate = estimate_token_cost_usd(bucket, str(model))
        if estimate is None:
            return None
        total += estimate
    if saw_model:
        return total
    return None


def cmd_usage() -> None:
    ctx = prepare_session_context()
    usage_path = usage_file_for(ctx.history_file)
    usage = load_usage(usage_path, ctx.session_id)
    totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else {}
    request_count = int(totals.get("request_count") or 0)
    if request_count <= 0:
        console.print("[dim]No token usage recorded for this session yet.[/dim]")
        return

    history = load_history(ctx.history_file)
    exchanges = sum(1 for item in history if isinstance(item, dict) and item.get("role") == "user")
    last_request = usage.get("last_request") if isinstance(usage.get("last_request"), dict) else None
    last_root = usage.get("last_root_request") if isinstance(usage.get("last_root_request"), dict) else None
    model = str((last_request or {}).get("model") or "unknown")
    estimated_cost = _estimated_total_cost(usage)

    root_model = str((last_root or {}).get("model") or "unknown")

    context_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    context_table.add_column("Field", style="dim", no_wrap=True)
    context_table.add_column("Value", no_wrap=False)
    context_table.add_row("Latest root model", Text(root_model, style="bold magenta"))
    context_table.add_row("Context usage", _context_usage_renderable(last_root))

    reasoning_tokens = int(totals.get("reasoning_output_tokens") or 0)
    token_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    token_table.add_column("Field", style="dim", no_wrap=True)
    token_table.add_column("Value", no_wrap=False)
    token_table.add_row("Last model", Text(model, style="bold magenta"))
    token_table.add_row("Messages", Text(_plural(exchanges, "exchange")))
    token_table.add_row("Requests", Text(_plural(request_count, "request")))
    token_table.add_row("Input tokens", Text(format_int(totals.get("input_tokens"))))
    token_table.add_row("Cached input", Text(format_int(totals.get("cached_input_tokens")), style="cyan"))
    token_table.add_row("New input", Text(format_int(totals.get("uncached_input_tokens"))))
    token_table.add_row("Output tokens", Text(format_int(totals.get("output_tokens"))))
    if reasoning_tokens:
        token_table.add_row("Reasoning output", Text(format_int(reasoning_tokens), style="green"))
    token_table.add_row("Total tokens", Text(format_int(totals.get("total_tokens")), style="bold"))
    token_table.add_row("Estimated cost", Text(format_cost(estimated_cost), style="bold green"))
    if last_request is not None:
        last_line = Text()
        last_line.append(format_int(last_request.get("input_tokens")), style="bold")
        last_line.append(" in ", style="dim")
        last_line.append("(", style="dim")
        last_line.append(format_int(last_request.get("cached_input_tokens")), style="cyan")
        last_line.append(" cached", style="dim")
        last_line.append(") · ", style="dim")
        last_line.append(format_int(last_request.get("output_tokens")), style="bold")
        last_line.append(" out", style="dim")
        token_table.add_row("Last request", last_line)

    breakdown = (last_root or {}).get("context_breakdown")
    panel_parts: list = [
        section_rule("session overview"),
        Text(""),
        context_table,
    ]
    if isinstance(breakdown, dict) and any(breakdown.get(k, 0) for k in _BREAKDOWN_KEYS):
        panel_parts += [
            Text(""),
            section_rule("context breakdown [dim](estimated)[/dim]"),
            Text(""),
            _breakdown_section(breakdown),
        ]
    panel_parts += [
        Text(""),
        section_rule("token totals"),
        Text(""),
        token_table,
    ]

    console.print(jarv_panel(Group(*panel_parts), title="usage", subtitle=str(usage_path)))


def cmd_config() -> None:
    config = load_config()
    table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for k, v in config.items():
        if k == "api_key" and v:
            val = Text("***", style="dim")
        elif isinstance(v, bool):
            val = Text(repr(v), style="bold magenta")
        elif isinstance(v, (int, float)):
            val = Text(repr(v), style="bold yellow")
        elif isinstance(v, str):
            val = Text(repr(v), style="green")
        else:
            val = Text(repr(v))
        table.add_row(k, val)

    body = Group(
        section_rule("settings"),
        Text(""),
        table,
    )
    console.print(jarv_panel(body, title="config", subtitle=str(CONFIG_FILE)))


def _parse_count(args: list, default: int = 1) -> int:
    if not args:
        return default
    try:
        return max(1, int(args[0]))
    except ValueError:
        return default


def _first_user_text(frame: list) -> str:
    for item in frame:
        if isinstance(item, dict) and item.get("role") == "user":
            return str(item.get("content", "")).strip().replace("\n", " ")[:80]
    return "(no user message)"


def cmd_undo(args: list) -> None:
    n = _parse_count(args)
    ctx = prepare_session_context()
    history = load_history(ctx.history_file)
    redo_path = redo_file_for(ctx.history_file)
    stack = load_redo_stack(redo_path)

    undone: list[list] = []
    for _ in range(n):
        history, frame = split_last_exchange(history)
        if not frame:
            break
        undone.append(frame)
        stack.append(frame)

    if not undone:
        console.print("[dim]○ Nothing to undo.[/dim]")
        return

    save_history(history, ctx.history_file)
    save_redo_stack(stack, redo_path)

    if len(undone) == 1:
        text = _first_user_text(undone[0])
        console.print(f"[bold yellow]↶[/bold yellow] [bold]Unsent[/bold] [cyan]{text!r}[/cyan]")
        console.print(f"[dim]  Removed {len(undone[0])} item(s). Run [bold]/redo[/bold] to put it back.[/dim]")
    else:
        console.print(f"[bold yellow]↶[/bold yellow] [bold]Unsent {len(undone)} exchanges:[/bold]")
        for i, frame in enumerate(undone, 1):
            console.print(f"  [dim]{i}.[/dim] [cyan]{_first_user_text(frame)!r}[/cyan]")
        console.print(f"[dim]  Run [bold]/redo {len(undone)}[/bold] to put them back.[/dim]")


def cmd_redo(args: list) -> None:
    n = _parse_count(args)
    ctx = prepare_session_context()
    history = load_history(ctx.history_file)
    redo_path = redo_file_for(ctx.history_file)
    stack = load_redo_stack(redo_path)

    restored: list[list] = []
    for _ in range(n):
        if not stack:
            break
        frame = stack.pop()
        history.extend(frame)
        restored.append(frame)

    if not restored:
        console.print("[dim]○ Nothing to redo.[/dim]")
        return

    save_history(history, ctx.history_file)
    save_redo_stack(stack, redo_path)

    if len(restored) == 1:
        text = _first_user_text(restored[0])
        console.print(f"[bold cyan]↷[/bold cyan] [bold]Restored[/bold] [cyan]{text!r}[/cyan]")
    else:
        console.print(f"[bold cyan]↷[/bold cyan] [bold]Restored {len(restored)} exchange(s).[/bold]")
