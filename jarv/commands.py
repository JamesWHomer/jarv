import json
import subprocess
import sys
import urllib.request

from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

from . import __version__
from .config import CONFIG_DIR, CONFIG_FILE, DEFAULT_CONFIG, load_config, save_config, validate_config
from .display import console, flatten_headings
from .history import (
    SESSIONS_DIR,
    SESSIONS_FILE,
    artifact_file_for,
    forget_current_session,
    load_history,
    load_sessions,
    prepare_session_context,
    set_terminal_session,
    short_hash,
    utc_now,
)

ARCHIVE_DIR = CONFIG_DIR / "archive"

GITHUB_REPO = "JamesWHomer/jarv"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/commits/main"
INSTALL_URL = f"https://github.com/{GITHUB_REPO}.git"
SHA_FILE = CONFIG_DIR / "last_sha.txt"


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
        console.print("[red]Usage:[/red] jarv /set <key> <value>")
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
        console.print("[red]Usage:[/red] jarv /unset <key>")
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
    cmd_table.add_row("jarv /set <key> <value>", "Set a config value")
    cmd_table.add_row("jarv /unset <key>", "Reset a config key to its default")
    cmd_table.add_row("jarv /clear", "Archive this terminal's session and start a fresh one")
    cmd_table.add_row("jarv /load", "Load the most recently used session into this terminal")
    cmd_table.add_row("jarv /load <id>", "Load a specific session into this terminal")
    cmd_table.add_row("jarv /history", "Show recent conversation history")
    cmd_table.add_row("jarv /config", "Show current settings")
    cmd_table.add_row("jarv /update", "Update jarv to the latest version")
    cmd_table.add_row("jarv /cleanup", "Delete orphaned session files from ~/.jarv/sessions/")
    cmd_table.add_row("jarv /about", "Show detailed information about jarv")
    cmd_table.add_row("jarv /help", "Show this help")

    key_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    key_table.add_column(style="bold yellow", no_wrap=True)
    key_table.add_column(style="dim")
    key_table.add_row("api_key", "OpenAI API key")
    key_table.add_row("model", "Model name (default: gpt-5.4-mini)")
    key_table.add_row("reasoning_effort", "Reasoning effort value (empty to disable)")
    key_table.add_row("max_history", "Number of messages to keep as context")
    key_table.add_row("command_timeout", "Seconds before a shell command is killed")
    key_table.add_row("system_prompt", "System prompt sent to the model")
    key_table.add_row("check_updates", "Check for updates on each run (true/false)")

    console.print(Panel(cmd_table, title="[bold]jarv[/bold]", border_style="bright_black", padding=(1, 2)))
    console.print()
    console.print("[bold]Config keys[/bold]")
    console.print(key_table)
    console.print(f"\n[dim]Config:   {CONFIG_FILE}[/dim]")
    console.print(f"[dim]Sessions: {SESSIONS_FILE}[/dim]")


def print_about() -> None:
    about = f"""# jarv

jarv is a command-line AI assistant powered by OpenAI.

## Basic usage

- `jarv` - Start heads-up mode so you can keep sending prompts without rerunning the command.
- `jarv <question>` - Ask jarv anything. Your words after `jarv` are sent as the user message.
- `jarv /help` - Show the short command overview. (`jarv help` also works as a permanent alias.)
- `jarv /about` - Show this detailed overview.
- `jarv /config` - Show current settings. The API key is masked.
- `jarv /set <key> <value>` - Set a config value. Values like `true`, `false`, integers, and floats are coerced.
- `jarv /unset <key>` - Reset a default config key, or remove a custom key.
- `jarv /history` - Show recent user and assistant messages.
- `jarv /clear` - Archive this terminal's session and start a fresh one on the next message.
- `jarv /load` - Bind this terminal to the most recently used session.
- `jarv /load <id>` - Bind this terminal to a specific session id.
- `jarv /update` - Check GitHub for the latest main commit and install it with pip.

## Heads-up mode

Run `jarv` with no prompt to start an interactive session. Type a prompt and press Enter to send it. Commands start with `/` (e.g. `/clear`, `/history`). Type `exit`, `quit`, or `/exit`, or press Ctrl+C, to leave.

## How jarv works

1. Loads config from `{CONFIG_FILE}`.
2. Detects the current terminal and resolves its active session (default: one session per terminal).
3. Loads recent conversation history from that session's history file.
4. Sends your query, recent history, the configured system prompt, and system info to the OpenAI Responses API.
5. Streams the assistant response in the terminal.
6. If the model calls the shell tool, jarv displays the command, runs it, shows stdout/stderr/exit status, and sends the full command result back to the model.
7. Saves the final assistant response back to history, trimmed to `max_history` items.

## Shell command behavior

- jarv exposes three tools to the model: `run_command`, `spawn`, and `read_artifact`.
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
- `system_prompt` - Instructions sent to the model before each request.
- `check_updates` - Whether to check for updates on each run. Default: `true`. Set to `false` to skip the background update check and avoid the ~200 ms network wait.

If the config file does not exist, jarv creates it and exits so you can add an API key.
If the config file is invalid JSON, jarv backs it up and creates a fresh default config.

## History and sessions

Session metadata file: `{SESSIONS_FILE}`

Each terminal is bound to exactly one session at a time. By default a fresh terminal gets its own session (id derived from terminal fingerprint). History for a session lives in `history-<hash>.json` under `{CONFIG_DIR}`.

- `jarv /clear` archives the current session's history+artifacts and removes the terminal's mapping. The next prompt starts a fresh session.
- `jarv /load` looks up the most recently used session anywhere and binds it to this terminal.
- `jarv /load <id>` binds a specific session id to this terminal.

## Updates

- `jarv /update` checks `{GITHUB_REPO}` on GitHub and installs the latest version from `{INSTALL_URL}`.
- Normal question runs also do a quick background update check and tell you if an update is available.
- Set `check_updates` to `false` (`jarv /set check_updates false`) to disable the background check and remove the ~200 ms latency it adds.
- After updating, run `jarv` again to use the new version.

## Files

- Config directory: `{CONFIG_DIR}`
- Config file: `{CONFIG_FILE}`
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


def maybe_print_update_available() -> None:
    if _update_available:
        sha = _update_available[0]
        if not _load_known_sha():
            _save_sha(sha)
        else:
            console.print("[yellow]Update available![/yellow] Run [bold]jarv /update[/bold] to install.")


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


def cmd_clear() -> None:
    session_context = prepare_session_context()
    history_path = session_context.history_file

    archived_any = False
    if history_path.exists() and load_history(history_path):
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        cleared_at = utc_now().strftime("%Y%m%dT%H%M%SZ")
        stem_suffix = history_path.stem[len("history"):]
        archived_history = ARCHIVE_DIR / f"history-{cleared_at}{stem_suffix}.json"
        history_path.rename(archived_history)

        artifact_path = artifact_file_for(history_path)
        if artifact_path.exists():
            archived_artifacts = ARCHIVE_DIR / f"artifacts-{cleared_at}{stem_suffix}.json"
            artifact_path.rename(archived_artifacts)

        console.print(f"[dim]Session archived to[/dim] {archived_history}")
        archived_any = True
    else:
        console.print("[dim]No history to archive.[/dim]")

    forget_current_session()
    if archived_any:
        console.print("[green]Fresh session will start on the next message.[/green]")


def cmd_cleanup() -> None:
    """Delete session files in ~/.jarv/sessions/ that have no matching sessions.json entry."""
    if not SESSIONS_DIR.exists():
        console.print("[dim]No sessions directory found — nothing to clean up.[/dim]")
        return

    data = load_sessions()
    known_hashes = {short_hash(sid) for sid in data["sessions"]}

    removed = 0
    for path in sorted(SESSIONS_DIR.iterdir()):
        if path.suffix != ".json":
            continue
        stem = path.stem
        for prefix in ("history-", "artifacts-"):
            if stem.startswith(prefix):
                file_hash = stem[len(prefix):]
                if file_hash not in known_hashes:
                    path.unlink()
                    console.print(f"[dim]Removed[/dim] {path.name}")
                    removed += 1
                break

    if removed:
        console.print(f"[green]Removed {removed} orphaned file{'s' if removed != 1 else ''}.[/green]")
    else:
        console.print("[dim]No orphaned session files found.[/dim]")


def cmd_load(args: list) -> None:
    data = load_sessions()
    sessions = data["sessions"]
    if not sessions:
        console.print("[yellow]No sessions exist yet.[/yellow]")
        return

    if args:
        session_id = args[0]
        if session_id not in sessions:
            console.print(f"[red]Unknown session id:[/red] {session_id}")
            console.print("[dim]Run `jarv` with no args after binding, or pick from existing ids.[/dim]")
            return
    else:
        session_id = max(
            sessions.keys(),
            key=lambda sid: sessions[sid].get("last_used_at", ""),
        )

    set_terminal_session(session_id)
    label = sessions[session_id].get("label", session_id)
    console.print(f"[green]This terminal is now bound to session[/green] [bold cyan]{session_id}[/bold cyan] [dim]({label})[/dim]")


def cmd_history() -> None:
    session_context = prepare_session_context()
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


def cmd_config() -> None:
    config = load_config()
    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")
    for k, v in config.items():
        val = "[dim]***[/dim]" if k == "api_key" and v else repr(v)
        table.add_row(k, val)
    console.print(f"[dim]{CONFIG_FILE}[/dim]")
    console.print(table)
