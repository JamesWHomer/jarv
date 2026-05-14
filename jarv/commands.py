import json
import os
import subprocess
import sys
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
from .display import console, flatten_headings
from .history import (
    SESSIONS_DIR,
    SESSIONS_FILE,
    artifact_file_for,
    detect_terminal,
    forget_current_session,
    load_history,
    load_redo_stack,
    load_sessions,
    parse_timestamp,
    prepare_session_context,
    redo_file_for,
    save_history,
    save_redo_stack,
    set_terminal_session,
    short_hash,
    split_last_exchange,
    utc_now,
)

ARCHIVE_DIR = CONFIG_DIR / "archive"

GITHUB_REPO = "JamesWHomer/jarv"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/commits/main"
INSTALL_URL = f"https://github.com/{GITHUB_REPO}.git"
SHA_FILE = CONFIG_DIR / "last_sha.txt"


def _read_key() -> str:
    """Read a single keypress and return a normalised token.

    Returns one of: UP, DOWN, HOME, END, PAGEUP, PAGEDOWN, ENTER, ESC, or the
    raw character.  Raises KeyboardInterrupt on Ctrl-C.
    """
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            second = msvcrt.getwch()
            return {
                "H": "UP", "P": "DOWN",
                "G": "HOME", "O": "END",
                "I": "PAGEUP", "Q": "PAGEDOWN",
            }.get(second, "OTHER")
        if ch == "\r":
            return "ENTER"
        if ch in ("\x1b", "q", "Q"):
            return "ESC"
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
                        "A": "UP", "B": "DOWN",
                        "H": "HOME", "F": "END",
                        "5": "PAGEUP", "6": "PAGEDOWN",
                    }.get(ch3, "OTHER")
                return "ESC"
            if ch in ("\r", "\n"):
                return "ENTER"
            if ch in ("q", "Q"):
                return "ESC"
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
    cmd_table.add_row("jarv /sessions", "List the 5 most recently active sessions")
    cmd_table.add_row("jarv /load", "Load the most recently used session into this terminal")
    cmd_table.add_row("jarv /load <id>", "Load a specific session into this terminal")
    cmd_table.add_row("jarv /history", "Show recent conversation history")
    cmd_table.add_row("jarv /undo [n]", "Unsend the last n exchanges (default 1)")
    cmd_table.add_row("jarv /redo [n]", "Restore the last n undone exchanges (default 1)")
    cmd_table.add_row("jarv /config", "Show current settings")
    cmd_table.add_row("jarv /update", "Update jarv to the latest version")
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
- `jarv /undo [n]` - Unsend the last n exchanges (default 1). The removed exchange is pushed onto a redo stack.
- `jarv /redo [n]` - Restore the last n undone exchanges (default 1). Sending a new message clears the redo stack.
- `jarv /clear` - Archive this terminal's session and start a fresh one on the next message.
- `jarv /sessions` - List the 5 most recently active sessions.
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
- `jarv /sessions` lists the 5 most recently active sessions.
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

        redo_path = redo_file_for(history_path)
        if redo_path.exists():
            redo_path.unlink()

        console.print(f"[dim]Session archived to[/dim] {archived_history}")
        archived_any = True
    else:
        console.print("[dim]No history to archive.[/dim]")

    forget_current_session()
    if archived_any:
        console.print("[green]Fresh session will start on the next message.[/green]")



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

    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    table.add_column("", no_wrap=True, width=1)
    table.add_column("ID prefix", style="bold cyan", no_wrap=True)
    table.add_column("Last active", no_wrap=True)
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
    console.print(table)
    if total > shown:
        console.print(f"[dim]Showing {shown} most recent of {total} sessions.[/dim]")
    console.print("[dim]Run [bold]jarv /load <id>[/bold] to switch to a session.[/dim]")


def cmd_sessions() -> None:
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
        })

    n = len(rows)
    selected = next((i for i, r in enumerate(rows) if r["is_current"]), 0)

    def _visible_rows(term_h: int, include_footer: bool = True) -> int:
        """Return a row count that keeps the whole live panel inside the viewport."""
        # Panel border is 2 rows. The header consumes 1 content row, and the
        # footer consumes 2 more (blank spacer + controls) when there is room.
        content_rows = max(1, term_h - 2)
        reserved = 3 if include_footer else 1
        return max(1, min(10, content_rows - reserved))

    def _max_vis() -> int:
        term_h = console.size.height
        return _visible_rows(term_h, include_footer=term_h >= 6)

    def _clamp_offset(sel: int, off: int) -> int:
        """Keep sel inside [off, off + max_vis). Scroll only when it leaves the window."""
        mv = _max_vis()
        if sel < off:
            return sel
        if sel >= off + mv:
            return sel - mv + 1
        return off

    offset = _clamp_offset(selected, 0)

    def _render(sel: int, off: int) -> Panel:
        term_h = console.size.height
        show_footer = term_h >= 6
        mv = _visible_rows(term_h, include_footer=show_footer)
        off = _clamp_offset(sel, off)
        start = off
        end = min(n, off + mv)

        parts: list = []

        parts.append(Text(f"  showing {start + 1}–{end} of {n}", style="dim", no_wrap=True, overflow="ellipsis"))

        for i in range(start, end):
            r = rows[i]
            is_sel = i == sel
            t = Text(no_wrap=True, overflow="ellipsis")
            t.append(" › " if is_sel else "   ", style="bold cyan" if is_sel else "")
            t.append("● " if r["is_current"] else "  ", style="green" if r["is_current"] else "")
            t.append(f"{r['short_id']:<24}", style="bold cyan" if is_sel else "cyan")
            t.append(f"{r['time_str']:<12}", style="bold" if is_sel else "dim")
            snip = r["snippet"] or "no messages"
            t.append(snip, style="bold" if is_sel else "dim")
            parts.append(t)

        if show_footer:
            parts.append(Text(""))
            parts.append(Text("↑↓ navigate   Enter load   q cancel", style="dim italic", no_wrap=True, overflow="ellipsis"))

        return Panel(Group(*parts), title="[bold]sessions[/bold]", border_style="bright_black", padding=(0, 1))

    with Live(
        get_renderable=lambda: _render(selected, offset),
        console=console,
        auto_refresh=True,
        refresh_per_second=8,
        transient=True,
        vertical_overflow="crop",
    ) as live:
        while True:
            live.refresh()
            try:
                key = _read_key()
            except KeyboardInterrupt:
                break

            if key == "UP":
                selected = max(0, selected - 1)
            elif key == "DOWN":
                selected = min(n - 1, selected + 1)
            elif key == "HOME":
                selected = 0
            elif key == "END":
                selected = n - 1
            elif key == "PAGEUP":
                selected = max(0, selected - _max_vis())
            elif key == "PAGEDOWN":
                selected = min(n - 1, selected + _max_vis())
            elif key == "ENTER":
                row = rows[selected]
                set_terminal_session(row["sid"])
                label = sessions[row["sid"]].get("label", row["sid"])
                console.print(
                    f"[green]Loaded[/green] [bold cyan]{row['short_id']}[/bold cyan] [dim]({label})[/dim]"
                )
                return
            elif key == "ESC":
                break

            offset = _clamp_offset(selected, offset)

    console.print("[dim]Cancelled.[/dim]")


def cmd_load(args: list) -> None:
    data = load_sessions()
    sessions = data["sessions"]
    if not sessions:
        console.print("[yellow]No sessions exist yet.[/yellow]")
        return

    if args:
        prefix = args[0]
        if prefix in sessions:
            session_id = prefix
        else:
            matches = [sid for sid in sessions if sid.startswith(prefix)]
            if not matches:
                console.print(f"[red]No session matches:[/red] {prefix}")
                console.print("[dim]Run [bold]jarv /sessions[/bold] to see available sessions.[/dim]")
                return
            if len(matches) > 1:
                console.print(f"[yellow]Ambiguous prefix[/yellow] [bold]{prefix}[/bold] [yellow]matches {len(matches)} sessions:[/yellow]")
                for m in matches:
                    console.print(f"  [dim]{m}[/dim]")
                return
            session_id = matches[0]
    else:
        session_id = max(
            sessions.keys(),
            key=lambda sid: sessions[sid].get("last_used_at", ""),
        )

    set_terminal_session(session_id)
    label = sessions[session_id].get("label", session_id)
    console.print(f"[green]Loaded[/green] [bold cyan]{_short_session_id(session_id)}[/bold cyan] [dim]({label})[/dim]")


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
        console.print("[dim]Nothing to undo.[/dim]")
        return

    save_history(history, ctx.history_file)
    save_redo_stack(stack, redo_path)

    if len(undone) == 1:
        text = _first_user_text(undone[0])
        console.print(f"[bold]↶ Unsent:[/bold] [cyan]{text!r}[/cyan]")
        console.print(f"[dim]Removed {len(undone[0])} item(s). Run [bold]/redo[/bold] to put it back.[/dim]")
    else:
        console.print(f"[bold]↶ Unsent {len(undone)} exchanges:[/bold]")
        for i, frame in enumerate(undone, 1):
            console.print(f"  {i}. [cyan]{_first_user_text(frame)!r}[/cyan]")
        console.print(f"[dim]Run [bold]/redo {len(undone)}[/bold] to put them back.[/dim]")


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
        console.print("[dim]Nothing to redo.[/dim]")
        return

    save_history(history, ctx.history_file)
    save_redo_stack(stack, redo_path)

    if len(restored) == 1:
        text = _first_user_text(restored[0])
        console.print(f"[bold]↷ Restored:[/bold] [cyan]{text!r}[/cyan]")
    else:
        console.print(f"[bold]↷ Restored {len(restored)} exchange(s).[/bold]")
