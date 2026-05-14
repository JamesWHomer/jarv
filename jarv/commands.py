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
from .display import console, flatten_headings, jarv_panel, section_rule, status_line
from .history import (
    SESSIONS_DIR,
    SESSIONS_FILE,
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

GITHUB_REPO = "JamesWHomer/jarv"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/commits/main"
INSTALL_URL = f"https://github.com/{GITHUB_REPO}.git"
SHA_FILE = CONFIG_DIR / "last_sha.txt"
UPDATE_FLAG_FILE = CONFIG_DIR / "update_available.txt"
LAST_CHECK_FILE = CONFIG_DIR / "last_update_check.txt"
UPDATE_CHECK_INTERVAL_HOURS = 24


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
    cmd_table.add_row("jarv /clear", "Start a fresh session on the next message")
    cmd_table.add_row("jarv /sessions, /session", "List sessions (all in a TTY; 5 most recent when piped/non-TTY)")
    cmd_table.add_row("jarv /load", "Load the most recently used session into this terminal")
    cmd_table.add_row("jarv /load <id>", "Load a specific session into this terminal")
    cmd_table.add_row("jarv /history", "Show recent conversation history")
    cmd_table.add_row("jarv /usage", "Show token usage for this session")
    cmd_table.add_row("jarv /undo [n]", "Unsend the last n exchanges (default 1)")
    cmd_table.add_row("jarv /redo [n]", "Restore the last n undone exchanges (default 1)")
    cmd_table.add_row("jarv /config", "Show current settings")
    cmd_table.add_row("jarv /update", "Update jarv to the latest version")
    cmd_table.add_row("jarv /about", "Show detailed information about jarv")
    cmd_table.add_row("jarv /help", "Show this help")

    key_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    key_table.add_column(style="bold yellow", no_wrap=True)
    key_table.add_column(style="white")
    key_table.add_row("api_key", "OpenAI API key")
    key_table.add_row("model", "Model name (default: gpt-5.4-mini)")
    key_table.add_row("reasoning_effort", "Reasoning effort value (empty to disable)")
    key_table.add_row("max_history", "Number of messages to keep as context")
    key_table.add_row("command_timeout", "Seconds before a shell command is killed")
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
- `jarv /usage` - Show token usage for the current session.
- `jarv /undo [n]` - Unsend the last n exchanges (default 1). The removed exchange is pushed onto a redo stack.
- `jarv /redo [n]` - Restore the last n undone exchanges (default 1). Sending a new message clears the redo stack.
- `jarv /clear` - Start a fresh session on the next message.
- `jarv /sessions` / `jarv /session` - List sessions by recency. In an interactive terminal you can scroll through all of them; when stdout is not a TTY (e.g. piped), only the 5 most recent are listed.
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

- `api_key` - OpenAI API key. Can also be provided with the `OPENAI_API_KEY` environment variable.
- `model` - OpenAI model name. Default: `{DEFAULT_CONFIG['model']}`.
- `reasoning_effort` - Optional reasoning effort value. Empty disables this setting.
- `max_history` - Number of history items kept as context. Default: `{DEFAULT_CONFIG['max_history']}`.
- `command_timeout` - Seconds before a shell command is killed. Default: `{DEFAULT_CONFIG['command_timeout']}`.
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

- `jarv /clear` removes the terminal's session mapping. The next prompt starts a fresh session.
- `jarv /sessions` / `jarv /session` lists sessions by recency (all in a TTY; 5 most recent when stdout is not a TTY).
- `jarv /load` looks up the most recently used session anywhere and binds it to this terminal.
- `jarv /load <id>` binds a specific session id to this terminal.

## Updates

- `jarv /update` checks `{GITHUB_REPO}` on GitHub and installs the latest version from `{INSTALL_URL}`.
- A one-shot `jarv <question>` (arguments on the command line, not heads-up mode) fires a fully non-blocking background check when `check_updates` is true. If an update is found it is saved locally; the next invocation shows the notification instantly with no network wait.
- The background check is throttled to at most once every {UPDATE_CHECK_INTERVAL_HOURS} hours.
- Set `check_updates` to `false` (`jarv /set check_updates false`) to disable the background check entirely.
- After updating, run `jarv` again to use the new version.

## Files

- Config directory: `{CONFIG_DIR}`
- Config file: `{CONFIG_FILE}`
- Session metadata file: `{SESSIONS_FILE}`
- Session history and artifacts: `{SESSIONS_DIR}`
- Last known update SHA: `{SHA_FILE}`

## Version

jarv {__version__}
"""
    console.print(jarv_panel(Markdown(flatten_headings(about)), title="about", subtitle=f"v{__version__}"))


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
    """Fetch the latest SHA from GitHub and write a flag file if an update is available.

    Runs in a daemon thread — never blocks the main process. The flag is read
    (and cleared) on the *next* jarv invocation so there is zero network wait
    on the current run.
    """
    if not _should_check_now():
        return
    _record_check_time()
    latest = _fetch_latest_sha()
    if not latest:
        return
    known = _load_known_sha()
    if not known:
        # First run — just record the baseline SHA silently.
        _save_sha(latest)
        return
    if latest != known:
        CONFIG_DIR.mkdir(exist_ok=True)
        UPDATE_FLAG_FILE.write_text(latest)


def maybe_print_update_available() -> None:
    """Show a pending update notification written by a previous run's background check."""
    if not UPDATE_FLAG_FILE.exists():
        return
    try:
        sha = UPDATE_FLAG_FILE.read_text().strip()
        UPDATE_FLAG_FILE.unlink(missing_ok=True)
        if sha and sha != _load_known_sha():
            console.print("[yellow]Update available![/yellow] Run [bold]jarv /update[/bold] to install.")
    except Exception:
        pass


def cmd_update() -> None:
    console.print("[dim]⟳ Checking for updates…[/dim]")
    latest = _fetch_latest_sha()
    if latest is None:
        console.print("[bold red]✗[/bold red] [red]Could not reach GitHub.[/red]")
        return
    known = _load_known_sha()
    if known and latest == known:
        console.print("[bold green]✓[/bold green] [green]Already up to date.[/green]")
        return
    short = latest[:12]
    if not known:
        console.print(f"[bold cyan]↓[/bold cyan] Installing latest version [dim]({short})[/dim]…")
    else:
        console.print(f"[bold cyan]↓[/bold cyan] Update found [dim]({short})[/dim]. Installing…")
    with console.status("[dim]Running pip install…[/dim]", spinner="dots"):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", f"git+https://github.com/{GITHUB_REPO}.git"],
            capture_output=True,
            text=True,
        )
    if result.returncode == 0:
        _save_sha(latest)
        console.print("[bold green]✓[/bold green] [green]Updated successfully.[/green] [dim]Run jarv again to use the new version.[/dim]")
    else:
        console.print("[bold red]✗[/bold red] [red]Update failed:[/red]")
        output = "\n".join(filter(None, [result.stdout.strip(), result.stderr.strip()]))
        if output:
            console.print(output, style="dim")


def cmd_clear() -> None:
    forget_current_session()
    console.print("[bold green]✓[/bold green] [green]Fresh session will start on the next message.[/green]")



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
    footer_parts += [Text("Run jarv /load <id> to switch to a session.", style="dim italic")]
    console.print(jarv_panel(Group(*footer_parts), title="sessions", subtitle=f"{shown}/{total}"))


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

    def _truncate(value: str, width: int) -> str:
        if width <= 0:
            return ""
        if len(value) <= width:
            return value
        if width <= 3:
            return value[:width]
        return value[:width - 3] + "..."

    def _visible_rows(term_h: int, include_footer: bool = True) -> int:
        """Return the row count that fills the alternate screen without overflowing."""
        # Panel border is 2 rows. The header consumes 1 content row, and the
        # footer consumes 2 more (blank spacer + controls) when there is room.
        content_rows = max(1, term_h - 2)
        reserved = 3 if include_footer else 1
        return max(1, content_rows - reserved)

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
        term_w = console.size.width
        term_h = console.size.height
        panel_width = max(1, term_w)
        inner_width = max(1, panel_width - 4)
        show_footer = term_h >= 6
        mv = _visible_rows(term_h, include_footer=show_footer)
        off = _clamp_offset(sel, off)
        start = off
        end = min(n, off + mv)

        parts: list = []

        parts.append(
            Text(
                _truncate(f"  showing {start + 1}–{end} of {n}", inner_width),
                style="dim",
                no_wrap=True,
                overflow="crop",
            )
        )

        for i in range(start, end):
            r = rows[i]
            is_sel = i == sel
            t = Text(no_wrap=True, overflow="ellipsis")
            prefix = " › " if is_sel else "   "
            marker = "● " if r["is_current"] else "  "
            remaining = inner_width - len(prefix) - len(marker)
            id_width = max(0, min(24, remaining))
            remaining -= id_width
            time_width = max(0, min(12, remaining))
            remaining -= time_width
            snippet_width = max(0, remaining)

            t.append(_truncate(prefix, inner_width), style="bold cyan" if is_sel else "")
            if inner_width > len(prefix):
                t.append(_truncate(marker, inner_width - len(prefix)), style="green" if r["is_current"] else "")
            if id_width:
                short_id = _truncate(r["short_id"], id_width)
                t.append(f"{short_id:<{id_width}}", style="bold cyan" if is_sel else "cyan")
            if time_width:
                time_str = _truncate(r["time_str"], time_width)
                t.append(f"{time_str:<{time_width}}", style="bold" if is_sel else "dim")
            snip = r["snippet"] or "no messages"
            if snippet_width:
                t.append(_truncate(snip, snippet_width), style="bold" if is_sel else "dim")
            parts.append(t)

        if show_footer:
            parts.append(Text(""))
            parts.append(
                Text(
                    _truncate("↑↓ navigate   Enter load   q cancel", inner_width),
                    style="dim italic",
                    no_wrap=True,
                    overflow="crop",
                )
            )

        return Panel(
            Group(*parts),
            title="[bold bright_white]jarv \u25b8 sessions[/bold bright_white]",
            title_align="left",
            subtitle=f"[dim]{sel + 1}/{n}[/dim]",
            subtitle_align="right",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            width=panel_width,
        )

    loaded_row: dict | None = None
    with Live(
        get_renderable=lambda: _render(selected, offset),
        console=console,
        screen=True,
        auto_refresh=True,
        refresh_per_second=8,
        transient=False,
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
                loaded_row = row
                break
            elif key == "ESC":
                break

            offset = _clamp_offset(selected, offset)

    if loaded_row is not None:
        label = sessions[loaded_row["sid"]].get("label", loaded_row["sid"])
        console.print(
            f"[bold green]✓[/bold green] [green]Loaded[/green] [bold cyan]{loaded_row['short_id']}[/bold cyan] [dim]({label})[/dim]"
        )
        return
    console.print("[dim]○ Cancelled.[/dim]")


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
                console.print(f"[bold red]✗[/bold red] [red]No session matches:[/red] [bold]{prefix}[/bold]")
                console.print("[dim]  Run [bold]jarv /sessions[/bold] to see available sessions.[/dim]")
                return
            if len(matches) > 1:
                console.print(f"[bold yellow]?[/bold yellow] [yellow]Ambiguous prefix[/yellow] [bold]{prefix}[/bold] [dim]matches {len(matches)} sessions:[/dim]")
                for m in matches:
                    console.print(f"  [dim]•[/dim] [cyan]{m}[/cyan]")
                return
            session_id = matches[0]
    else:
        session_id = max(
            sessions.keys(),
            key=lambda sid: sessions[sid].get("last_used_at", ""),
        )

    set_terminal_session(session_id)
    label = sessions[session_id].get("label", session_id)
    console.print(f"[bold green]✓[/bold green] [green]Loaded[/green] [bold cyan]{_short_session_id(session_id)}[/bold cyan] [dim]({label})[/dim]")


def cmd_history() -> None:
    session_context = prepare_session_context()
    history = load_history(session_context.history_file)
    if not history:
        console.print("[dim]○ No history yet.[/dim]")
        return

    exchanges = sum(1 for m in history if isinstance(m, dict) and m.get("role") == "user")
    parts: list = [
        section_rule("conversation"),
        Text(""),
    ]
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
