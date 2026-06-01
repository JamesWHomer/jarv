import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import CONFIG_DIR
from .display import console
from .unicode_safety import sanitize_json_value

SESSIONS_FILE = CONFIG_DIR / "sessions.json"
SESSIONS_DIR = CONFIG_DIR / "sessions"


def load_history(path: Path) -> list:
    if not path.exists():
        return []
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(history, list):
            return sanitize_json_value(history)
        console.print(f"[yellow]Ignoring invalid history format:[/yellow] {path}")
    except json.JSONDecodeError as e:
        console.print(f"[yellow]Ignoring malformed history:[/yellow] {e}")
    except (OSError, UnicodeDecodeError) as e:
        console.print(f"[yellow]Could not read history:[/yellow] {e}")
    return []


def save_history(history: list, path: Path) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    try:
        path.write_text(json.dumps(sanitize_json_value(history), separators=(",", ":")), encoding="utf-8")
    except OSError as e:
        console.print(f"[yellow]Could not save history:[/yellow] {e}")


def load_sessions() -> dict:
    if not SESSIONS_FILE.exists():
        return {"terminals": {}, "sessions": {}}
    try:
        data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = sanitize_json_value(data)
            data.setdefault("terminals", {})
            data.setdefault("sessions", {})
            if isinstance(data["terminals"], dict) and isinstance(data["sessions"], dict):
                return data
    except json.JSONDecodeError as e:
        console.print(f"[yellow]Ignoring malformed sessions metadata:[/yellow] {e}")
    except (OSError, UnicodeDecodeError) as e:
        console.print(f"[yellow]Could not read sessions metadata:[/yellow] {e}")
    return {"terminals": {}, "sessions": {}}


def save_sessions(data: dict) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    try:
        SESSIONS_FILE.write_text(json.dumps(sanitize_json_value(data), indent=2), encoding="utf-8")
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


def get_shell_name() -> str:
    shell = os.environ.get("SHELL")
    if shell:
        return shell
    if os.name == "nt" and os.environ.get("PSModulePath"):
        return "Windows PowerShell 5.1 (powershell.exe)"
    return os.environ.get("ComSpec", "cmd.exe")


def short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:length]


def get_windows_console_id() -> tuple[str, str] | None:
    """Return a stable id for a classic Windows console window when available."""
    if os.name != "nt":
        return None
    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    except Exception:
        return None
    if not hwnd:
        return None
    terminal_id = f"windows-console-{short_hash(str(hwnd))}"
    return terminal_id, f"Windows console {terminal_id[-6:]}"


def detect_terminal() -> tuple[str, str]:
    """Return (terminal_id, label) for the current terminal."""
    candidates = [
        ("windows-terminal", os.environ.get("WT_SESSION")),
        ("term-session", os.environ.get("TERM_SESSION_ID")),
        ("tmux", os.environ.get("TMUX")),
        ("screen", os.environ.get("STY")),
    ]
    for source, value in candidates:
        if value:
            terminal_id = f"{source}-{short_hash(value)}"
            return terminal_id, f"{source} {terminal_id[-6:]}"

    windows_console = get_windows_console_id()
    if windows_console is not None:
        return windows_console

    user = os.environ.get("USERNAME") or os.environ.get("USER") or "unknown-user"
    raw = "|".join([str(os.getppid()), os.getcwd(), user, get_shell_name()])
    terminal_id = f"parent-{short_hash(raw)}"
    return terminal_id, f"parent process {os.getppid()}"


def history_file_for_session(session_id: str) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR / f"history-{short_hash(session_id)}.json"


def migrate_flat_session_files() -> None:
    """Move history-*.json and artifacts-*.json from ~/.jarv/ into ~/.jarv/sessions/."""
    flat_history = list(CONFIG_DIR.glob("history-*.json"))
    flat_artifacts = list(CONFIG_DIR.glob("artifacts-*.json"))
    files_to_move = flat_history + flat_artifacts
    if not files_to_move:
        return
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for src in files_to_move:
        dest = SESSIONS_DIR / src.name
        if not dest.exists():
            src.rename(dest)
        else:
            src.unlink()


def artifact_file_for(history_path: Path) -> Path:
    return history_path.with_name(history_path.name.replace("history", "artifacts", 1))


def last_user_message(history: list) -> dict | None:
    for item in reversed(history):
        if isinstance(item, dict) and item.get("role") == "user":
            return item
    return None


@dataclass
class SessionContext:
    session_id: str
    session_label: str
    history_file: Path
    now: datetime


def prepare_session_context(mark_message: bool = False) -> SessionContext:
    """Resolve the active session for this terminal, creating it if needed."""
    now = utc_now()
    terminal_id, terminal_label = detect_terminal()

    sessions_data = load_sessions()
    terminals = sessions_data["terminals"]
    sessions = sessions_data["sessions"]

    session_id = terminals.get(terminal_id)
    if session_id is None:
        session_id = terminal_id
    terminals[terminal_id] = session_id

    history_path = history_file_for_session(session_id)
    session_existed = session_id in sessions
    meta = sessions.setdefault(
        session_id,
        {
            "label": terminal_label,
            "first_seen_at": isoformat_utc(now),
        },
    )
    meta["last_used_at"] = isoformat_utc(now)
    meta["history_file"] = str(history_path)
    if mark_message:
        meta["last_message_at"] = isoformat_utc(now)

    if mark_message or session_existed:
        save_sessions(sessions_data)

    return SessionContext(
        session_id=session_id,
        session_label=meta.get("label", terminal_label),
        history_file=history_path,
        now=now,
    )


def history_metadata(context: SessionContext) -> dict:
    return {
        "created_at": isoformat_utc(context.now),
        "session_id": context.session_id,
        "session_label": context.session_label,
    }


def set_terminal_session(session_id: str) -> None:
    terminal_id, _ = detect_terminal()
    data = load_sessions()
    data["terminals"][terminal_id] = session_id
    save_sessions(data)


def forget_current_session() -> None:
    """Point the current terminal at a brand-new session (keeps old session metadata)."""
    import uuid

    terminal_id, _ = detect_terminal()
    data = load_sessions()
    data["terminals"][terminal_id] = f"{terminal_id}-{uuid.uuid4().hex[:8]}"
    save_sessions(data)


def split_last_exchange(history: list) -> tuple[list, list]:
    """Return (history_without_last_exchange, last_exchange).

    A frame starts at the last user message and extends to the end of history.
    If there is no user message, the second element is an empty list.
    """
    for i in range(len(history) - 1, -1, -1):
        item = history[i]
        if isinstance(item, dict) and item.get("role") == "user":
            return history[:i], history[i:]
    return history, []


def redo_file_for(history_path: Path) -> Path:
    return history_path.with_name(history_path.name.replace("history", "redo", 1))


def load_redo_stack(path: Path) -> list[list]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [frame for frame in data if isinstance(frame, list)]
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        pass
    return []


def save_redo_stack(stack: list[list], path: Path) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    try:
        path.write_text(json.dumps(stack, indent=2), encoding="utf-8")
    except OSError as e:
        console.print(f"[yellow]Could not save redo stack:[/yellow] {e}")
