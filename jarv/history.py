import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import CONFIG_DIR, DEFAULT_CONFIG
from .display import console

HISTORY_FILE = CONFIG_DIR / "history.json"
SESSIONS_FILE = CONFIG_DIR / "sessions.json"


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
    if shell:
        return shell
    if os.name == "nt" and os.environ.get("PSModulePath"):
        # execute_command() invokes powershell.exe explicitly on Windows.
        # Be precise here so the model does not assume Bash/cmd/PowerShell 7
        # syntax such as `&&`. Windows PowerShell 5.1 is the default
        # powershell.exe on Windows 10.
        return "Windows PowerShell 5.1 (powershell.exe)"
    return os.environ.get("ComSpec", "cmd.exe")


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


def artifact_file_for(history_path: Path) -> Path:
    return history_path.with_name(history_path.name.replace("history", "artifacts", 1))


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


def history_metadata(context: SessionContext) -> dict:
    return {
        "created_at": isoformat_utc(context.now),
        "session_id": context.session_id,
        "session_label": context.session_label,
    }
