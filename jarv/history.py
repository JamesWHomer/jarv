import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

from .storage import read_json, write_json, transaction, delete_json, StorageError, JsonList
from .display import console
from .paths import CONFIG_DIR, SESSIONS_DIR, SESSIONS_FILE
from .unicode_safety import sanitize_json_value


_RESTART_TERMINAL_ENV = "_JARV_RESTART_TERMINAL"


def _consume_restart_terminal() -> tuple[str, str] | None:
    """Consume a relaunch identity so it cannot leak into tool subprocesses."""
    raw = os.environ.pop(_RESTART_TERMINAL_ENV, None)
    if raw is None:
        return None
    import json

    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if isinstance(value, list) and len(value) == 2 and all(
        isinstance(part, str) and part for part in value
    ):
        return value[0], value[1]
    return None


_restart_terminal = _consume_restart_terminal()


def load_history(path: Path) -> list:
    data = read_json(path, [], list)
    data[:] = sanitize_json_value(data)
    return data


def save_history(history: list, path: Path) -> None:
    write_json(path, sanitize_json_value(history), snapshot=history if hasattr(history, "baseline") else None)


class SessionMetadata(dict):
    """Carry the original snapshot across other metadata reads."""


def load_sessions() -> dict:
    raw = read_json(SESSIONS_FILE, {"terminals": {}, "sessions": {}}, dict)
    data = sanitize_json_value(raw)
    data.setdefault("terminals", {})
    data.setdefault("sessions", {})
    if not isinstance(data["terminals"], dict) or not isinstance(data["sessions"], dict):
        raise StorageError(f"Invalid sessions metadata: {SESSIONS_FILE}")
    result = SessionMetadata(data)
    result.baseline = raw.baseline
    return result


def save_sessions(data: dict) -> None:
    write_json(SESSIONS_FILE, sanitize_json_value(data), merge=True,
               snapshot=data if isinstance(data, SessionMetadata) else None)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    # Session timestamps are UTC, including older records without an offset.
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def get_shell_name() -> str:
    shell = os.environ.get("SHELL")
    if shell:
        return shell
    if os.name == "nt" and os.environ.get("PSModulePath"):
        return "Windows PowerShell 5.1 (powershell.exe)"
    return os.environ.get("ComSpec", "cmd.exe")


def short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:length]


def new_frame_id() -> str:
    """Stable id stamped on a user message so a prompt can anchor a branch."""
    import uuid

    return uuid.uuid4().hex


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
    if _restart_terminal is not None:
        return _restart_terminal
    candidates = [
        ("tmux-pane", "|".join([os.environ.get("TMUX", ""), os.environ["TMUX_PANE"]])
         if os.environ.get("TMUX_PANE") else None),
        ("screen-window", "|".join([os.environ["STY"], os.environ["WINDOW"]])
         if os.environ.get("STY") and os.environ.get("WINDOW") else None),
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


def session_directory(path: str | Path | None = None) -> str:
    """Canonical directory key, including Windows case and symlink handling."""
    return os.path.normcase(os.path.realpath(path if path is not None else os.getcwd()))


def latest_session_for_directory() -> str | None:
    """Find the latest saved conversation used in the invocation's directory."""
    directory = session_directory()
    candidates = []
    for session_id, meta in load_sessions()["sessions"].items():
        if meta.get("archived"):
            continue
        timestamp = parse_timestamp(meta.get("directories", {}).get(directory))
        if timestamp is not None:
            candidates.append((timestamp, session_id, meta))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for _, session_id, meta in candidates:
        history_path = meta.get("history_file")
        if history_path and Path(history_path).is_file() and load_history(Path(history_path)):
            return session_id
    return None


def migrate_flat_session_files() -> None:
    """Move flat per-session sidecars from ~/.jarv/ into ~/.jarv/sessions/."""
    with transaction(CONFIG_DIR / "sessions.json"):
        # Discover after recovery and while holding the lock: another startup
        # may already have moved these files while this process was waiting.
        flat_history = list(CONFIG_DIR.glob("history-*.json"))
        flat_artifacts = list(CONFIG_DIR.glob("artifacts-*.json"))
        flat_reads = list(CONFIG_DIR.glob("reads-*.json"))
        files_to_move = flat_history + flat_artifacts + flat_reads
        if not files_to_move and not SESSIONS_FILE.exists():
            return
        if files_to_move:
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        moved_histories = {}
        for src in files_to_move:
            dest = SESSIONS_DIR / src.name
            value = read_json(src, {}, (dict, list))
            if dest.exists() and read_json(dest, {}, (dict, list)) != value:
                raise StorageError(f"Conflicting legacy session files: {src} and {dest}")
            write_json(dest, value)
            delete_json(src)
            if src in flat_history:
                moved_histories[src.resolve()] = dest

        data = load_sessions()
        changed = False
        legacy_root = CONFIG_DIR.resolve()
        for meta in data["sessions"].values():
            old_path = meta.get("history_file") if isinstance(meta, dict) else None
            if isinstance(old_path, str) and old_path:
                old_path = Path(old_path).resolve()
                if old_path.parent != legacy_root or not old_path.match("history-*.json"):
                    continue
                dest = moved_histories.get(old_path)
                if dest is None and not old_path.exists():
                    # Older versions moved files without updating metadata.
                    # Repair only a known legacy location with a saved target.
                    candidate = SESSIONS_DIR / old_path.name
                    if candidate.is_file():
                        dest = candidate
                if dest is not None:
                    meta["history_file"] = str(dest)
                    changed = True
        if changed:
            save_sessions(data)


def artifact_file_for(history_path: Path) -> Path:
    return history_path.with_name(history_path.name.replace("history", "artifacts", 1))


def reads_file_for(history_path: Path) -> Path:
    return history_path.with_name(history_path.name.replace("history", "reads", 1))


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


def ephemeral_session_context() -> SessionContext:
    """Allocate a private session identity without opening persistent storage."""
    import uuid

    session_id = f"incognito-{uuid.uuid4().hex}"
    return SessionContext(
        session_id=session_id,
        session_label="incognito",
        history_file=SESSIONS_DIR / f"history-{session_id}.json",
        now=utc_now(),
    )


_session_override: str | None = None


@contextmanager
def session_override(session_id: str):
    """Select a session for one invocation without rebinding its terminal."""
    global _session_override
    previous = _session_override
    _session_override = session_id
    try:
        if load_sessions()["sessions"].get(session_id, {}).get("archived"):
            raise ValueError(f"Session {session_id!r} is archived; restore it before resuming.")
        yield
    finally:
        _session_override = previous


def prepare_session_context(
    mark_message: bool = False,
    *,
    persist_metadata: bool = True,
) -> SessionContext:
    """Resolve the active session for this terminal, creating it if needed."""
    now = utc_now()
    terminal_id, terminal_label = detect_terminal()

    sessions_data = load_sessions()
    terminals = sessions_data["terminals"]
    sessions = sessions_data["sessions"]

    session_id = _session_override or terminals.get(terminal_id)
    if session_id is None:
        session_id = terminal_id
    if sessions.get(session_id, {}).get("archived"):
        import uuid
        session_id = f"{terminal_id}-{uuid.uuid4().hex[:8]}"
    if _session_override is None:
        terminals[terminal_id] = session_id
    else:
        terminal_label = session_id

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
        # A terminal can chat in several projects. Retain each directory's
        # message time; opening a menu must not change which session resumes.
        meta.setdefault("directories", {})[session_directory()] = now.isoformat()

    if persist_metadata and (mark_message or session_existed):
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
    global _session_override
    if _session_override is not None:
        _session_override = session_id
        return
    terminal_id, _ = detect_terminal()
    data = load_sessions()
    data["terminals"][terminal_id] = session_id
    save_sessions(data)


def forget_current_session() -> None:
    """Point the current terminal at a brand-new session (keeps old session metadata)."""
    import uuid

    global _session_override
    if _session_override is not None:
        _session_override = f"session-{uuid.uuid4().hex}"
        return

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
    return read_json(path, [], list)


def save_redo_stack(stack: list[list], path: Path) -> None:
    write_json(path, sanitize_json_value(stack), snapshot=stack if hasattr(stack, "baseline") else None)


def branches_file_for(history_path: Path) -> Path:
    """Sidecar holding every off-spine prompt frame for a session's prompt tree."""
    return history_path.with_name(history_path.name.replace("history", "branches", 1))


def load_branches(path: Path) -> list[dict]:
    data = read_json(path, {"version": 1, "frames": []}, dict)
    if not isinstance(data.get("frames"), list):
        raise StorageError(f"Invalid branches: {path}")
    frames = JsonList(sanitize_json_value(data["frames"]))
    frames.baseline = data.baseline
    return frames


def save_branches(frames: list[dict], path: Path) -> None:
    write_json(path, sanitize_json_value({"version": 1, "frames": frames}),
               snapshot=frames if hasattr(frames, "baseline") else None)
