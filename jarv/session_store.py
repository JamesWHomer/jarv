"""Session archive and sidecar file operations."""

from pathlib import Path
from functools import wraps
from contextlib import contextmanager
import copy
from .storage import transaction, read_json, write_json, delete_json, StorageConflict

from .history import (
    artifact_file_for,
    branches_file_for,
    history_file_for_session,
    load_branches,
    reads_file_for,
    redo_file_for,
    save_history,
    utc_now,
    isoformat_utc,
)
from .paths import ARCHIVE_DIR
from .usage import usage_file_for


def _coordinated(fn):
    @wraps(fn)
    def wrapped(history_path, *args, **kwargs):
        with transaction(history_path):
            return fn(history_path, *args, **kwargs)
    return wrapped


def _move(source, destination):
    if destination.exists():
        raise StorageConflict(f"Refusing to overwrite existing session file: {destination}")
    value = read_json(source, {}, (dict, list))
    write_json(destination, value)
    delete_json(source)


def mark_session_archived(data: dict, session_id: str, archived_path: Path) -> None:
    """Apply the metadata half of an archive consistently for every entry point."""
    meta = data["sessions"].setdefault(session_id, {})
    meta.update(history_file=str(archived_path), archived=True,
                archived_at=isoformat_utc(utc_now()))
    for terminal_id, mapped_id in list(data["terminals"].items()):
        if mapped_id == session_id:
            del data["terminals"][terminal_id]


@contextmanager
def session_metadata_transaction(history_path: Path, data: dict):
    """Commit files and metadata together, restoring the browser state on error."""
    original = copy.deepcopy(data)
    try:
        with transaction(history_path):
            yield
    except BaseException:
        # Keep the shared sessions/terminals mappings used by the browser.
        for key in ("sessions", "terminals"):
            data[key].clear()
            data[key].update(original[key])
        raise


@_coordinated
def archive_session_files(history_path: Path) -> Path | None:
    """Move history and sidecars for a session into ARCHIVE_DIR.

    Returns the new archived history path, or None if nothing was archived.
    """
    from .session_tree import preserve_redo_branches

    history = preserve_redo_branches(history_path)
    branches_path = branches_file_for(history_path)
    branches = load_branches(branches_path)
    if not history and not branches:
        return None
    if not history_path.exists():
        save_history(history, history_path)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    cleared_at = utc_now().strftime("%Y%m%dT%H%M%SZ")
    stem_suffix = history_path.stem[len("history"):]
    archived_history = ARCHIVE_DIR / f"history-{cleared_at}{stem_suffix}.json"
    _move(history_path, archived_history)

    for kind, path_for in (
        ("artifacts", artifact_file_for),
        ("reads", reads_file_for),
        ("usage", usage_file_for),
    ):
        sidecar = path_for(history_path)
        if sidecar.exists():
            _move(sidecar, ARCHIVE_DIR / f"{kind}-{cleared_at}{stem_suffix}.json")

    # Normalizing legacy redo can have staged a new sidecar that is not yet
    # visible through Path.exists(); it must move in this same transaction.
    if branches or branches_path.exists():
        _move(branches_path, ARCHIVE_DIR / f"branches-{cleared_at}{stem_suffix}.json")

    redo_path = redo_file_for(history_path)
    if redo_path.exists():
        delete_json(redo_path)

    return archived_history


@_coordinated
def unarchive_session_files(archived_history_path: Path, session_id: str) -> Path | None:
    """Reverse archive_session_files for the given session id."""
    if not archived_history_path.exists():
        return None
    restored_history = history_file_for_session(session_id)
    _move(archived_history_path, restored_history)

    archived_dir = archived_history_path.parent
    archived_tail = archived_history_path.stem[len("history"):]  # "-{ts}-{hash}"
    restored_suffix = restored_history.stem[len("history"):]  # "-{hash}"
    for kind in ("artifacts", "reads", "usage", "branches"):
        sib = archived_dir / f"{kind}{archived_tail}.json"
        if sib.exists():
            _move(sib, restored_history.parent / f"{kind}{restored_suffix}.json")
    return restored_history

@_coordinated
def delete_session_files(history_path: Path) -> None:
    """Permanently remove history and sidecars for a session."""
    from .session_titles import forget_session_title

    forget_session_title(history_path)
    for path in (
        history_path,
        artifact_file_for(history_path),
        reads_file_for(history_path),
        usage_file_for(history_path),
        redo_file_for(history_path),
        branches_file_for(history_path),
    ):
        delete_json(path)
