"""Session archive and sidecar file operations."""

from pathlib import Path
from functools import wraps
from .storage import transaction, read_json, write_json, delete_json, StorageConflict

from .history import (
    artifact_file_for,
    branches_file_for,
    history_file_for_session,
    load_history,
    reads_file_for,
    redo_file_for,
    utc_now,
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


@_coordinated
def archive_session_files(history_path: Path) -> Path | None:
    """Move history and sidecars for a session into ARCHIVE_DIR.

    Returns the new archived history path, or None if nothing was archived.
    """
    if not history_path.exists() or not load_history(history_path):
        return None
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    cleared_at = utc_now().strftime("%Y%m%dT%H%M%SZ")
    stem_suffix = history_path.stem[len("history"):]
    archived_history = ARCHIVE_DIR / f"history-{cleared_at}{stem_suffix}.json"
    _move(history_path, archived_history)

    artifact_path = artifact_file_for(history_path)
    if artifact_path.exists():
        _move(artifact_path, ARCHIVE_DIR / f"artifacts-{cleared_at}{stem_suffix}.json")

    reads_path = reads_file_for(history_path)
    if reads_path.exists():
        _move(reads_path, ARCHIVE_DIR / f"reads-{cleared_at}{stem_suffix}.json")

    usage_path = usage_file_for(history_path)
    if usage_path.exists():
        _move(usage_path, ARCHIVE_DIR / f"usage-{cleared_at}{stem_suffix}.json")

    branches_path = branches_file_for(history_path)
    if branches_path.exists():
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
    for path in (
        history_path,
        artifact_file_for(history_path),
        reads_file_for(history_path),
        usage_file_for(history_path),
        redo_file_for(history_path),
        branches_file_for(history_path),
    ):
        delete_json(path)
