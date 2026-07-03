"""Session archive and sidecar file operations."""

from pathlib import Path

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
    history_path.rename(archived_history)

    artifact_path = artifact_file_for(history_path)
    if artifact_path.exists():
        artifact_path.rename(ARCHIVE_DIR / f"artifacts-{cleared_at}{stem_suffix}.json")

    reads_path = reads_file_for(history_path)
    if reads_path.exists():
        reads_path.rename(ARCHIVE_DIR / f"reads-{cleared_at}{stem_suffix}.json")

    usage_path = usage_file_for(history_path)
    if usage_path.exists():
        usage_path.rename(ARCHIVE_DIR / f"usage-{cleared_at}{stem_suffix}.json")

    branches_path = branches_file_for(history_path)
    if branches_path.exists():
        branches_path.rename(ARCHIVE_DIR / f"branches-{cleared_at}{stem_suffix}.json")

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
    for kind in ("artifacts", "reads", "usage", "branches"):
        sib = archived_dir / f"{kind}{archived_tail}.json"
        if sib.exists():
            sib.rename(restored_history.parent / f"{kind}{restored_suffix}.json")
    return restored_history

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
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass
