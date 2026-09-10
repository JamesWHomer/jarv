import copy
import json
from types import SimpleNamespace

import pytest

from jarv import history, session_browser, session_commands, session_store
from jarv.storage import StorageError


def _session(tmp_path, monkeypatch, *, archived=False):
    sessions_dir = tmp_path / "sessions"
    archive_dir = tmp_path / "archive"
    sessions_dir.mkdir()
    archive_dir.mkdir()
    path = (archive_dir if archived else sessions_dir) / "history-test.json"
    path.write_text('[{"role":"user","content":"keep me"}]', encoding="utf-8")
    sidecar = history.reads_file_for(path)
    sidecar.write_text('{"cmd_test":{"content":"keep output"}}', encoding="utf-8")
    metadata_path = tmp_path / "sessions.json"
    metadata = {"terminals": {"terminal": "test"}, "sessions": {
        "test": {"history_file": str(path), "archived": archived}
    }}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(history, "SESSIONS_FILE", metadata_path)
    monkeypatch.setattr(history, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(history, "detect_terminal", lambda: ("terminal", "test"))
    monkeypatch.setattr(session_store, "ARCHIVE_DIR", archive_dir)
    monkeypatch.setattr(session_commands, "prepare_session_context", lambda: SimpleNamespace(
        history_file=path, session_id="test"
    ))
    original_files = {p: p.read_bytes() for p in (path, sidecar, metadata_path)}
    return path, metadata, original_files


def _failed_save(data):
    # Fail after staging the metadata too: the outer transaction must discard
    # both moves and metadata, not merely keep the old JSON when save fails.
    history.save_sessions(data)
    raise StorageError("injected metadata failure")


def test_cli_archive_rolls_back_files_and_metadata_together(tmp_path, monkeypatch):
    _path, _metadata, original = _session(tmp_path, monkeypatch)
    monkeypatch.setattr(session_commands, "save_sessions", _failed_save)
    with pytest.raises(StorageError, match="injected"):
        session_commands.cmd_archive()
    assert all(path.read_bytes() == data for path, data in original.items())
    assert list((tmp_path / "archive").glob("*.json")) == []


def test_archive_restore_commits_metadata_and_terminal_mapping(tmp_path, monkeypatch):
    original_path, _metadata, _original = _session(tmp_path, monkeypatch)
    session_commands.cmd_archive()
    archived = history.load_sessions()
    assert archived["sessions"]["test"]["archived"] is True
    assert archived["terminals"]["terminal"] != "test"
    assert not original_path.exists()

    assert session_browser._cmd_sessions_load("test") == 0

    restored = history.load_sessions()
    assert not restored["sessions"]["test"].get("archived")
    assert restored["terminals"]["terminal"] == "test"
    restored_path = history.history_file_for_session("test")
    assert history.load_history(restored_path)[0]["content"] == "keep me"
    assert history.reads_file_for(restored_path).exists()
    assert list((tmp_path / "archive").glob("*.json")) == []


@pytest.mark.parametrize("operation", ["archive", "unarchive", "activate", "cli_load"])
def test_browser_lifecycle_rolls_back_moves_and_metadata(tmp_path, monkeypatch, operation):
    path, metadata, original = _session(tmp_path, monkeypatch, archived=operation != "archive")
    data = history.load_sessions()
    snapshot = copy.deepcopy(data)
    screen = object.__new__(session_browser.SessionBrowserScreen)
    screen.data = data
    screen.sessions = data["sessions"]
    screen.terminals = data["terminals"]
    row = {"sid": "test", "archived": operation != "archive", "is_current": True}
    original_row = dict(row)
    monkeypatch.setattr(session_browser, "save_sessions", _failed_save)

    with pytest.raises(StorageError, match="injected"):
        if operation == "archive":
            screen._archive_row(row)
        elif operation == "unarchive":
            screen._unarchive_row(row)
        elif operation == "activate":
            screen._activate_row(row)
        else:
            session_browser._cmd_sessions_load("test")

    assert all(path.read_bytes() == contents for path, contents in original.items())
    assert row == original_row
    assert data == snapshot
    opposite = tmp_path / ("archive" if operation == "archive" else "sessions")
    assert list(opposite.glob("*.json")) == []
