import pytest
from jarv.storage import StorageError
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import jarv.history as history


class TerminalDetectionTests(unittest.TestCase):
    def test_windows_console_id_uses_console_window_handle(self):
        windll = SimpleNamespace(
            kernel32=SimpleNamespace(GetConsoleWindow=lambda: 123456)
        )

        with patch.object(history.os, "name", "nt"), patch("ctypes.windll", windll, create=True):
            terminal_id, label = history.get_windows_console_id()

        self.assertTrue(terminal_id.startswith("windows-console-"))
        self.assertTrue(label.startswith("Windows console "))

    def test_windows_terminal_env_takes_precedence_over_console_handle(self):
        windll = SimpleNamespace(
            kernel32=SimpleNamespace(GetConsoleWindow=lambda: 123456)
        )

        with (
            patch.object(history.os, "name", "nt"),
            patch("ctypes.windll", windll, create=True),
            patch.dict(history.os.environ, {"WT_SESSION": "stable-tab"}, clear=False),
        ):
            terminal_id, _ = history.detect_terminal()

        self.assertTrue(terminal_id.startswith("windows-terminal-"))

    def test_new_windows_console_uses_own_session_not_legacy_parent(self):
        with TemporaryDirectory() as tmp:
            sessions_file = Path(tmp) / "sessions.json"
            sessions_dir = Path(tmp) / "sessions"
            legacy_history = sessions_dir / "history-legacy.json"
            sessions_dir.mkdir()
            legacy_history.write_text('[{"role": "user", "content": "legacy"}]', encoding="utf-8")

            sessions_file.write_text(
                json.dumps(
                    {
                        "terminals": {},
                        "sessions": {
                            "parent-legacy": {
                                "history_file": str(legacy_history),
                                "last_message_at": "2026-05-19T02:00:00Z",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.object(history, "SESSIONS_FILE", sessions_file),
                patch.object(history, "SESSIONS_DIR", sessions_dir),
                patch.object(
                    history,
                    "detect_terminal",
                    return_value=("windows-console-new", "Windows console new"),
                ),
            ):
                context = history.prepare_session_context(mark_message=True)

            data = json.loads(sessions_file.read_text(encoding="utf-8"))

            self.assertEqual(context.session_id, "windows-console-new")
            self.assertEqual(data["terminals"]["windows-console-new"], "windows-console-new")
            self.assertIn("windows-console-new", data["sessions"])
            self.assertIn("parent-legacy", data["sessions"])

    def test_forget_current_session_maps_terminal_to_new_session(self):
        with TemporaryDirectory() as tmp:
            sessions_file = Path(tmp) / "sessions.json"
            sessions_file.write_text(
                json.dumps(
                    {
                        "terminals": {"windows-console-old": "windows-console-old"},
                        "sessions": {
                            "windows-console-old": {
                                "history_file": str(Path(tmp) / "old.json"),
                                "last_message_at": "2026-05-19T02:00:00Z",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.object(history, "SESSIONS_FILE", sessions_file),
                patch.object(
                    history,
                    "detect_terminal",
                    return_value=("windows-console-old", "Windows console old"),
                ),
            ):
                history.forget_current_session()

            data = json.loads(sessions_file.read_text(encoding="utf-8"))
            mapped_session = data["terminals"]["windows-console-old"]

            self.assertTrue(mapped_session.startswith("windows-console-old-"))
            self.assertNotEqual(mapped_session, "windows-console-old")

    def test_history_replaces_lone_surrogates(self):
        with TemporaryDirectory() as tmp:
            history_file = Path(tmp) / "history.json"

            history.save_history([{"role": "user", "content": "abc\udc8fdef"}], history_file)
            loaded = history.load_history(history_file)

            self.assertEqual(loaded[0]["content"], "abc?def")
            history_file.read_text(encoding="utf-8").encode("utf-8")


# --- sessions metadata I/O (pytest style) ----------------------------------- #

def test_load_sessions_missing_file_returns_default_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "SESSIONS_FILE", tmp_path / "sessions.json")

    assert history.load_sessions() == {"terminals": {}, "sessions": {}}


def test_load_sessions_malformed_json_raises(tmp_path, monkeypatch):
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(history, "SESSIONS_FILE", sessions_file)

    with pytest.raises(StorageError):
        history.load_sessions()


def test_load_sessions_non_dict_payload_raises(tmp_path, monkeypatch):
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setattr(history, "SESSIONS_FILE", sessions_file)

    with pytest.raises(StorageError):
        history.load_sessions()


def test_load_sessions_wrong_typed_keys_raise(tmp_path, monkeypatch):
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text(json.dumps({"terminals": [], "sessions": {}}), encoding="utf-8")
    monkeypatch.setattr(history, "SESSIONS_FILE", sessions_file)

    with pytest.raises(StorageError):
        history.load_sessions()


def test_save_then_load_sessions_round_trips(tmp_path, monkeypatch):
    sessions_file = tmp_path / "sessions.json"
    monkeypatch.setattr(history, "SESSIONS_FILE", sessions_file)
    data = {
        "terminals": {"term-1": "session-1"},
        "sessions": {"session-1": {"label": "Test", "last_used_at": "2026-01-01T00:00:00Z"}},
    }

    history.save_sessions(data)

    assert history.load_sessions() == data


if __name__ == "__main__":
    unittest.main()
def test_tmux_panes_have_distinct_sessions_even_inside_same_host_terminal(monkeypatch):
    monkeypatch.setenv("WT_SESSION", "shared-tab")
    monkeypatch.setenv("TMUX", "/tmp/tmux/server,123,0")
    monkeypatch.setenv("TMUX_PANE", "%1")
    first = history.detect_terminal()[0]
    monkeypatch.setenv("TMUX_PANE", "%2")
    second = history.detect_terminal()[0]
    assert first.startswith("tmux-pane-")
    assert first != second


def test_archived_default_session_is_never_reopened_as_empty_history(monkeypatch, tmp_path):
    monkeypatch.setattr(history, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(history, "SESSIONS_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(history, "detect_terminal", lambda: ("terminal", "Terminal"))
    history.save_sessions({"terminals": {}, "sessions": {"terminal": {
        "archived": True, "history_file": "archived-history.json"}}})
    context = history.prepare_session_context(mark_message=True)
    data = history.load_sessions()
    assert context.session_id != "terminal"
    assert data["sessions"]["terminal"]["history_file"] == "archived-history.json"
def test_parse_timestamp_normalizes_legacy_utc_and_ignores_bad_types():
    from datetime import datetime, timezone
    from jarv.history import parse_timestamp

    expected = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert parse_timestamp("2026-01-02T03:04:05") == expected
    assert parse_timestamp("2026-01-02T03:04:05Z") == expected
    assert parse_timestamp("2026-01-02T13:04:05+10:00") == expected
    for value in (None, "", "invalid", 123, [], {}):
        assert parse_timestamp(value) is None


@pytest.fixture
def legacy_session(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(history, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(history, "SESSIONS_FILE", tmp_path / "sessions.json")
    path = tmp_path / f"history-{history.short_hash('legacy')}.json"
    transcript = [{"role": "user", "content": "Find this older conversation"}]
    path.write_text(json.dumps(transcript), encoding="utf-8")
    metadata = {"terminals": {"terminal": "legacy"}, "sessions": {"legacy": {
        "history_file": str(path), "label": "Original label",
        "directories": {history.session_directory(): "2026-10-06T00:00:00Z"},
    }}}
    history.SESSIONS_FILE.write_text(json.dumps(metadata), encoding="utf-8")
    return path, transcript, metadata


def test_migration_moves_sidecars_and_updates_session_metadata(legacy_session):
    path, transcript, metadata = legacy_session
    sidecars = [history.artifact_file_for(path), history.reads_file_for(path)]
    for sidecar in sidecars:
        sidecar.write_text('{"saved":"sidecar"}', encoding="utf-8")

    history.migrate_flat_session_files()

    destination = history.SESSIONS_DIR / path.name
    expected = metadata["sessions"]["legacy"] | {"history_file": str(destination)}
    assert history.load_sessions() == {"terminals": metadata["terminals"],
                                       "sessions": {"legacy": expected}}
    assert history.load_history(destination) == transcript
    assert history.latest_session_for_directory() == "legacy"
    for source in [path, *sidecars]:
        assert not source.exists()
        assert (history.SESSIONS_DIR / source.name).is_file()


def test_migration_repairs_previously_moved_history_metadata(legacy_session):
    path, transcript, metadata = legacy_session
    history.SESSIONS_DIR.mkdir()
    path.rename(history.SESSIONS_DIR / path.name)
    sessions = metadata["sessions"]
    archive = path.parent / "archive" / path.name
    unrelated = path.parent / "another-directory" / path.name
    missing = path.parent / "history-missing.json"
    sessions.update(archived={"history_file": str(archive), "archived": True},
                    unrelated={"history_file": str(unrelated)},
                    missing={"history_file": str(missing)})
    history.SESSIONS_FILE.write_text(json.dumps(metadata), encoding="utf-8")

    history.migrate_flat_session_files()

    saved = history.load_sessions()["sessions"]
    assert saved["legacy"]["history_file"] == str(history.SESSIONS_DIR / path.name)
    assert history.latest_session_for_directory() == "legacy"
    for key in ("archived", "unrelated", "missing"):
        assert saved[key] == sessions[key]


def test_simultaneous_migrations_recheck_files_under_lock(legacy_session, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from threading import Barrier
    from jarv import storage

    path, transcript, _ = legacy_session
    entering = Barrier(2)

    @contextmanager
    def synchronized_transaction(target):
        entering.wait(timeout=5)
        with storage.transaction(target):
            yield

    monkeypatch.setattr(history, "transaction", synchronized_transaction)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(history.migrate_flat_session_files) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)

    assert not path.exists()
    assert history.load_history(history.SESSIONS_DIR / path.name) == transcript
    assert history.load_sessions()["sessions"]["legacy"]["history_file"] == str(history.SESSIONS_DIR / path.name)


def test_migration_failure_before_commit_preserves_files_and_metadata(legacy_session, monkeypatch):
    from jarv import storage

    path, transcript, metadata = legacy_session
    replace = storage.os.replace

    def fail_journal(source, destination):
        if Path(destination).name == ".jarv-transaction.json":
            raise OSError("disk full")
        return replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", fail_journal)
    with pytest.raises(StorageError, match="disk full"):
        history.migrate_flat_session_files()

    assert json.loads(path.read_text(encoding="utf-8")) == transcript
    assert json.loads(history.SESSIONS_FILE.read_text(encoding="utf-8")) == metadata
    assert not (history.SESSIONS_DIR / path.name).exists()


def test_interrupted_migration_recovers_files_and_metadata_together(legacy_session, monkeypatch):
    from jarv import storage

    path, transcript, _ = legacy_session
    atomic = storage._atomic

    def fail_metadata(target, value):
        if target == history.SESSIONS_FILE:
            raise OSError("interrupted metadata replacement")
        return atomic(target, value)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_atomic", fail_metadata)
        with pytest.raises(StorageError, match="interrupted metadata replacement"):
            history.migrate_flat_session_files()

    assert not path.exists()
    assert (path.parent / ".jarv-transaction.json").exists()
    saved = history.load_sessions()
    destination = history.SESSIONS_DIR / path.name
    assert saved["sessions"]["legacy"]["history_file"] == str(destination)
    assert history.load_history(destination) == transcript
    assert not (path.parent / ".jarv-transaction.json").exists()
