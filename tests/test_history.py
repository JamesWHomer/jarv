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


def test_load_sessions_malformed_json_returns_default_shape(tmp_path, monkeypatch):
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(history, "SESSIONS_FILE", sessions_file)

    assert history.load_sessions() == {"terminals": {}, "sessions": {}}


def test_load_sessions_non_dict_payload_returns_default_shape(tmp_path, monkeypatch):
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setattr(history, "SESSIONS_FILE", sessions_file)

    assert history.load_sessions() == {"terminals": {}, "sessions": {}}


def test_load_sessions_wrong_typed_keys_return_default_shape(tmp_path, monkeypatch):
    sessions_file = tmp_path / "sessions.json"
    sessions_file.write_text(json.dumps({"terminals": [], "sessions": {}}), encoding="utf-8")
    monkeypatch.setattr(history, "SESSIONS_FILE", sessions_file)

    assert history.load_sessions() == {"terminals": {}, "sessions": {}}


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
