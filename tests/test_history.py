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


if __name__ == "__main__":
    unittest.main()
