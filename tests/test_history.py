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

    def test_most_recent_legacy_parent_session_ignores_empty_sessions(self):
        with TemporaryDirectory() as tmp:
            old_history = Path(tmp) / "old.json"
            new_history = Path(tmp) / "new.json"
            old_history.write_text('[{"role": "user", "content": "old"}]', encoding="utf-8")
            new_history.write_text('[{"role": "user", "content": "new"}]', encoding="utf-8")

            sessions = {
                "parent-old": {
                    "history_file": str(old_history),
                    "last_message_at": "2026-05-19T01:00:00Z",
                },
                "parent-empty": {
                    "history_file": str(Path(tmp) / "missing.json"),
                    "last_message_at": "2026-05-19T03:00:00Z",
                },
                "parent-new": {
                    "history_file": str(new_history),
                    "last_message_at": "2026-05-19T02:00:00Z",
                },
            }

            self.assertEqual(
                history.most_recent_legacy_parent_session(sessions),
                "parent-new",
            )


if __name__ == "__main__":
    unittest.main()
