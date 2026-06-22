import unittest
import platform
import subprocess
import sys
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.shell import (
    COMMAND_OUTPUT_UNSET,
    CommandResult,
    InteractiveCommandProcess,
    compact_command_output,
    execute_command,
    resolve_command_output_window,
    truncate_command_output,
    truncate_model_output,
)


class ShellOutputLimitTests(unittest.TestCase):
    def test_compact_command_output_collapses_single_row_table(self):
        output = "\nPath\n----\nC:\\Users\\ubers\n\n"

        self.assertEqual(
            compact_command_output(output),
            "Path  C:\\Users\\ubers",
        )

    def test_command_output_is_middle_truncated_for_model(self):
        output = truncate_model_output("a" * 50 + "MIDDLE" + "z" * 50, 100)

        self.assertIn("tool output truncated to 100 characters", output)
        self.assertIn("characters omitted from the middle", output)
        self.assertTrue(output.startswith("a"))
        self.assertTrue(output.endswith("z"))
        self.assertNotIn("MIDDLE", output)

    def test_command_result_keeps_status_inside_limit(self):
        result = CommandResult("cmd", "x" * 200, "", 2)

        output = result.to_model_output(120)

        self.assertIn("command output truncated", output)
        self.assertIn("[exit code 2]", output)

    def test_command_output_window_uses_exact_head_and_tail_counts(self):
        output = truncate_command_output("abcdeMIDDLEvwxyz", 5, 5)

        self.assertTrue(output.startswith("abcde"))
        self.assertTrue(output.endswith("vwxyz"))
        self.assertIn("6 characters omitted from the middle", output)
        self.assertNotIn("MIDDLE", output)

    def test_command_output_window_returns_full_output_when_it_fits(self):
        self.assertEqual(
            truncate_command_output("abcdefghij", 4, 6),
            "abcdefghij",
        )
        self.assertEqual(
            truncate_command_output("abcdefghij", 10, 10),
            "abcdefghij",
        )

    def test_command_output_window_supports_head_only_tail_only_and_zero(self):
        head_only = truncate_command_output("abcdefghij", 3, 0)
        tail_only = truncate_command_output("abcdefghij", 0, 3)
        hidden = truncate_command_output("abcdefghij", 0, 0)

        self.assertTrue(head_only.startswith("abc"))
        self.assertFalse(head_only.endswith("hij"))
        self.assertTrue(tail_only.endswith("hij"))
        self.assertFalse(tail_only.startswith("abc"))
        self.assertIn("10 characters omitted from the middle", hidden)
        self.assertNotIn("abcdefghij", hidden)

    def test_command_output_window_defaults_split_config_limit(self):
        self.assertEqual(
            resolve_command_output_window(
                COMMAND_OUTPUT_UNSET,
                COMMAND_OUTPUT_UNSET,
                9,
            ),
            (4, 5),
        )
        self.assertEqual(
            resolve_command_output_window(2, COMMAND_OUTPUT_UNSET, 9),
            (2, 5),
        )
        self.assertEqual(
            resolve_command_output_window(COMMAND_OUTPUT_UNSET, 3, 9),
            (4, 3),
        )

    def test_command_output_window_rejects_invalid_values(self):
        for value in (-1, 1.5, "10", True, None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_command_output_window(value, COMMAND_OUTPUT_UNSET, 20)

    def test_explicit_command_window_overrides_max_chars(self):
        result = CommandResult("cmd", "a" * 50 + "z" * 50, "", 0)

        output = result.to_model_output(
            10,
            head_chars=20,
            tail_chars=20,
        )

        self.assertTrue(output.startswith("a" * 20))
        self.assertTrue(output.endswith("z" * 20))
        self.assertIn("60 characters omitted from the middle", output)
        self.assertNotIn("truncated to 10 characters", output)

    def test_cancellation_kills_active_process_tree(self):
        token = CancellationToken()
        killed = threading.Event()

        class FakeProcess:
            pid = 123
            returncode = None

            def communicate(self, timeout=None):
                if killed.is_set():
                    return "", ""
                raise subprocess.TimeoutExpired("cmd", timeout)

        timer = threading.Timer(0.02, token.cancel)
        timer.start()
        try:
            with (
                patch("jarv.shell.platform.system", return_value="Windows"),
                patch("jarv.shell.subprocess.Popen", return_value=FakeProcess()),
                patch("jarv.shell._kill_process_tree", side_effect=lambda _proc: killed.set()),
            ):
                with self.assertRaises(TurnCancelled):
                    execute_command("sleep", cancellation_token=token)
        finally:
            timer.cancel()

        self.assertTrue(killed.is_set())

    def test_interactive_process_check_in_does_not_kill_busy_command(self):
        with TemporaryDirectory() as tmp:
            script = Path(tmp) / "busy.py"
            script.write_text(
                "\n".join([
                    "import time",
                    "for i in range(50):",
                    "    print(f'tick {i}', flush=True)",
                    "    time.sleep(0.05)",
                ]),
                encoding="utf-8",
            )
            if platform.system() == "Windows":
                command = f'& "{sys.executable}" "{script}"'
            else:
                command = f'"{sys.executable}" "{script}"'
            process = InteractiveCommandProcess.start(command)
            try:
                snapshot = process.wait_until_idle(
                    idle_seconds=1.0,
                    first_output_grace_seconds=0.0,
                    check_in_seconds=1.0,
                )
            finally:
                process.kill_tree()

        self.assertFalse(snapshot.exited)
        self.assertTrue(snapshot.check_in)
        self.assertGreaterEqual(snapshot.elapsed_seconds, 0.8)
        self.assertIn("tick", snapshot.stdout_delta)
if __name__ == "__main__":
    unittest.main()
