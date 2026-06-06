import unittest
import subprocess
import threading
from unittest.mock import patch

from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.shell import CommandResult, execute_command, truncate_model_output


class ShellOutputLimitTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
