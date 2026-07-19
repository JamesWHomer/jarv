import os
import unittest
import platform
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.shell import (
    COMMAND_OUTPUT_UNSET,
    CommandResult,
    InteractiveCommandProcess,
    ShellState,
    compact_command_output,
    execute_command,
    resolve_command_output_window,
    truncate_command_output,
    truncate_model_output,
)


def _state_temp_files() -> set:
    return set(Path(tempfile.gettempdir()).glob("jarv-shell-state-*"))


class InteractiveSnapshotConsistencyTests(unittest.TestCase):
    def test_snapshot_polls_once_for_consistent_exit_state(self):
        # The process can exit between two poll() calls; a snapshot must never
        # report exited=True with exit_code=None.
        from types import SimpleNamespace

        polls = iter([None, 0])
        proc = SimpleNamespace(
            stdout=None,
            stderr=None,
            poll=lambda: next(polls, 0),
        )
        process = InteractiveCommandProcess("cmd", proc)

        snapshot = process.snapshot()

        self.assertEqual(snapshot.exited, snapshot.exit_code is not None)


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


class ShellStateTests(unittest.TestCase):
    def test_shell_state_persists_cwd(self):
        state = ShellState.initial()
        with TemporaryDirectory() as tmp:
            result = execute_command(f'cd "{tmp}"', shell_state=state)

            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.samefile(state.cwd, tmp))

            if platform.system() == "Windows":
                follow = execute_command("(Get-Location).Path", shell_state=state)
            else:
                follow = execute_command("pwd", shell_state=state)
            self.assertTrue(os.path.samefile(follow.stdout.strip(), tmp))

    def test_shell_state_persists_env_var(self):
        state = ShellState.initial()
        if platform.system() == "Windows":
            set_cmd = "$env:JARV_TEST_VALUE = 'hello-123'"
            get_cmd = "$env:JARV_TEST_VALUE"
        else:
            set_cmd = "export JARV_TEST_VALUE=hello-123"
            get_cmd = 'printf "%s" "$JARV_TEST_VALUE"'

        execute_command(set_cmd, shell_state=state)
        follow = execute_command(get_cmd, shell_state=state)

        self.assertIn("hello-123", follow.stdout)
        self.assertIsInstance(state.env, dict)
        self.assertEqual(state.env.get("JARV_TEST_VALUE"), "hello-123")
        self.assertFalse(
            any(key.upper().startswith("JARV_STATE_") for key in state.env)
        )
        if platform.system() == "Windows":
            self.assertTrue(any(key.lower() == "systemroot" for key in state.env))

    def test_shell_state_env_value_with_newline_survives(self):
        state = ShellState.initial()
        if platform.system() == "Windows":
            set_cmd = '$env:JARV_TEST_NL = "a" + [char]10 + "b"'
        else:
            set_cmd = "export JARV_TEST_NL=\"$(printf 'a\\nb')\""

        execute_command(set_cmd, shell_state=state)

        self.assertIsInstance(state.env, dict)
        self.assertEqual(state.env.get("JARV_TEST_NL"), "a\nb")

    def test_exit_code_preserved_with_shell_state(self):
        state = ShellState.initial()
        if platform.system() == "Windows":
            command = 'cmd /c "exit 5"'
        else:
            command = "sh -c 'exit 5'"

        result = execute_command(command, shell_state=state)

        self.assertEqual(result.exit_code, 5)

    def test_exit_in_command_keeps_previous_state(self):
        state = ShellState.initial()
        initial_cwd = state.cwd

        result = execute_command("exit 3", shell_state=state)

        self.assertEqual(result.exit_code, 3)
        if platform.system() == "Windows":
            # The appended capture never ran, so nothing was recorded.
            self.assertIsNone(state.env)
            self.assertEqual(state.cwd, initial_cwd)

    def test_timed_out_command_keeps_state(self):
        state = ShellState.initial()
        initial_cwd = state.cwd
        before = _state_temp_files()
        if platform.system() == "Windows":
            command = "Start-Sleep -Seconds 5"
        else:
            command = "sleep 5"

        result = execute_command(command, timeout=0.3, shell_state=state)

        self.assertTrue(result.timed_out)
        self.assertEqual(state.cwd, initial_cwd)
        self.assertIsNone(state.env)
        self.assertEqual(_state_temp_files(), before)

    def test_interactive_process_applies_state_on_exit(self):
        state = ShellState.initial()
        with TemporaryDirectory() as tmp:
            process = InteractiveCommandProcess.start(
                f'cd "{tmp}"', shell_state=state
            )
            try:
                snapshot = process.wait_until_idle(idle_seconds=0.2)
            finally:
                process.kill_tree()

            self.assertTrue(snapshot.exited)
            self.assertTrue(os.path.samefile(state.cwd, tmp))

    def test_shell_state_copy_is_independent(self):
        state = ShellState("original", {"A": "1"})
        clone = state.copy()
        clone.cwd = "changed"
        clone.env["A"] = "2"

        self.assertEqual(state.cwd, "original")
        self.assertEqual(state.env, {"A": "1"})
        self.assertIsNone(ShellState("x", None).copy().env)

    def test_execute_command_without_state_unchanged(self):
        before = _state_temp_files()

        result = execute_command("echo hi")

        self.assertIn("hi", result.stdout)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(_state_temp_files(), before)


if __name__ == "__main__":
    unittest.main()
