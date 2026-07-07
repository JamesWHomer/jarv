import json
import platform
import shlex
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from jarv.agent import _advance_interactive_continuation
from jarv.config import DEFAULT_CONFIG
from jarv.interactive_command import (
    _finalize_interactive_record,
    _parse_terminal_actions,
    _record_interactive_input,
    _run_command_waiting_prompt,
    _screen_stdin_text,
)
from jarv.orchestrator import (
    PendingRunCommand,
    RunCommandDispatchResult,
    ToolExecutionHooks,
    execute_tool_calls,
)
from jarv.provider import ToolCallDone
from jarv.shell import InteractiveCommandProcess, InteractiveCommandSnapshot


class TerminalReplyParsingTests(unittest.TestCase):
    def test_empty_reply_waits_instead_of_pressing_enter(self):
        # Enter can confirm a destructive [y/N] prompt; an empty model reply
        # must never imply it.
        actions, note = _parse_terminal_actions("")
        self.assertEqual(actions, [("wait", None)])
        self.assertIsNone(note)

    def test_narrative_before_control_executes_the_control(self):
        actions, note = _parse_terminal_actions("I'll press <ENTER>")
        self.assertEqual(actions, [("stdin", "\n")])
        self.assertIn("not sent", note)

    def test_single_word_answer_keeps_stdin_and_drops_trailing_controls(self):
        actions, note = _parse_terminal_actions("3<WAIT><WAIT 2s><EOF>Ran it.")
        self.assertEqual(actions, [("stdin", "3\n")])
        self.assertIn("ignored", note)

    def test_chained_controls_run_in_order(self):
        actions, note = _parse_terminal_actions("<DOWN> <DOWN> <ENTER>")
        self.assertEqual(
            actions,
            [("stdin_raw", "\x1b[B"), ("stdin_raw", "\x1b[B"), ("stdin", "\n")],
        )
        self.assertIsNone(note)

    def test_malformed_wait_is_invalid_not_typed_into_stdin(self):
        actions, _note = _parse_terminal_actions("<WAIT 2m>")
        self.assertEqual(actions[0][0], "invalid")

    def test_unknown_token_only_line_is_invalid(self):
        actions, _note = _parse_terminal_actions("<CTRL_Z>")
        self.assertEqual(actions[0][0], "invalid")

    def test_plain_text_with_unknown_token_stays_literal_stdin(self):
        actions, note = _parse_terminal_actions("echo <hello>")
        self.assertEqual(actions, [("stdin", "echo <hello>\n")])
        self.assertIsNone(note)


class WaitingPromptTests(unittest.TestCase):
    def test_waiting_prompt_truncates_output_to_prepared_window(self):
        snapshot = InteractiveCommandSnapshot(
            "cmd", "a" * 50 + "MIDDLE" + "z" * 50, "", None
        )
        prepared = SimpleNamespace(head_chars=10, tail_chars=10)
        prompt = _run_command_waiting_prompt(snapshot, prepared=prepared)
        self.assertIn("a" * 10, prompt)
        self.assertIn("z" * 10, prompt)
        self.assertNotIn("MIDDLE", prompt)

    def test_waiting_prompt_reports_closed_stdin(self):
        snapshot = InteractiveCommandSnapshot(
            "cmd", "out\n", "", None, stdin_closed=True
        )
        prompt = _run_command_waiting_prompt(snapshot)
        self.assertIn("stdin is closed", prompt)

    def test_waiting_prompt_carries_extra_statuses(self):
        snapshot = InteractiveCommandSnapshot("cmd", "out\n", "", None)
        prompt = _run_command_waiting_prompt(
            snapshot, statuses=("Note: the text before your controls was not sent.",)
        )
        self.assertIn("Note: the text before your controls was not sent.", prompt)


class InteractiveRecordTests(unittest.TestCase):
    def _pending(self):
        return SimpleNamespace(
            input_markers=[],
            transcript_segments=["Choose:"],
            output_item={"output": "old"},
        )

    def test_record_keeps_output_alongside_markers(self):
        pending = self._pending()
        _record_interactive_input(pending, "stdin", "3", "Name:")
        # A checkpoint taken now must still contain the command output, not
        # just the stdin markers.
        self.assertEqual(pending.output_item["output"], "Choose:\nstdin> 3\nName:")

    def test_finalize_appends_final_output_after_transcript(self):
        pending = self._pending()
        _record_interactive_input(pending, "stdin", "3", "Name:")
        _finalize_interactive_record(pending, "choice=3\n[exit code 0]")
        self.assertEqual(
            pending.output_item["output"],
            "Choose:\nstdin> 3\nName:\nchoice=3\n[exit code 0]",
        )


class StdinSafetyScreenTests(unittest.TestCase):
    def _pending(self):
        return SimpleNamespace(prepared=SimpleNamespace(cmd="bash"))

    def test_risky_stdin_line_denied_by_user_is_blocked(self):
        config = {**DEFAULT_CONFIG, "command_safety": "risky"}
        with patch("jarv.safety.prompt_confirmation", return_value=False):
            denial = _screen_stdin_text(
                self._pending(), [("stdin", "rm -rf /\n")], config
            )
        self.assertIsNotNone(denial)
        self.assertIn("blocked", denial)

    def test_risky_stdin_line_approved_by_user_passes(self):
        config = {**DEFAULT_CONFIG, "command_safety": "risky"}
        with patch("jarv.safety.prompt_confirmation", return_value=True):
            denial = _screen_stdin_text(
                self._pending(), [("stdin", "rm -rf /\n")], config
            )
        self.assertIsNone(denial)

    def test_benign_stdin_line_never_prompts(self):
        config = {**DEFAULT_CONFIG, "command_safety": "risky"}
        def boom(*_args):  # pragma: no cover - must not be called
            raise AssertionError("prompt_confirmation should not run")
        with patch("jarv.safety.prompt_confirmation", side_effect=boom):
            denial = _screen_stdin_text(self._pending(), [("stdin", "y\n")], config)
        self.assertIsNone(denial)


class InteractiveRoundCapTests(unittest.TestCase):
    def test_round_cap_kills_process_and_ends_interaction(self):
        killed = {}
        pending = SimpleNamespace(
            rounds=3,
            card=None,
            live=None,
            live_depth_cm=None,
            process=SimpleNamespace(kill_tree=lambda: killed.setdefault("yes", True)),
            unregister_cancel=None,
            input_markers=[],
            transcript_segments=["output so far"],
            output_item={"output": "output so far"},
        )
        config = {**DEFAULT_CONFIG, "interactive_max_rounds": 3}

        items, still_pending = _advance_interactive_continuation(
            pending,
            None,  # renderer: the cap branch must return before touching it
            [],
            config=config,
            cancellation_token=None,
            retained_store=None,
            ui=None,
            interactive_help={"sent": True},
        )

        self.assertIsNone(still_pending)
        self.assertTrue(killed)
        self.assertIn("aborted", items[-1]["content"])
        self.assertIn("output so far", pending.output_item["output"])
        self.assertIn("aborted", pending.output_item["output"])


class SkippedTrailingToolCallTests(unittest.TestCase):
    def test_calls_after_pending_interactive_command_get_skip_results(self):
        pending = PendingRunCommand(
            process=SimpleNamespace(),
            prepared=SimpleNamespace(),
            call_id="",
        )
        calls = [
            ToolCallDone(
                id="fc_1",
                call_id="call_1",
                name="run_command",
                arguments=json.dumps({"command": "python menu.py"}),
            ),
            ToolCallDone(
                id="fc_2",
                call_id="call_2",
                name="run_command",
                arguments=json.dumps({"command": "echo later"}),
            ),
        ]
        results = []
        hooks = ToolExecutionHooks(
            run_command=lambda args: RunCommandDispatchResult("waiting...", pending)
        )

        result = execute_tool_calls(
            calls,
            node=None,
            store=None,
            client=None,
            config=dict(DEFAULT_CONFIG),
            append_tool_result=lambda item, output: results.append(
                (item.call_id, output)
            ),
            hooks=hooks,
        )

        self.assertIs(result.pending_command, pending)
        self.assertEqual(pending.call_id, "call_1")
        self.assertEqual(results[0], ("call_1", "waiting..."))
        self.assertEqual(results[1][0], "call_2")
        self.assertIn("[skipped:", results[1][1])


class RequireOutputWaitTests(unittest.TestCase):
    def test_bare_wait_holds_until_check_in_on_silent_process(self):
        command_parts = " ".join(
            shlex.quote(part)
            for part in (sys.executable, "-c", "import time; time.sleep(10)")
        )
        if platform.system() == "Windows":
            command_parts = f"& {command_parts}"
        process = InteractiveCommandProcess.start(command_parts)
        try:
            snapshot = process.wait_until_idle(
                check_in_seconds=0.5,
                require_output=True,
            )
            # Without require_output the silent process would have returned as
            # "idle" after the first-output grace; with it, only the check-in
            # interval ends the wait.
            self.assertTrue(snapshot.check_in)
            self.assertFalse(snapshot.exited)
        finally:
            process.kill_tree()


if __name__ == "__main__":
    unittest.main()
