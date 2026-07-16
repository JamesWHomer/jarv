import json
import platform
import shlex
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from jarv.agent import _advance_interactive_continuation
from jarv.config import DEFAULT_CONFIG
from jarv.interactive_command import (
    _MAX_CONSECUTIVE_INVALID,
    _finalize_interactive_record,
    _invalid_reply_message,
    _parse_terminal_actions,
    _record_interactive_input,
    _run_command_waiting_prompt,
    _screen_stdin_text,
    _terminal_action_display,
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
        actions, _note = _parse_terminal_actions("<WAIT soon>")
        self.assertEqual(actions[0][0], "invalid")

    def test_unknown_token_only_line_is_invalid(self):
        actions, _note = _parse_terminal_actions("<FROB>")
        self.assertEqual(actions[0][0], "invalid")

    def test_plain_text_with_unknown_token_stays_literal_stdin(self):
        actions, note = _parse_terminal_actions("echo <hello>")
        self.assertEqual(actions, [("stdin", "echo <hello>\n")])
        self.assertIsNone(note)

    def test_wait_accepts_minute_and_second_units(self):
        self.assertEqual(_parse_terminal_actions("<WAIT 2m>")[0], [("wait", 120.0)])
        self.assertEqual(_parse_terminal_actions("<WAIT 1min>")[0], [("wait", 60.0)])
        self.assertEqual(_parse_terminal_actions("<WAIT 90>")[0], [("wait", 90.0)])
        self.assertEqual(_parse_terminal_actions("<WAIT 5sec>")[0], [("wait", 5.0)])
        self.assertEqual(_parse_terminal_actions("<WAIT 500ms>")[0], [("wait", 0.5)])

    def test_new_key_tokens_send_raw_sequences(self):
        cases = {
            "<SPACE>": " ",
            "<BACKSPACE>": "\x08",
            "<DELETE>": "\x1b[3~",
            "<HOME>": "\x1b[H",
            "<END>": "\x1b[F",
            "<PAGE_UP>": "\x1b[5~",
            "<PAGE_DOWN>": "\x1b[6~",
        }
        for token, sequence in cases.items():
            actions, note = _parse_terminal_actions(token)
            self.assertEqual(actions, [("stdin_raw", sequence)], token)
            self.assertIsNone(note)

    def test_generic_ctrl_letter_sends_control_byte(self):
        actions, _note = _parse_terminal_actions("<CTRL_Z>")
        self.assertEqual(actions, [("stdin_raw", "\x1a")])

    def test_ctrl_c_and_ctrl_d_keep_signal_semantics(self):
        # The generic CTRL_x rule must not shadow interrupt/EOF handling.
        self.assertEqual(_parse_terminal_actions("<CTRL_C>")[0], [("ctrl_c", None)])
        self.assertEqual(_parse_terminal_actions("<CTRL_D>")[0], [("eof", None)])

    def test_dash_and_space_token_spellings_normalize(self):
        self.assertEqual(_parse_terminal_actions("<Ctrl-C>")[0], [("ctrl_c", None)])
        self.assertEqual(
            _parse_terminal_actions("<PAGE DOWN>")[0], [("stdin_raw", "\x1b[6~")]
        )

    def test_prose_line_then_control_line_executes_the_control(self):
        # The worst old failure: narration on line one was typed into the
        # live process and the real control dropped.
        actions, note = _parse_terminal_actions("Let me wait for it.\n<WAIT 30s>")
        self.assertEqual(actions, [("wait", 30.0)])
        self.assertIn("not typed into stdin", note)

    def test_prose_line_then_control_chain_executes_in_order(self):
        actions, _note = _parse_terminal_actions(
            "I'll pick the second entry.\n<DOWN> <ENTER>"
        )
        self.assertEqual(actions, [("stdin_raw", "\x1b[B"), ("stdin", "\n")])

    def test_single_word_first_line_is_never_redirected(self):
        actions, note = _parse_terminal_actions("y\n<ENTER>")
        self.assertEqual(actions, [("stdin", "y\n")])
        self.assertIsNone(note)

    def test_multiword_stdin_without_control_line_stays_stdin(self):
        actions, note = _parse_terminal_actions(
            "fix parser bug\nSecond thoughts about naming."
        )
        self.assertEqual(actions, [("stdin", "fix parser bug\n")])
        self.assertIsNone(note)

    def test_prose_then_malformed_control_line_is_invalid(self):
        # The model clearly meant a control; re-prompt instead of typing the
        # prose into stdin.
        actions, _note = _parse_terminal_actions("Waiting a bit.\n<WAIT soon>")
        self.assertEqual(actions[0][0], "invalid")

    def test_invalid_message_names_token_and_suggests_close_match(self):
        message = _invalid_reply_message("<ENTR>")
        self.assertIn("<ENTR>", message)
        self.assertIn("<ENTER>", message)

    def test_invalid_message_explains_wait_forms_and_single_letters(self):
        self.assertIn("<WAIT 2m>", _invalid_reply_message("<WAIT soon>"))
        self.assertIn("literal text y", _invalid_reply_message("<Y>"))

    def test_display_round_trips_new_keys_and_ctrl_letters(self):
        self.assertEqual(_terminal_action_display("stdin_raw", " "), "<SPACE>")
        self.assertEqual(_terminal_action_display("stdin_raw", "\x1a"), "<CTRL_Z>")
        self.assertEqual(_terminal_action_display("stdin_raw", "\x1b[6~"), "<PAGE_DOWN>")


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
        def boom(*_args, **_kwargs):  # pragma: no cover - must not be called
            raise AssertionError("prompt_confirmation should not run")
        with patch("jarv.safety.prompt_confirmation", side_effect=boom):
            denial = _screen_stdin_text(self._pending(), [("stdin", "y\n")], config)
        self.assertIsNone(denial)

    def test_all_level_gates_benign_stdin_line(self):
        # An approved bare shell must not be an ungated escape hatch when the
        # user asked for every command to be confirmed.
        config = {**DEFAULT_CONFIG, "command_safety": "all"}
        prompts = []

        def record(command, reason, **_kwargs):
            prompts.append((command, reason))
            return False

        with patch("jarv.safety.prompt_confirmation", side_effect=record):
            denial = _screen_stdin_text(
                self._pending(), [("stdin", "echo hello\n")], config
            )
        self.assertEqual(len(prompts), 1)
        self.assertIn("echo hello", prompts[0][0])
        self.assertIn("all commands require approval", prompts[0][1])
        self.assertEqual(
            denial, "[stdin blocked by user — safety level is set to 'all']"
        )

    def test_all_level_approved_benign_line_passes(self):
        config = {**DEFAULT_CONFIG, "command_safety": "all"}
        with patch("jarv.safety.prompt_confirmation", return_value=True):
            denial = _screen_stdin_text(
                self._pending(), [("stdin", "echo hello\n")], config
            )
        self.assertIsNone(denial)

    def test_all_level_risky_line_keeps_risky_denial_message(self):
        config = {**DEFAULT_CONFIG, "command_safety": "all"}
        with patch("jarv.safety.prompt_confirmation", return_value=False):
            denial = _screen_stdin_text(
                self._pending(), [("stdin", "rm -rf /\n")], config
            )
        self.assertIn("detected as risky", denial)

    def test_all_level_never_prompts_for_controls_or_plain_enter(self):
        config = {**DEFAULT_CONFIG, "command_safety": "all"}
        def boom(*_args, **_kwargs):  # pragma: no cover - must not be called
            raise AssertionError("prompt_confirmation should not run")
        actions = [
            ("stdin", "\n"),          # plain <ENTER>
            ("stdin_raw", "\x1b[B"),  # arrow key
            ("wait", None),
            ("ctrl_c", None),
            ("eof", None),
        ]
        with patch("jarv.safety.prompt_confirmation", side_effect=boom):
            denial = _screen_stdin_text(self._pending(), actions, config)
        self.assertIsNone(denial)

    def test_prompt_suspends_inline_card_live_around_confirmation(self):
        # The held-open card Live hides the cursor and repaints over the
        # prompt; it must stop before the question and resume after.
        events = []
        live = SimpleNamespace(
            stop=lambda: events.append("stop"),
            start=lambda refresh=False: events.append(f"start(refresh={refresh})"),
        )
        pending = SimpleNamespace(
            prepared=SimpleNamespace(cmd="bash"), live=live
        )
        config = {**DEFAULT_CONFIG, "command_safety": "risky"}

        def prompt(*_args, **_kwargs):
            events.append("prompt")
            return True

        with patch("jarv.safety.prompt_confirmation", side_effect=prompt):
            _screen_stdin_text(pending, [("stdin", "rm -rf /\n")], config)
        self.assertEqual(events, ["stop", "prompt", "start(refresh=True)"])

    def test_prompt_leaves_live_alone_when_confirm_handler_registered(self):
        # Heads-up mode: the handler owns the display; the (absent) inline
        # Live must not be touched.
        from jarv.safety import clear_confirm_handler, set_confirm_handler

        events = []
        live = SimpleNamespace(
            stop=lambda: events.append("stop"),
            start=lambda refresh=False: events.append("start"),
        )
        pending = SimpleNamespace(
            prepared=SimpleNamespace(cmd="bash"), live=live
        )
        config = {**DEFAULT_CONFIG, "command_safety": "risky"}
        set_confirm_handler(lambda request: True)
        self.addCleanup(clear_confirm_handler)

        denial = _screen_stdin_text(pending, [("stdin", "rm -rf /\n")], config)
        self.assertIsNone(denial)
        self.assertEqual(events, [])


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


class InvalidReplyStreakTests(unittest.TestCase):
    def _pending(self, rounds=5, streak=0):
        return SimpleNamespace(
            rounds=rounds,
            invalid_streak=streak,
            card=None,
            live=None,
            live_depth_cm=None,
            process=SimpleNamespace(kill_tree=lambda: None),
            unregister_cancel=None,
            input_markers=[],
            transcript_segments=["Choose:"],
            output_item={"output": "Choose:"},
            prepared=SimpleNamespace(cmd="menu", head_chars=None, tail_chars=None),
        )

    def _advance(self, pending, reply):
        renderer = SimpleNamespace(
            tool_calls=[],
            thought_started=time.perf_counter(),
            reply_text=reply,
        )
        return _advance_interactive_continuation(
            pending,
            renderer,
            [],
            config=dict(DEFAULT_CONFIG),
            cancellation_token=None,
            retained_store=None,
            ui=None,
            interactive_help={"sent": True},
        )

    def test_invalid_reply_refunds_round_and_records_rejection(self):
        pending = self._pending(rounds=5)
        items, still_pending = self._advance(pending, "<FROB>")
        self.assertIs(still_pending, pending)
        # The round was incremented then refunded — nothing touched the process.
        self.assertEqual(pending.rounds, 5)
        self.assertEqual(pending.invalid_streak, 1)
        self.assertIn("[reply rejected:", pending.output_item["output"])
        self.assertIn("[terminal reply not applied]", items[-1]["content"])

    def test_consecutive_invalid_replies_abort_the_interaction(self):
        killed = {}
        pending = self._pending(streak=_MAX_CONSECUTIVE_INVALID - 1)
        pending.process = SimpleNamespace(
            kill_tree=lambda: killed.setdefault("yes", True)
        )
        items, still_pending = self._advance(pending, "<FROB>")
        self.assertIsNone(still_pending)
        self.assertTrue(killed)
        self.assertIn("unparseable", items[-1]["content"])
        self.assertIn("aborted", pending.output_item["output"])

    def test_applied_reply_resets_invalid_streak(self):
        pending = self._pending(streak=2)
        delta = SimpleNamespace(
            to_model_output=lambda head_chars=None, tail_chars=None: "out",
            full_model_output=lambda: "out",
        )
        snapshot = SimpleNamespace(
            exited=False,
            command="menu",
            check_in=False,
            stdin_closed=False,
            to_delta_command_result=lambda: delta,
        )
        pending.process = SimpleNamespace(
            wait_until_idle=lambda **_kwargs: snapshot,
        )
        steps = []
        pending.card = SimpleNamespace(
            add_step=lambda *args, **kwargs: steps.append(args)
        )
        _items, still_pending = self._advance(pending, "<WAIT 1s>")
        self.assertIs(still_pending, pending)
        self.assertEqual(pending.invalid_streak, 0)
        self.assertTrue(steps)

    def test_blocked_reply_resets_streak_and_is_recorded(self):
        pending = self._pending(streak=3)
        with patch("jarv.safety.prompt_confirmation", return_value=False):
            _items, still_pending = self._advance(pending, "rm -rf /")
        self.assertIs(still_pending, pending)
        self.assertEqual(pending.invalid_streak, 0)
        # Blocked replies still consume a round; only unparseable ones refund.
        self.assertEqual(pending.rounds, 6)
        self.assertIn("blocked", pending.output_item["output"])


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
