"""Tests for the safety confirm funnel and its display routing."""

import threading
import unittest
from unittest.mock import MagicMock, patch

from rich.text import Text

from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.display import live_display_depth, track_live_display
from jarv.safety import (
    ConfirmRequest,
    check_command,
    clear_confirm_handler,
    confirm_handler_active,
    prompt_confirmation,
    request_confirmation,
    set_confirm_handler,
)


class ConfirmHandlerRegistryTests(unittest.TestCase):
    def tearDown(self):
        clear_confirm_handler()

    def test_set_and_clear_handler(self):
        self.assertFalse(confirm_handler_active())
        set_confirm_handler(lambda request: True)
        self.assertTrue(confirm_handler_active())
        clear_confirm_handler()
        self.assertFalse(confirm_handler_active())

    def test_handler_decision_is_returned(self):
        request = ConfirmRequest(body=Text("$ rm -rf /"))
        set_confirm_handler(lambda _request: True)
        self.assertTrue(request_confirmation(request))
        set_confirm_handler(lambda _request: False)
        self.assertFalse(request_confirmation(request))

    def test_handler_exception_denies_instead_of_propagating(self):
        # A broken prompt must never take the turn down: the run_command
        # dispatch chain has no exception guard around the safety gate.
        def boom(_request):
            raise RuntimeError("display broke")

        set_confirm_handler(boom)
        self.assertFalse(request_confirmation(ConfirmRequest(body=Text("x"))))

    def test_handler_turn_cancelled_propagates(self):
        def cancel(_request):
            raise TurnCancelled

        set_confirm_handler(cancel)
        with self.assertRaises(TurnCancelled):
            request_confirmation(ConfirmRequest(body=Text("x")))

    def test_prompt_confirmation_reaches_handler_with_metadata(self):
        seen = {}

        def handler(request):
            seen["command"] = request.command
            seen["reason"] = request.reason
            seen["kind"] = request.kind
            seen["question"] = request.question
            return True

        set_confirm_handler(handler)
        approved = prompt_confirmation(
            "rm -rf build", "recursive deletion", kind="stdin",
            question="Allow this input?",
        )
        self.assertTrue(approved)
        self.assertEqual(seen["command"], "rm -rf build")
        self.assertEqual(seen["reason"], "recursive deletion")
        self.assertEqual(seen["kind"], "stdin")
        self.assertEqual(seen["question"], "Allow this input?")

    def test_check_command_routes_denial_through_handler(self):
        set_confirm_handler(lambda _request: False)
        allowed, denial = check_command("rm -rf /", "risky", audit=False)
        self.assertFalse(allowed)
        self.assertIn("denied by user", denial)

    def test_check_command_routes_approval_through_handler(self):
        set_confirm_handler(lambda _request: True)
        allowed, denial = check_command("rm -rf /", "risky", audit=False)
        self.assertTrue(allowed)
        self.assertEqual(denial, "")

    def test_audit_gate_hands_auditor_state_to_handler(self):
        # With a handler registered the non-TTY console fallback is skipped
        # and the handler owns the auditor's async state.
        seen = {}

        def handler(request):
            self.assertIsNotNone(request.audit_state)
            # The auditor thread runs concurrently; wait for its verdict the
            # same way a real display would.
            deadline = threading.Event()
            for _ in range(200):
                if request.audit_state.get("done"):
                    break
                deadline.wait(0.01)
            seen["state"] = dict(request.audit_state)
            return bool(request.audit_state.get("allow"))

        set_confirm_handler(handler)
        with patch(
            "jarv.auditor.audit_command", return_value=(True, "read-only")
        ):
            allowed, denial = check_command("rm -rf /", "risky", audit=True)
        self.assertTrue(allowed)
        self.assertEqual(denial, "")
        self.assertTrue(seen["state"]["done"])
        self.assertTrue(seen["state"]["allow"])
        self.assertEqual(seen["state"]["reason"], "read-only")


class ConsoleFallbackTests(unittest.TestCase):
    def test_eof_on_console_prompt_denies(self):
        fake_console = MagicMock()
        fake_console.input.side_effect = EOFError
        with patch("jarv.safety.console", fake_console):
            approved = prompt_confirmation("rm -rf /", "recursive deletion")
        self.assertFalse(approved)

    def test_audit_display_avoids_nested_live_across_threads(self):
        # The live-vs-plain poll choice must see a Live held by *another*
        # thread (the spawn panel on the main thread while a subagent worker
        # confirms), or the worker starts a second Live and Rich raises.
        audit_state = {"done": True, "allow": True, "reason": "ok"}
        request = ConfirmRequest(body=Text("x"), audit_state=audit_state)

        entered = threading.Event()
        release = threading.Event()

        def hold_live():
            with track_live_display():
                entered.set()
                release.wait(timeout=5.0)

        thread = threading.Thread(target=hold_live, daemon=True)
        thread.start()
        self.assertTrue(entered.wait(timeout=5.0))
        try:
            with patch(
                "jarv.safety._audit_poll_without_live", return_value=True
            ) as plain_poll, patch(
                "jarv.safety._live_audit_poll", return_value=True
            ) as live_poll:
                self.assertTrue(request_confirmation(request))
            plain_poll.assert_called_once()
            live_poll.assert_not_called()
        finally:
            release.set()
            thread.join(timeout=5.0)

    def test_audit_display_uses_live_poll_when_no_live_active(self):
        audit_state = {"done": True, "allow": True, "reason": "ok"}
        request = ConfirmRequest(body=Text("x"), audit_state=audit_state)
        with patch(
            "jarv.safety._audit_poll_without_live", return_value=True
        ) as plain_poll, patch(
            "jarv.safety._live_audit_poll", return_value=True
        ) as live_poll:
            self.assertTrue(request_confirmation(request))
        live_poll.assert_called_once()
        plain_poll.assert_not_called()


class LiveDisplayDepthTests(unittest.TestCase):
    def test_depth_is_process_wide_across_threads(self):
        self.assertEqual(live_display_depth(), 0)
        entered = threading.Event()
        release = threading.Event()

        def hold():
            with track_live_display():
                entered.set()
                release.wait(timeout=5.0)

        thread = threading.Thread(target=hold, daemon=True)
        thread.start()
        self.assertTrue(entered.wait(timeout=5.0))
        try:
            self.assertEqual(live_display_depth(), 1)
        finally:
            release.set()
            thread.join(timeout=5.0)
        self.assertEqual(live_display_depth(), 0)

    def test_depth_nests(self):
        with track_live_display():
            with track_live_display():
                self.assertEqual(live_display_depth(), 2)
            self.assertEqual(live_display_depth(), 1)
        self.assertEqual(live_display_depth(), 0)


class CancellationPropagationTests(unittest.TestCase):
    def tearDown(self):
        clear_confirm_handler()

    def test_cancelled_token_reaches_handler(self):
        token = CancellationToken()
        token.cancel()

        def handler(request):
            request.cancellation_token.throw_if_cancelled()
            return True  # pragma: no cover - throw above raises

        set_confirm_handler(handler)
        with self.assertRaises(TurnCancelled):
            request_confirmation(
                ConfirmRequest(body=Text("x"), cancellation_token=token)
            )


if __name__ == "__main__":
    unittest.main()
