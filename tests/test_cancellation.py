from unittest.mock import patch

import pytest

from jarv import cancellation
from jarv.cancellation import CancellationToken, TurnCancelled, cancel_token_on_sigint
from jarv.safety import prompt_confirmation


def test_cancellation_token_runs_registered_cleanup_once():
    calls = []
    token = CancellationToken()
    token.register(lambda: calls.append("closed"))

    token.cancel()
    token.cancel()

    assert calls == ["closed"]
    with pytest.raises(TurnCancelled):
        token.throw_if_cancelled()


def test_command_confirmation_ctrl_c_propagates():
    with patch("jarv.safety.console.input", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            prompt_confirmation("dangerous", "test")


def test_sigint_scope_cancels_token_before_keyboard_interrupt(monkeypatch):
    token = CancellationToken()
    handlers = {cancellation.signal.SIGINT: cancellation.signal.default_int_handler}

    def fake_signal(signum, handler):
        previous = handlers.get(signum, cancellation.signal.default_int_handler)
        handlers[signum] = handler
        return previous

    monkeypatch.setattr(cancellation.signal, "signal", fake_signal)

    with pytest.raises(KeyboardInterrupt):
        with cancel_token_on_sigint(token):
            handlers[cancellation.signal.SIGINT](cancellation.signal.SIGINT, None)

    assert token.cancelled
    assert handlers[cancellation.signal.SIGINT] is cancellation.signal.default_int_handler
