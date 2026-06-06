from unittest.mock import patch

import pytest

from jarv.cancellation import CancellationToken, TurnCancelled
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
