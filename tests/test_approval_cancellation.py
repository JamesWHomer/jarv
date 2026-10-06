"""Cancellation at the approval/execution boundary must not start a process."""

from unittest.mock import Mock, patch

import pytest

from jarv.agent import _dispatch_run_command_with_ui
from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.config import DEFAULT_CONFIG


@pytest.mark.parametrize("cancel_during", ["approval", "display"])
def test_cancelled_interactive_command_never_starts_process(cancel_during):
    token = CancellationToken()
    config = {**DEFAULT_CONFIG, "command_safety": "all", "audit": False,
              "interactive_commands": True}
    ui = Mock()

    def approve(request):
        if cancel_during == "approval":
            token.cancel()
        return True

    if cancel_during == "display":
        ui.show_tool_card.side_effect = lambda _card: token.cancel()

    with (
        patch("jarv.safety._confirm_handler", approve),
        patch("jarv.agent.InteractiveCommandProcess.start") as start,
        pytest.raises(TurnCancelled),
    ):
        _dispatch_run_command_with_ui(
            {"command": "echo hello"}, config, cancellation_token=token, ui=ui,
        )

    start.assert_not_called()
