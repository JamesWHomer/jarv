"""Restart must release the old UI before a new process owns the terminal."""

from unittest.mock import Mock

import pytest

from conftest import FakeLive
from jarv import headsup
from jarv.commands import UpdateOutcome
from jarv.safety import confirm_handler_active
from jarv.text_editor import initialize_text_editor


@pytest.mark.parametrize("draft", ["/restart", "/RESTART", "/resta", "jarv /restart"])
def test_restart_relaunches_only_after_terminal_and_client_cleanup(
    monkeypatch, headsup_app_factory, neutral_tui_terminal, draft,
):
    from jarv import restart

    app = headsup_app_factory()
    client = Mock()
    app.client = client
    session_id = app.session_context.session_id
    initialize_text_editor(app.editor, draft)
    keys = [("ENTER", 1)]
    monkeypatch.setattr(headsup, "Live", FakeLive)
    monkeypatch.setattr(headsup, "_key_available", lambda: bool(keys))
    monkeypatch.setattr(headsup, "_read_key_with_repeats", lambda **_: keys.pop(0))
    monkeypatch.setattr(headsup, "enable_mouse_wheel_reporting", lambda: None)
    disable_mouse = Mock()
    monkeypatch.setattr(headsup, "disable_mouse_capture", disable_mouse)
    monkeypatch.setattr(headsup, "HeadsupApp", lambda *a, **kw: app)

    def relaunch(args, active_session):
        assert args is app.args
        assert active_session == session_id
        assert app._closing
        assert app.live is None
        assert not confirm_handler_active()
        client.close.assert_called_once_with()
        disable_mouse.assert_called_once_with()
        assert FakeLive.instances[-1].exited
        return 7

    relaunch_mock = Mock(side_effect=relaunch)
    monkeypatch.setattr(restart, "restart_heads_up", relaunch_mock)
    with pytest.raises(SystemExit) as result:
        headsup.run_heads_up_mode(
            app.config, client, args=app.args, agent_loader=None,
            handle_slash=app.handle_slash, maybe_command=app.maybe_command,
        )
    assert result.value.code == 7
    relaunch_mock.assert_called_once()


def test_restart_rejects_arguments_without_stopping(headsup_app_factory):
    app = headsup_app_factory()
    assert app._run_slash("/restart", ["extra"]) is None
    assert app.result is None
    assert "does not accept arguments" in app._notice.renderable.plain


def test_restart_waits_for_update_and_staged_update_requires_exit(headsup_app_factory):
    app = headsup_app_factory()
    task = headsup._UpdateTask(app)
    app._update_task = task
    assert app._run_slash("/restart", []) is None
    assert "update is running" in app._notice.renderable.plain

    task._finish(UpdateOutcome("staged", "Update staged.", latest="9.9.9"))
    assert app._update_task is None
    assert app._run_slash("/restart", []) is None
    assert "Exit jarv" in app._notice.renderable.plain

    # A later failed check must not forget the already staged installer.
    headsup._UpdateTask(app)._finish(UpdateOutcome("failed", "Network error."))
    assert app._run_slash("/restart", []) is None


def test_restart_allowed_after_finished_update(headsup_app_factory):
    app = headsup_app_factory()
    task = headsup._UpdateTask(app)
    app._update_task = task
    task._finish(UpdateOutcome("updated", "Updated successfully.", latest="9.9.9"))
    assert app._run_slash("/restart", []) == "restart"


def test_normal_exit_does_not_relaunch(monkeypatch):
    from jarv import restart

    app = Mock()
    app.run.return_value = "exit"
    monkeypatch.setattr(headsup, "HeadsupApp", Mock(return_value=app))
    relaunch = Mock()
    monkeypatch.setattr(restart, "restart_heads_up", relaunch)
    headsup.run_heads_up_mode(
        {}, None, args=None, agent_loader=None, handle_slash=Mock(), maybe_command=Mock(),
    )
    relaunch.assert_not_called()
