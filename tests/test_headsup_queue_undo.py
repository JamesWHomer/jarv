"""Heads-up undo removes pending messages before touching saved exchanges."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from conftest import FakeLive, make_console
from jarv.cancellation import CancellationToken
from jarv.headsup import HeadsupApp
from jarv.text_editor import initialize_text_editor


@pytest.fixture
def app(monkeypatch):
    ready = threading.Event()
    ready.set()
    console, _ = make_console()
    app = HeadsupApp(
        {"provider": "openai", "model": "test-model"}, object(), args=None,
        agent_loader=({"module": SimpleNamespace()}, ready),
        handle_slash=Mock(), maybe_command=lambda *_: None,
        render_console=console,
    )
    app.handle_slash.return_value = (app.config, app.client)
    monkeypatch.setattr(app, "_sync_after_slash", Mock())
    return app


def notice_text(app):
    return "\n".join(line.plain for line in app._transcript_lines(100))


def test_idle_waiter_cannot_observe_worker_before_start(app, monkeypatch):
    original_start = threading.Thread.start
    exposed = []

    def checked_start(thread):
        if thread.name == "headsup-agent-turn":
            def inspect_worker():
                acquired = app.lock.acquire(blocking=False)
                try:
                    exposed.append(acquired and app._agent_thread is thread)
                finally:
                    if acquired:
                        app.lock.release()

            observer = threading.Thread(target=inspect_worker)
            original_start(observer)
            observer.join(timeout=2.0)
            assert not observer.is_alive()
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", checked_start)
    monkeypatch.setattr(app, "_run_agent_query_now", lambda query: None)
    app._queue_or_start_agent_query("prompt")
    app._wait_for_agent_idle(timeout=2.0)
    assert exposed == [False]
    assert not app._agent_busy


@pytest.mark.parametrize("incognito", [False, True])
@pytest.mark.parametrize(("rest", "remaining", "message"), [
    ([], ["oldest", "newer"], "Cancelled queued message:"),
    (["2"], ["oldest"], "Cancelled 2 queued messages."),
    (["99"], [], "Cancelled 3 queued messages."),
])
def test_undo_cancels_newest_pending_queries_without_interrupting_worker(
    app, monkeypatch, incognito, rest, remaining, message,
):
    started = threading.Event()
    release = threading.Event()
    token = CancellationToken()
    run_order = []
    active_complete = Mock()
    queued_complete = {query: Mock() for query in ("oldest", "newer")}
    aside_complete = Mock()

    def run_agent(query, *_args, **_kwargs):
        run_order.append(query)
        if query == "active":
            app.bind_cancel_token(token)
            started.set()
            release.wait(timeout=5.0)
            app.unbind_cancel_token()
        return SimpleNamespace(cancelled=False)

    app.live = FakeLive()
    app.incognito = incognito
    app._foreground_input_active = True
    app.agent_import["module"] = SimpleNamespace(run_agent=run_agent)
    monkeypatch.setattr(app, "_after_btw", aside_complete)
    try:
        app._run_agent_query("active", on_complete=active_complete)
        assert started.wait(timeout=1.0)
        for query, callback in queued_complete.items():
            app._run_agent_query(query, on_complete=callback)
        app._run_slash("/btw", ["cancelled", "aside"])
        initialize_text_editor(app.editor, "draft in progress")

        app._run_slash("/undo", rest)

        assert message in notice_text(app)
        assert [query for query, _ in app._queued_queries] == remaining
        assert app.editor["buffer"] == "draft in progress"
        assert app._agent_busy
        assert app._cancel_token is token
        assert not token.cancelled
        app.handle_slash.assert_not_called()
        app._sync_after_slash.assert_not_called()
    finally:
        release.set()
        app._wait_for_agent_idle(timeout=2.0)

    assert not app._agent_busy
    assert run_order == ["active", *remaining]
    active_complete.assert_called_once()
    aside_complete.assert_not_called()
    for query, callback in queued_complete.items():
        assert callback.call_count == int(query in remaining)


@pytest.mark.parametrize("rest", [["garbage"], ["0"], ["-1"], ["1", "2"]])
def test_invalid_undo_count_keeps_pending_messages(app, rest):
    token = CancellationToken()
    callback = Mock()
    app._agent_busy = True
    app._cancel_token = token
    app._queued_queries.extend([("first", None), ("second", callback)])

    app._run_slash("/undo", rest)

    assert list(app._queued_queries) == [("first", None), ("second", callback)]
    assert "Expected one positive integer count." in notice_text(app)
    assert "Usage: /undo [n]" in notice_text(app)
    assert not token.cancelled
    callback.assert_not_called()
    app.handle_slash.assert_not_called()
    app._sync_after_slash.assert_not_called()


def test_undo_with_no_queue_keeps_active_history_guard(app):
    app._agent_busy = True

    app._run_slash("/undo", [])

    assert "/undo is unavailable during an active turn" in notice_text(app)
    app.handle_slash.assert_not_called()


def test_undo_with_no_queue_and_idle_worker_routes_to_history(app):
    app._run_slash("/undo", ["2"])

    app.handle_slash.assert_called_once_with(
        "/undo", ["2"], app.config, app.client, app.args, True,
    )
    app._sync_after_slash.assert_called_once_with("/undo")
