"""Provider clients have explicit owners across turns and settings changes."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jarv import agent, cli
from jarv.cancellation import TurnCancelled
from jarv.config_schema import build_default_config
from jarv.provider import ProviderError, StreamDone, TextDelta


@pytest.mark.parametrize("failure", [None, ProviderError("failed"), TurnCancelled()])
@pytest.mark.parametrize("borrowed", [False, True])
def test_run_closes_only_owned_clients(monkeypatch, failure, borrowed):
    client = Mock(spec=["close"])
    monkeypatch.setattr(agent, "create_client", lambda config: client)
    monkeypatch.setattr(agent, "build_instructions", lambda *args, **kwargs: "system")

    def stream(*args, **kwargs):
        if failure is not None:
            raise failure
        yield TextDelta("answer")
        yield StreamDone(None)

    monkeypatch.setattr(agent, "stream_response", stream)
    result = agent.run_agent(
        "hello", build_default_config(), client=client if borrowed else None,
        incognito=True, ui=SimpleNamespace(),
    )
    assert result.cancelled is isinstance(failure, TurnCancelled)
    assert result.error == ("failed" if isinstance(failure, ProviderError) else None)
    assert client.close.call_count == (0 if borrowed else 1)


def test_instruction_failure_closes_new_client_once(monkeypatch):
    client = Mock(spec=["close"])
    monkeypatch.setattr(agent, "create_client", lambda config: client)
    monkeypatch.setattr(agent, "build_instructions", Mock(side_effect=ValueError("bad context")))
    result = agent.run_agent("hello", build_default_config(), incognito=True, ui=SimpleNamespace())
    assert result.error == "bad context"
    client.close.assert_called_once_with()


def test_close_failure_does_not_replace_success(monkeypatch):
    client = Mock(spec=["close"])
    client.close.side_effect = RuntimeError("cleanup failed")
    monkeypatch.setattr(agent, "create_client", lambda config: client)
    monkeypatch.setattr(agent, "build_instructions", lambda *args, **kwargs: "system")
    monkeypatch.setattr(agent, "stream_response", lambda *args, **kwargs: iter([
        TextDelta("answer"), StreamDone(None),
    ]))
    result = agent.run_agent("hello", build_default_config(), incognito=True, ui=SimpleNamespace())
    assert result.text == "answer"
    assert result.error is None
    client.close.assert_called_once_with()


def _install_reload(app, monkeypatch, replacement):
    config = {**app.config, "base_url": "http://localhost:9999/v1"}
    app.args = cli._build_parser().parse_args([])
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr("jarv.provider.create_client", lambda config: replacement)
    app.handle_slash = lambda command, rest, config, client, args, hint: (
        cli._reload_heads_up_runtime(config, client, args)
    )


def test_idle_reload_and_shutdown_close_clients(monkeypatch, headsup_app_factory):
    app = headsup_app_factory()
    old, replacement = Mock(spec=["close"]), Mock(spec=["close"])
    app.client = old
    _install_reload(app, monkeypatch, replacement)
    app._run_slash("/set", ["base_url", "http://localhost:9999/v1"])
    assert app.client is replacement
    old.close.assert_called_once_with()
    replacement.close.assert_not_called()
    monkeypatch.setattr("jarv.headsup.disable_mouse_capture", lambda: None)
    app.on_stop()
    assert app.client is None
    replacement.close.assert_called_once_with()
    app.on_stop()
    replacement.close.assert_called_once_with()


def test_reload_keeps_active_client_until_turn_finishes(monkeypatch, headsup_app_factory):
    entered, release = threading.Event(), threading.Event()
    old, replacement = Mock(spec=["close"]), Mock(spec=["close"])
    observed = []

    def run(query, config, client, **kwargs):
        observed.append(client)
        if query == "first":
            entered.set()
            assert release.wait(5)
            old.close.assert_not_called()
        return SimpleNamespace(cancelled=False)

    app = headsup_app_factory(run_agent=run)
    app.client = old
    _install_reload(app, monkeypatch, replacement)
    app._run_agent_query("first")
    try:
        assert entered.wait(5)
        app._run_slash("/set", ["base_url", "http://localhost:9999/v1"])
        old.close.assert_not_called()
        app._run_agent_query("second")
    finally:
        release.set()
        app._wait_for_agent_idle(timeout=5)
    assert observed == [old, replacement]
    old.close.assert_called_once_with()
    replacement.close.assert_not_called()


def test_shutdown_defers_close_until_slow_worker_releases_client(monkeypatch, headsup_app_factory):
    entered, release = threading.Event(), threading.Event()
    client = Mock(spec=["close"])

    def run(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        client.close.assert_not_called()
        return SimpleNamespace(cancelled=False)

    app = headsup_app_factory(run_agent=run)
    app.client = client
    wait = app._wait_for_agent_idle
    monkeypatch.setattr(app, "_wait_for_agent_idle", lambda **kwargs: None)
    monkeypatch.setattr("jarv.headsup.disable_mouse_capture", lambda: None)
    app._run_agent_query("first")
    try:
        assert entered.wait(5)
        app.on_stop()
        client.close.assert_not_called()
        app._run_agent_query("must not start after shutdown")
    finally:
        release.set()
        wait(timeout=5)
    assert not app._agent_busy
    assert app.client is None
    client.close.assert_called_once_with()
