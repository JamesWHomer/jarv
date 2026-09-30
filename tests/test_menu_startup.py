"""Startup contracts: lightweight menus, usable input, and cancellable setup."""
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from conftest import FakeLive, make_console
from jarv import cli
from jarv.command_input import TextInput
from jarv.headsup import HeadsupApp
from scripts.benchmark.benchmark_coldstart import CHILD, ROOT, SEED


@pytest.mark.parametrize("args", [
    ["--incognito"], ["/settings"], ["/setup"], ["/sessions"], ["/tree"],
    ["/help"], ["/config"], ["/usage"],
])
def test_menu_reaches_real_input_without_loading_agent_or_network(tmp_path, args):
    config_dir = tmp_path / ".jarv"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps({
        "provider": "ollama", "model": "llama3.2", "check_updates": False,
    }), encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("JARV_BENCH_")}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path),
               WT_SESSION="jarv-menu-test", TERM="xterm-256color",
               COLUMNS="100", LINES="30", PYTHONIOENCODING="utf-8",
               JARV_BENCH_MENU_INPUT="1")
    subprocess.run([sys.executable, "-c", SEED], cwd=ROOT, env=env,
                   capture_output=True, check=True, timeout=10)
    result = subprocess.run([sys.executable, "-c", CHILD, "menu", *args],
                            cwd=ROOT, env=env, capture_output=True,
                            encoding="utf-8", timeout=10)
    assert result.returncode == 0, result.stderr
    assert "BENCH_INPUT_PAINT " in result.stderr
    modules = next(line.split(" ", 1)[1].split(",") for line in result.stderr.splitlines()
                   if line.startswith("BENCH_MODULES "))
    forbidden = {"jarv.agent", "jarv.orchestrator", "httpx", "httpcore", "jarv.standalone"}
    # The session browser renders Markdown previews on a background worker.
    # Whether that worker imports Markdown before the input repaint is a race.
    if args != ["/sessions"]:
        forbidden.add("rich.markdown")
    assert not forbidden.intersection(modules)


def make_app():
    ready = threading.Event()
    ready.set()
    run = Mock(return_value=SimpleNamespace(cancelled=False))
    console, output = make_console()
    app = HeadsupApp(
        {"provider": "ollama", "model": "llama3.2", "check_updates": False}, None,
        args=SimpleNamespace(incognito=True, new=False),
        agent_loader=({"module": SimpleNamespace(run_agent=run)}, ready),
        handle_slash=lambda command, rest, config, client, args, hint: (config, client),
        maybe_command=lambda first, rest: None, render_console=console,
    )
    app.live = FakeLive()
    return app, run


@pytest.mark.parametrize("cancel", [False, True])
def test_slow_client_setup_keeps_input_live_and_honors_cancel(monkeypatch, cancel):
    app, run = make_app()
    started, release = threading.Event(), threading.Event()
    client = Mock()

    def create(config):
        started.set()
        assert release.wait(5)
        return client

    create_mock = Mock(side_effect=create)
    monkeypatch.setattr("jarv.provider.create_client", create_mock)
    app._run_agent_query("first prompt")
    try:
        assert started.wait(5)
        app.on_key(TextInput("next draft"), 1)
        assert app.editor["buffer"] == "next draft"
        assert app._agent_busy
        if cancel:
            app.on_key("ESC", 1)  # clear the draft
            app.on_key("ESC", 1)  # cancel setup
    finally:
        release.set()
        app._wait_for_agent_idle(timeout=5)
    assert not app._agent_busy
    if cancel:
        run.assert_not_called()
        client.close.assert_called_once_with()
        assert app.client is None
    else:
        assert run.call_args.args[2] is client
        app._run_agent_query("second prompt")
        app._wait_for_agent_idle(timeout=5)
        assert run.call_count == 2
        create_mock.assert_called_once()
        client.close.assert_not_called()


def test_client_failure_is_visible_and_retryable(monkeypatch):
    app, run = make_app()
    client = Mock()
    create = Mock(side_effect=[RuntimeError("connection setup failed"), client])
    monkeypatch.setattr("jarv.provider.create_client", create)
    app._run_agent_query("first")
    app._wait_for_agent_idle(timeout=5)
    assert not app._agent_busy
    assert app._cancel_token is None
    assert any("connection setup failed" in str(entry.renderable) for entry in app.entries)
    run.assert_not_called()
    app._run_agent_query("retry")
    app._wait_for_agent_idle(timeout=5)
    run.assert_called_once()
    assert run.call_args.args[2] is client


def test_config_reload_does_not_create_an_unused_transport(monkeypatch):
    args = cli._build_parser().parse_args([])
    config = {"provider": "ollama", "model": "old"}
    refreshed = {"provider": "ollama", "model": "new"}
    monkeypatch.setattr(cli, "load_config", lambda: refreshed)
    monkeypatch.setattr(cli, "validate_config", lambda config: True)
    create = Mock(side_effect=AssertionError("opening settings must not create a client"))
    monkeypatch.setattr("jarv.provider.create_client", create)
    actual, client = cli._reload_heads_up_runtime(config, None, args)
    assert actual == refreshed
    assert client is None
    create.assert_not_called()
