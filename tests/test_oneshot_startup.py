"""Startup ordering and cleanup, without machine-dependent timing assertions."""

import builtins
import io
import subprocess
import sys
import threading
from unittest.mock import Mock

import pytest
from rich.console import Console

from jarv import agent, cli
from jarv.config import DEFAULT_CONFIG
from jarv.response_wait import start_response_wait


def test_agent_import_defers_rendering_and_network_libraries():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import jarv.agent; "
            "blocked = ['rich.markdown', 'httpx', 'httpcore', 'jarv.standalone']; "
            "assert not any(name in sys.modules for name in blocked), "
            "[name for name in blocked if name in sys.modules]"
        )],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_wait_indicator_paints_synchronously_and_restores_console():
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, legacy_windows=False)
    indicator, live = start_response_wait(True, 0, console=console)
    try:
        assert "Waiting" in output.getvalue()
        assert live.is_started
        assert indicator.has_reasoning is False
    finally:
        live.stop()
    assert not live.is_started
    assert "\x1b[?25h" in output.getvalue()


def test_pipe_has_no_startup_indicator():
    factory = Mock(side_effect=AssertionError("must not create a Live for a pipe"))
    assert start_response_wait(False, 0, console=None, live_factory=factory) == (None, None)


def test_client_setup_overlaps_complete_instruction_collection(monkeypatch):
    context_started = threading.Event()
    client_started = threading.Event()
    client = Mock()

    def instructions(config, *, cwd):
        context_started.set()
        assert client_started.wait(5), "HTTP setup did not overlap context collection"
        assert cwd == "project-dir"
        return "system prompt + complete project instructions"

    def create(config):
        client_started.set()
        assert context_started.wait(5)
        return client

    monkeypatch.setattr(agent, "build_instructions", instructions)
    monkeypatch.setattr(agent, "create_client", create)
    assert agent._prepare_client_and_instructions({}, None, cwd="project-dir") == (
        client, "system prompt + complete project instructions",
    )
    client.close.assert_not_called()


def test_existing_client_is_reused(monkeypatch):
    client = Mock()
    monkeypatch.setattr(agent, "create_client", Mock(side_effect=AssertionError("new client")))
    monkeypatch.setattr(agent, "build_instructions", lambda config, *, cwd: cwd)
    assert agent._prepare_client_and_instructions({}, client, cwd="cwd") == (client, "cwd")


def test_context_failure_closes_new_client(monkeypatch):
    client = Mock()
    monkeypatch.setattr(agent, "create_client", lambda config: client)
    monkeypatch.setattr(agent, "build_instructions", Mock(side_effect=ValueError("context failed")))
    with pytest.raises(ValueError, match="context failed"):
        agent._prepare_client_and_instructions({}, None, cwd="cwd")
    client.close.assert_called_once_with()


@pytest.mark.parametrize("interrupted", [False, True])
def test_cli_paints_before_agent_import_and_cleans_up(monkeypatch, interrupted):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    output = Terminal()
    console = Console(file=output, force_terminal=True, legacy_windows=False)
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(sys, "argv", ["jarv", "--incognito", "hello"])
    monkeypatch.setattr(cli, "_console", lambda: console)
    monkeypatch.setattr(cli, "load_config", lambda: {
        **DEFAULT_CONFIG, "provider": "ollama", "model": "llama3.2", "check_updates": False,
    })
    monkeypatch.setattr(cli, "_print_previous_update_result", lambda: None)
    monkeypatch.setattr(cli, "_print_previous_uninstall_result", lambda: None)
    run = Mock(return_value=agent.AgentRunResult())
    monkeypatch.setattr(agent, "run_agent", run)
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "agent" and "run_agent" in fromlist:
            assert "Waiting" in output.getvalue()
            if interrupted:
                raise KeyboardInterrupt
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    if interrupted:
        with pytest.raises(SystemExit) as stopped:
            cli.main()
        assert stopped.value.code == 130
        run.assert_not_called()
    else:
        cli.main()
        assert run.call_args.kwargs["startup_wait"] is not None
        assert not run.call_args.kwargs["startup_wait"][2].is_started
    assert sys.stdout is output
    assert "\x1b[?25h" in output.getvalue()
