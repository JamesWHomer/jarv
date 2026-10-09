"""Relaunch contracts without replacing pytest or opening another terminal."""

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jarv import __version__, cli, history, restart


def _parsed_restart(args, session_id="current-session"):
    command = restart._restart_command(args, session_id)
    prefix_length = 1 if getattr(restart.sys, "frozen", False) else 4
    return cli._build_parser().parse_args(command[prefix_length:])


def test_runtime_flags_round_trip_without_prompt_or_new(monkeypatch):
    monkeypatch.delattr(restart.sys, "frozen", raising=False)
    parser = cli._build_parser()
    args = parser.parse_args([
        "--provider", "openai", "--model", "test-model", "--effort", "high",
        "--timeout", "12", "--base-url", "https://example.test/v1",
        "--service-tier", "priority", "--command-safety", "all",
        "--max-turns", "15", "--run-timeout", "80", "--system=-instructions",
        "--tools", "read,web_search", "--no-project-context", "--no-update-check",
        "--no-color", "--new", "old prompt that must not replay",
        "--config", "colour=false", "--config", "command_timeout=20",
        "--config", 'disabled_tools=["edit"]',
        "--config", 'api_keys={"openai":"test"}',
        "--config", "system_prompt=literal $value and = signs",
    ])
    relaunched = _parsed_restart(args)
    for name, value in vars(args).items():
        if name not in {"new", "query"}:
            assert getattr(relaunched, name) == value, name
    assert relaunched.new is False
    assert relaunched.query == []
    assert args.new is True


@pytest.mark.parametrize("incognito", [False, True])
def test_restart_keeps_default_session_mode_and_privacy(incognito):
    args = cli._build_parser().parse_args(["--incognito"] if incognito else [])
    relaunched = _parsed_restart(args)
    assert relaunched.incognito is incognito
    assert relaunched.session is None


def test_named_restart_uses_current_selected_session():
    args = cli._build_parser().parse_args(["--session", "original"])
    relaunched = _parsed_restart(args, "selected-later")
    assert relaunched.session == "selected-later"
    assert args.session == "original"


def test_relative_system_file_and_cwd_use_original_paths(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    args = cli._build_parser().parse_args(["--cwd", "project", "--system-file", "system.txt"])
    args._restart_invocation_cwd = str(tmp_path)
    monkeypatch.chdir(project)
    relaunched = _parsed_restart(args)
    assert Path(relaunched.cwd) == project
    assert Path(relaunched.system_file) == tmp_path / "system.txt"


def test_system_file_without_original_directory_uses_loaded_text():
    args = cli._build_parser().parse_args(["--system-file", "system.txt"])
    args.system_file_text = "loaded system text\n--literal-flag"
    relaunched = _parsed_restart(args)
    assert relaunched.system == args.system_file_text
    assert relaunched.system_file is None


def test_empty_system_and_no_tools_survive_restart():
    args = cli._build_parser().parse_args(["--system", "", "--no-tools"])
    relaunched = _parsed_restart(args)
    assert relaunched.system == ""
    assert relaunched.no_tools is True


@pytest.mark.parametrize("frozen", [False, True])
def test_restart_uses_python_bootstrap_or_frozen_executable(monkeypatch, frozen):
    monkeypatch.setattr(restart.sys, "executable", "python")
    monkeypatch.setattr(restart.sys, "frozen", frozen, raising=False)
    expected = ["python"] if frozen else [
        "python", "-c", restart._PYTHON_RESTART_BOOTSTRAP,
        str(Path(restart.__file__).resolve().parent.parent),
    ]
    assert restart._restart_command(None, "session") == expected


def test_restart_bootstrap_uses_original_package_from_conflicting_project(monkeypatch, tmp_path):
    monkeypatch.delattr(restart.sys, "frozen", raising=False)
    shadow = tmp_path / "jarv"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("raise RuntimeError('wrong jarv imported')", encoding="utf-8")
    command = restart._restart_command(None, "session")
    # A real interpreter verifies import/argv behavior; --version exits before
    # configuration, networking, or entering a heads-up environment.
    result = subprocess.run(command + ["--version"], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"jarv {__version__}"


def _prepare_process_restart(monkeypatch, *, windows, frozen=False):
    execute = Mock()
    monkeypatch.setattr(restart, "os", SimpleNamespace(
        name="nt" if windows else "posix", environ={"EXISTING": "value"}, execve=execute,
    ))
    monkeypatch.setattr(restart, "_restart_command", Mock(return_value=["jarv", "--no-color"]))
    monkeypatch.setattr(history, "detect_terminal", lambda: ("terminal-before", "Original terminal"))
    monkeypatch.setattr(restart.sys, "frozen", frozen, raising=False)
    state = Mock()
    monkeypatch.setitem(restart.sys.modules, "jarv.shell", SimpleNamespace(_session_shell_state=state))
    return execute, state


def test_posix_exec_closes_idle_shell_and_preserves_terminal_environment(monkeypatch):
    execute, state = _prepare_process_restart(monkeypatch, windows=False)
    execute.side_effect = lambda *a: state.close.assert_called_once_with()
    spawn = Mock()
    monkeypatch.setattr(restart.subprocess, "Popen", spawn)
    restart.restart_heads_up(None, "session")
    executable, command, environment = execute.call_args.args
    assert executable == "jarv"
    assert command == ["jarv", "--no-color"]
    assert environment["EXISTING"] == "value"
    assert "PYTHONPATH" not in environment
    assert json.loads(environment[history._RESTART_TERMINAL_ENV]) == ["terminal-before", "Original terminal"]
    assert restart.os.environ == {"EXISTING": "value"}
    spawn.assert_not_called()


def test_source_restart_does_not_change_existing_pythonpath(monkeypatch):
    execute, _state = _prepare_process_restart(monkeypatch, windows=False)
    restart.os.environ["PYTHONPATH"] = "existing-import-root"
    restart.restart_heads_up(None, "session")
    environment = execute.call_args.args[2]
    assert environment["PYTHONPATH"] == "existing-import-root"


def test_windows_waits_through_ctrl_c_and_propagates_child_exit(monkeypatch):
    execute, state = _prepare_process_restart(monkeypatch, windows=True, frozen=True)
    child = Mock()
    child.wait.side_effect = [KeyboardInterrupt, KeyboardInterrupt, 7]
    spawn = Mock(return_value=child)
    monkeypatch.setattr(restart.subprocess, "Popen", spawn)
    assert restart.restart_heads_up(None, "session") == 7
    state.close.assert_called_once_with()
    execute.assert_not_called()
    assert child.wait.call_count == 3
    assert spawn.call_args.kwargs["env"]["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
    assert set(spawn.call_args.kwargs) == {"env"}  # Inherit the same console and streams.


def test_restart_does_not_import_unused_shell(monkeypatch):
    execute, _state = _prepare_process_restart(monkeypatch, windows=False)
    monkeypatch.delitem(restart.sys.modules, "jarv.shell")
    restart.restart_heads_up(None, "session")
    execute.assert_called_once()
    assert "jarv.shell" not in restart.sys.modules


def test_launch_failure_propagates_without_leaking_terminal_marker(monkeypatch):
    execute, _state = _prepare_process_restart(monkeypatch, windows=False)
    execute.side_effect = OSError("could not relaunch")
    with pytest.raises(OSError, match="could not relaunch"):
        restart.restart_heads_up(None, "session")
    assert history._RESTART_TERMINAL_ENV not in restart.os.environ


def test_relaunch_terminal_identity_is_consumed_and_survives_parent_change(monkeypatch):
    marker = json.dumps(["parent-original", "Original shell"])
    monkeypatch.setenv(history._RESTART_TERMINAL_ENV, marker)
    original = history._consume_restart_terminal()
    monkeypatch.setattr(history, "_restart_terminal", original)
    monkeypatch.setattr(history.os, "getppid", lambda: 9999)
    assert history.detect_terminal() == ("parent-original", "Original shell")
    assert history._RESTART_TERMINAL_ENV not in os.environ
    assert history._consume_restart_terminal() is None


@pytest.mark.parametrize("marker", ["invalid json", '"text"', "[]", '["", "label"]', '["id", 1]'])
def test_invalid_relaunch_terminal_marker_is_discarded(monkeypatch, marker):
    monkeypatch.setenv(history._RESTART_TERMINAL_ENV, marker)
    assert history._consume_restart_terminal() is None
    assert history._RESTART_TERMINAL_ENV not in os.environ


def test_default_session_binding_survives_restart_without_named_override(monkeypatch, tmp_path):
    monkeypatch.setattr(history, "SESSIONS_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(history, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(history, "_restart_terminal", ("original-terminal", "Original terminal"))
    monkeypatch.setattr(history, "_session_override", None)
    history.set_terminal_session("selected-session")
    before = history.prepare_session_context(mark_message=True)
    monkeypatch.setattr(history.os, "getppid", lambda: 12345)
    after = history.prepare_session_context(persist_metadata=False)
    assert before.session_id == after.session_id == "selected-session"
    assert history._session_override is None
