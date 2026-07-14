import io
import json
import os
import sys
from pathlib import Path

import pytest
from rich.console import Console

from jarv import cli, commands, standalone, uninstall


class TTY(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture
def captured_console(monkeypatch):
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)
    monkeypatch.setattr(uninstall, "console", test_console)
    return test_console, output


@pytest.mark.parametrize(
    ("executable", "kind", "command"),
    [
        (
            "C:/Users/test/AppData/Local/Microsoft/WinGet/Links/jarv.exe",
            "winget",
            "winget uninstall JamesWHomer.Jarv",
        ),
        ("C:/Users/test/scoop/apps/jarv/current/jarv.exe", "scoop", "scoop uninstall jarv"),
        ("/home/linuxbrew/.linuxbrew/Cellar/jarv/1.0/bin/jarv", "brew", "brew uninstall jarv"),
        ("C:/Users/test/AppData/Local/Programs/Jarv/jarv.exe", "standalone", None),
    ],
)
def test_detects_frozen_install_channels(monkeypatch, executable, kind, command):
    monkeypatch.setattr(standalone.sys, "frozen", True, raising=False)
    monkeypatch.setattr(uninstall.sys, "executable", executable)

    channel = uninstall.detect_install_channel()

    assert channel.kind == kind
    assert channel.manual_command == command


@pytest.mark.parametrize(
    ("editable", "pipx", "uv", "kind", "command"),
    [
        (True, False, False, "editable", None),
        (False, True, False, "pipx", "pipx uninstall jarv"),
        (False, False, True, "uv", "uv tool uninstall jarv"),
        (False, False, False, "pip", f"{sys.executable} -m pip uninstall jarv"),
    ],
)
def test_detects_python_install_channels(monkeypatch, editable, pipx, uv, kind, command):
    monkeypatch.delattr(standalone.sys, "frozen", raising=False)
    monkeypatch.setattr(commands, "_is_editable_install", lambda: editable)
    monkeypatch.setattr(commands, "_is_pipx_env", lambda: pipx)
    monkeypatch.setattr(commands, "_is_uv_tool_env", lambda: uv)

    channel = uninstall.detect_install_channel()

    assert channel.kind == kind
    assert channel.manual_command == command


def test_declining_standalone_uninstall_leaves_executable(monkeypatch, tmp_path, captured_console):
    executable = tmp_path / "jarv"
    executable.write_text("binary", encoding="utf-8")
    monkeypatch.setattr(
        uninstall,
        "detect_install_channel",
        lambda: uninstall.InstallChannel("standalone", executable),
    )
    monkeypatch.setattr(uninstall.sys, "platform", "linux")
    monkeypatch.setattr(uninstall.sys, "stdin", TTY())
    monkeypatch.setattr(captured_console[0], "input", lambda *_args, **_kwargs: "n")

    assert uninstall.cmd_uninstall([]) == 1
    assert executable.exists()
    assert "cancelled" in captured_console[1].getvalue().lower()


def test_confirmed_unix_standalone_uninstall_removes_executable(monkeypatch, tmp_path, captured_console):
    install_dir = tmp_path / "bin"
    install_dir.mkdir()
    executable = install_dir / "jarv"
    executable.write_text("binary", encoding="utf-8")
    monkeypatch.setattr(
        uninstall,
        "detect_install_channel",
        lambda: uninstall.InstallChannel("standalone", executable),
    )
    monkeypatch.setattr(uninstall.sys, "platform", "linux")
    monkeypatch.setattr(uninstall.sys, "stdin", TTY())
    monkeypatch.setattr(captured_console[0], "input", lambda *_args, **_kwargs: "y")

    assert uninstall.cmd_uninstall([]) == 0
    assert not executable.exists()
    assert not install_dir.exists()


def test_non_tty_requires_yes_before_standalone_side_effects(monkeypatch, tmp_path, captured_console):
    executable = tmp_path / "jarv"
    executable.write_text("binary", encoding="utf-8")
    monkeypatch.setattr(
        uninstall,
        "detect_install_channel",
        lambda: uninstall.InstallChannel("standalone", executable),
    )
    monkeypatch.setattr(uninstall.sys, "platform", "linux")
    monkeypatch.setattr(uninstall.sys, "stdin", io.StringIO())

    assert uninstall.cmd_uninstall([]) == 1
    assert executable.exists()
    assert "--yes" in captured_console[1].getvalue()


def test_windows_standalone_stages_deferred_uninstall(monkeypatch, tmp_path, captured_console):
    executable = tmp_path / "Jarv" / "jarv.exe"
    executable.parent.mkdir()
    executable.write_text("binary", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        uninstall,
        "detect_install_channel",
        lambda: uninstall.InstallChannel("standalone", executable),
    )
    monkeypatch.setattr(uninstall.sys, "platform", "win32")
    monkeypatch.setattr(uninstall, "_windows_user_path_contains", lambda _path: True)
    monkeypatch.setattr(
        uninstall,
        "_stage_windows_uninstaller",
        lambda path, **kwargs: calls.append((path, kwargs)),
    )

    outcome = uninstall.run_uninstall(["--yes", "--purge"])

    assert outcome == uninstall.UninstallOutcome(0, destructive=True)
    assert calls == [(executable, {"remove_path_entry": True, "purge": True})]
    assert executable.exists()


@pytest.mark.parametrize("purge", [False, True])
def test_staged_windows_script_has_handoff_path_filter_and_optional_purge(
    monkeypatch,
    tmp_path,
    purge,
):
    install_dir = tmp_path / "installed"
    install_dir.mkdir()
    executable = install_dir / "jarv.exe"
    executable.write_text("binary", encoding="utf-8")
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    result_file = tmp_path / "config" / "uninstall-result.json"
    staged_processes = []
    monkeypatch.setattr(uninstall, "UNINSTALL_RESULT_FILE", result_file)
    monkeypatch.setattr(uninstall.tempfile, "mkdtemp", lambda **_kwargs: str(staging_root))
    monkeypatch.setattr(uninstall.shutil, "which", lambda _name: "powershell")
    monkeypatch.setattr(
        uninstall.subprocess,
        "Popen",
        lambda args, **kwargs: staged_processes.append((args, kwargs)),
    )

    uninstall._stage_windows_uninstaller(
        executable,
        remove_path_entry=True,
        purge=purge,
    )

    script = staging_root / "jarv-uninstall.ps1"
    text = script.read_text(encoding="utf-8")
    assert "Wait-Process -Id $ParentPid" in text
    assert "OrdinalIgnoreCase" in text
    assert "SetEnvironmentVariable" in text
    assert script.parent != install_dir
    assert (str(uninstall.CONFIG_DIR) in text) is purge
    assert (str(uninstall.clipboard_image_dir()) in text) is purge
    # PATH cleanup and purge only run once the executable is gone; otherwise a
    # failure breadcrumb is written for the next launch to report.
    assert str(result_file) in text
    assert '"status":"failed"' in text
    assert "} else {" in text
    assert text.index('"status":"failed"') < text.index("SetEnvironmentVariable")
    args, kwargs = staged_processes[0]
    assert "-NonInteractive" in args
    assert kwargs["stdin"] == uninstall.subprocess.DEVNULL
    assert kwargs["creationflags"] == standalone._windows_updater_creation_flags()


def test_staged_windows_uninstaller_clears_stale_breadcrumb(monkeypatch, tmp_path):
    executable = tmp_path / "Jarv" / "jarv.exe"
    executable.parent.mkdir()
    executable.write_text("binary", encoding="utf-8")
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    result_file = tmp_path / "uninstall-result.json"
    result_file.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(uninstall, "UNINSTALL_RESULT_FILE", result_file)
    monkeypatch.setattr(uninstall.tempfile, "mkdtemp", lambda **_kwargs: str(staging_root))
    monkeypatch.setattr(uninstall.shutil, "which", lambda _name: "powershell")
    monkeypatch.setattr(uninstall.subprocess, "Popen", lambda args, **kwargs: object())

    uninstall._stage_windows_uninstaller(executable, remove_path_entry=False, purge=False)

    assert not result_file.exists()


def test_staged_windows_uninstaller_retries_without_breakaway_when_job_rejects_it(
    monkeypatch,
    tmp_path,
):
    executable = tmp_path / "Jarv" / "jarv.exe"
    executable.parent.mkdir()
    executable.write_text("binary", encoding="utf-8")
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    creation_flags = []
    process = object()
    monkeypatch.setattr(uninstall, "UNINSTALL_RESULT_FILE", tmp_path / "uninstall-result.json")
    monkeypatch.setattr(uninstall.tempfile, "mkdtemp", lambda **_kwargs: str(staging_root))
    monkeypatch.setattr(uninstall.shutil, "which", lambda _name: "powershell")

    def popen(_args, **kwargs):
        creation_flags.append(kwargs["creationflags"])
        if len(creation_flags) == 1:
            error = OSError("access denied")
            error.winerror = 5
            raise error
        return process

    monkeypatch.setattr(uninstall.subprocess, "Popen", popen)

    staged = uninstall._stage_windows_uninstaller(
        executable,
        remove_path_entry=False,
        purge=False,
    )

    assert staged is process
    assert creation_flags == [
        standalone._windows_updater_creation_flags(),
        standalone._windows_updater_creation_flags(allow_breakaway=False),
    ]


def test_path_value_contains_expands_registry_style_entries(monkeypatch, tmp_path):
    directory = tmp_path / "Jarv"
    monkeypatch.setenv("JARV_TEST_ROOT", str(tmp_path))
    raw = ";".join(
        [
            str(tmp_path / "other"),
            f"%JARV_TEST_ROOT%{os.sep}Jarv{os.sep}",
        ]
    )

    assert uninstall._path_value_contains(raw, directory)
    assert uninstall._path_value_contains(str(directory), directory)
    assert not uninstall._path_value_contains(str(tmp_path / "other"), directory)
    assert not uninstall._path_value_contains("", directory)
    assert not uninstall._path_value_contains(None, directory)


@pytest.mark.parametrize(
    ("kind", "command"),
    [
        ("winget", "winget uninstall JamesWHomer.Jarv"),
        ("scoop", "scoop uninstall jarv"),
        ("brew", "brew uninstall jarv"),
        ("pipx", "pipx uninstall jarv"),
        ("uv", "uv tool uninstall jarv"),
        ("pip", "python -m pip uninstall jarv"),
    ],
)
def test_manager_channels_print_exact_command(monkeypatch, captured_console, kind, command):
    monkeypatch.setattr(
        uninstall,
        "detect_install_channel",
        lambda: uninstall.InstallChannel(kind, Path("jarv"), command),
    )

    assert uninstall.cmd_uninstall([]) == 0
    assert command in captured_console[1].getvalue()


def test_manager_purge_removes_data_but_default_keeps_it(monkeypatch, tmp_path, captured_console):
    config_dir = tmp_path / "config"
    clipboard_dir = tmp_path / "clipboard"
    config_dir.mkdir()
    clipboard_dir.mkdir()
    (config_dir / "config.json").write_text("{}", encoding="utf-8")
    (clipboard_dir / "image.png").write_bytes(b"image")
    monkeypatch.setattr(uninstall, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(uninstall, "clipboard_image_dir", lambda: clipboard_dir)
    monkeypatch.setattr(
        uninstall,
        "detect_install_channel",
        lambda: uninstall.InstallChannel("pipx", Path("python"), "pipx uninstall jarv"),
    )

    outcome = uninstall.run_uninstall([])
    assert outcome == uninstall.UninstallOutcome(0, destructive=False)
    assert config_dir.exists()
    assert clipboard_dir.exists()

    outcome = uninstall.run_uninstall(["--purge", "--yes"])
    assert outcome == uninstall.UninstallOutcome(0, destructive=True)
    assert not config_dir.exists()
    assert not clipboard_dir.exists()


def test_unknown_argument_is_usage_error(monkeypatch, captured_console):
    monkeypatch.setattr(
        uninstall,
        "detect_install_channel",
        lambda: pytest.fail("detection should not run for invalid arguments"),
    )

    assert uninstall.cmd_uninstall(["--nope"]) == 2
    assert "Usage:" in captured_console[1].getvalue()


def test_previous_failed_uninstall_is_reported_and_consumed(monkeypatch, tmp_path):
    from jarv import paths

    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)
    result_file = tmp_path / "uninstall-result.json"
    result_file.write_text(
        json.dumps({"status": "failed", "message": "could not remove jarv.exe"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_console", lambda: test_console)
    monkeypatch.setattr(paths, "UNINSTALL_RESULT_FILE", result_file)

    cli._print_previous_uninstall_result()

    text = output.getvalue()
    assert "could not remove jarv.exe" in text
    assert "jarv /uninstall" in text
    assert not result_file.exists()

    cli._print_previous_uninstall_result()

    assert output.getvalue() == text


def test_cli_forwards_uninstall_flags_and_propagates_status(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["jarv", "/uninstall", "--purge", "--yes"])
    monkeypatch.setattr(
        uninstall,
        "cmd_uninstall",
        lambda args: calls.append(args) or 1,
    )

    with pytest.raises(SystemExit) as raised:
        cli.main()

    assert raised.value.code == 1
    assert calls == [["--purge", "--yes"]]
