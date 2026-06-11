import io
import json
import subprocess
from contextlib import nullcontext

from rich.console import Console

from jarv import commands, update_check


def test_fetch_latest_release_returns_exact_version_spec(monkeypatch):
    payload = {
        "info": {"version": "0.15.1"},
        "urls": [
            {"packagetype": "sdist", "url": "https://example.test/jarv.tar.gz"},
            {"packagetype": "bdist_wheel", "url": "https://example.test/jarv.whl"},
        ],
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(update_check.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())

    assert update_check._fetch_latest_pypi_release() == (
        "0.15.1",
        "jarv==0.15.1",
    )


def test_version_comparison_does_not_treat_older_release_as_update():
    assert update_check._is_newer_version("0.15.1", "0.15.0")
    assert not update_check._is_newer_version("0.15.0", "0.15.1")
    assert not update_check._is_newer_version("not-a-version", "0.15.1")


def test_failed_background_check_is_not_throttled(monkeypatch):
    recorded = []

    monkeypatch.setattr(update_check, "_should_check_now", lambda: True)
    monkeypatch.setattr(update_check, "_fetch_latest_pypi_version", lambda: None)
    monkeypatch.setattr(update_check, "_record_check_time", lambda: recorded.append(True))

    update_check._check_update_background()

    assert recorded == []


def test_pipx_detection_does_not_resolve_interpreter_symlink(monkeypatch):
    monkeypatch.setattr(
        commands.sys,
        "executable",
        "/home/user/.local/share/pipx/venvs/jarv/bin/python",
    )

    assert commands._is_pipx_env()


def test_uv_tool_detection(monkeypatch):
    monkeypatch.setattr(
        commands.sys,
        "executable",
        "C:/Users/test/AppData/Roaming/uv/tools/jarv/Scripts/python.exe",
    )

    assert commands._is_uv_tool_env()
    assert commands._installation_manager() == "uv"


def test_update_installs_exact_artifact_and_verifies_version(monkeypatch, tmp_path):
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)
    package_spec = "jarv==0.15.1"
    calls = []

    monkeypatch.setattr(commands, "console", test_console)
    monkeypatch.setattr(test_console, "status", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(commands, "_fetch_latest_pypi_release", lambda: ("0.15.1", package_spec))
    monkeypatch.setattr(commands, "__version__", "0.15.0")
    monkeypatch.setattr(commands, "UPDATE_FLAG_FILE", tmp_path / "update_available.txt")
    monkeypatch.setattr(commands, "_installed_version", lambda: "0.15.1")
    monkeypatch.setattr(commands, "_is_editable_install", lambda: False)
    monkeypatch.setattr(commands, "_installation_manager", lambda: "pip")

    def run_install(manager, spec):
        calls.append((manager, spec))
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(commands, "_run_update_install", run_install)

    assert commands.cmd_update() == 0

    assert calls == [("pip", package_spec)]
    assert "Updated successfully" in output.getvalue()


def test_update_does_not_report_success_after_noop(monkeypatch):
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)

    monkeypatch.setattr(commands, "console", test_console)
    monkeypatch.setattr(test_console, "status", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(
        commands,
        "_fetch_latest_pypi_release",
        lambda: ("0.15.1", "jarv==0.15.1"),
    )
    monkeypatch.setattr(commands, "__version__", "0.15.0")
    monkeypatch.setattr(commands, "_is_editable_install", lambda: False)
    monkeypatch.setattr(commands, "_installation_manager", lambda: "pip")
    monkeypatch.setattr(
        commands,
        "_run_update_install",
        lambda _manager, _spec: subprocess.CompletedProcess([], 0, "Requirement already satisfied", ""),
    )
    monkeypatch.setattr(commands, "_installed_version", lambda: "0.15.0")

    assert commands.cmd_update() == 1

    text = output.getvalue()
    assert "Update failed" in text
    assert "Expected jarv 0.15.1, but this process still sees 0.15.0." in text
    assert "Updated successfully" not in text


def test_update_verifies_pipx_fallback(monkeypatch):
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)
    calls = []

    monkeypatch.setattr(commands, "console", test_console)
    monkeypatch.setattr(test_console, "status", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(
        commands,
        "_fetch_latest_pypi_release",
        lambda: ("0.15.1", "jarv==0.15.1"),
    )
    monkeypatch.setattr(commands, "__version__", "0.15.0")
    monkeypatch.setattr(commands, "_is_editable_install", lambda: False)
    monkeypatch.setattr(commands, "_installation_manager", lambda: "pip")
    monkeypatch.setattr(commands, "_fallback_tool_manager", lambda: "pipx")
    monkeypatch.setattr(commands, "_installed_version", lambda: "0.15.1")

    def run_install(manager, spec):
        calls.append((manager, spec))
        if manager == "pip":
            return subprocess.CompletedProcess([], 1, "", "externally-managed-environment")
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(commands, "_run_update_install", run_install)

    assert commands.cmd_update() == 0

    assert calls == [("pip", "jarv==0.15.1"), ("pipx", "jarv==0.15.1")]
    assert "Updated successfully" in output.getvalue()


def test_update_skips_editable_install(monkeypatch):
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)

    monkeypatch.setattr(commands, "console", test_console)
    monkeypatch.setattr(commands, "_fetch_latest_pypi_release", lambda: ("0.15.1", "jarv==0.15.1"))
    monkeypatch.setattr(commands, "__version__", "0.15.0")
    monkeypatch.setattr(commands, "_is_editable_install", lambda: True)

    assert commands.cmd_update() == 1
    assert "Editable install detected" in output.getvalue()


def test_update_command_timeout_becomes_failure(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["pip"], commands.UPDATE_INSTALL_TIMEOUT_SECONDS)

    monkeypatch.setattr(commands.subprocess, "run", timeout)

    result = commands._run_update_command(["pip"])

    assert result.returncode == 124
    assert "timed out" in result.stderr
