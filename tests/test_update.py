import io
import json
import subprocess
from contextlib import nullcontext

from rich.console import Console

from jarv import commands, update_check


def test_fetch_latest_release_prefers_wheel(monkeypatch):
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
        "https://example.test/jarv.whl",
    )


def test_pipx_detection_does_not_resolve_interpreter_symlink(monkeypatch):
    monkeypatch.setattr(
        commands.sys,
        "executable",
        "/home/user/.local/share/pipx/venvs/jarv/bin/python",
    )

    assert commands._is_pipx_env()


def test_update_installs_exact_artifact_and_verifies_version(monkeypatch, tmp_path):
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)
    artifact_url = "https://example.test/jarv-0.15.1-py3-none-any.whl"
    calls = []

    monkeypatch.setattr(commands, "console", test_console)
    monkeypatch.setattr(test_console, "status", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(commands, "_fetch_latest_pypi_release", lambda: ("0.15.1", artifact_url))
    monkeypatch.setattr(commands, "__version__", "0.15.0")
    monkeypatch.setattr(commands, "UPDATE_FLAG_FILE", tmp_path / "update_available.txt")
    monkeypatch.setattr(commands, "_installed_version", lambda: "0.15.1")

    def run_pip(package_spec):
        calls.append(package_spec)
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(commands, "_run_pip_upgrade", run_pip)

    commands.cmd_update()

    assert calls == [artifact_url]
    assert "Updated successfully" in output.getvalue()


def test_update_does_not_report_success_after_noop(monkeypatch):
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)

    monkeypatch.setattr(commands, "console", test_console)
    monkeypatch.setattr(test_console, "status", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(
        commands,
        "_fetch_latest_pypi_release",
        lambda: ("0.15.1", "https://example.test/jarv.whl"),
    )
    monkeypatch.setattr(commands, "__version__", "0.15.0")
    monkeypatch.setattr(
        commands,
        "_run_pip_upgrade",
        lambda _spec: subprocess.CompletedProcess([], 0, "Requirement already satisfied", ""),
    )
    monkeypatch.setattr(commands, "_installed_version", lambda: "0.15.0")

    commands.cmd_update()

    text = output.getvalue()
    assert "Update failed" in text
    assert "Expected jarv 0.15.1, but the active installation is 0.15.0." in text
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
        lambda: ("0.15.1", "https://example.test/jarv.whl"),
    )
    monkeypatch.setattr(commands, "__version__", "0.15.0")
    monkeypatch.setattr(commands, "_is_pipx_env", lambda: False)
    monkeypatch.setattr(
        commands,
        "_run_pip_upgrade",
        lambda _spec: subprocess.CompletedProcess([], 1, "", "externally-managed-environment"),
    )
    monkeypatch.setattr(commands, "_pipx_installed_version", lambda: "0.15.1")

    def run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(commands.subprocess, "run", run)

    commands.cmd_update()

    assert ["pipx", "install", "--force", "https://example.test/jarv.whl"] in calls
    assert "Updated successfully" in output.getvalue()
