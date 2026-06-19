import io
import json
import subprocess
import zipfile
from contextlib import nullcontext
from pathlib import Path

from rich.console import Console

from jarv import commands, standalone, update_check


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


def test_standalone_install_detection(monkeypatch):
    monkeypatch.setattr(standalone.sys, "frozen", True, raising=False)

    assert standalone.is_standalone_install()


def test_standalone_manifest_selects_matching_platform_asset():
    manifest = {
        "version": "0.15.1",
        "assets": [
            {
                "platform": "linux",
                "architecture": "aarch64",
                "name": "jarv-0.15.1-linux-aarch64.tar.gz",
                "download_url": "https://example.test/linux-arm64",
                "sha256": "abc",
                "size": 123,
            },
            {
                "platform": "windows",
                "architecture": "arm64",
                "name": "jarv-0.15.1-windows-arm64.zip",
                "download_url": "https://example.test/windows-arm64",
                "sha256": "def",
                "size": 456,
            },
        ],
    }

    asset = standalone.select_release_asset(
        manifest,
        target_platform="linux",
        target_architecture="aarch64",
    )

    assert asset is not None
    assert asset.name == "jarv-0.15.1-linux-aarch64.tar.gz"
    assert asset.version == "0.15.1"


def test_standalone_download_rejects_checksum_mismatch(monkeypatch, tmp_path):
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(
        standalone.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(b"not the expected bytes"),
    )
    asset = standalone.ReleaseAsset(
        version="0.15.1",
        platform="linux",
        architecture="x86_64",
        name="jarv-0.15.1-linux-x86_64.tar.gz",
        download_url="https://example.test/jarv.tar.gz",
        sha256="0" * 64,
    )

    try:
        standalone.download_asset(asset, tmp_path / asset.name)
    except ValueError as exc:
        assert "Checksum mismatch" in str(exc)
    else:
        raise AssertionError("checksum mismatch did not fail")


def test_standalone_update_already_current(monkeypatch):
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)

    monkeypatch.setattr(commands, "console", test_console)
    monkeypatch.setattr(commands, "__version__", "0.15.1")
    monkeypatch.setattr(standalone, "fetch_release_manifest", lambda: {"version": "0.15.1", "assets": []})

    assert commands._cmd_update_standalone() == 0
    assert "Already up to date" in output.getvalue()


def test_standalone_update_fails_when_no_matching_asset(monkeypatch):
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)
    manifest = {
        "version": "0.15.1",
        "assets": [
            {
                "platform": "macos",
                "architecture": "arm64",
                "name": "jarv-0.15.1-macos-arm64.tar.gz",
                "download_url": "https://example.test/macos-arm64",
                "sha256": "abc",
            }
        ],
    }

    monkeypatch.setattr(commands, "console", test_console)
    monkeypatch.setattr(commands, "__version__", "0.15.0")
    monkeypatch.setattr(standalone, "fetch_release_manifest", lambda: manifest)
    monkeypatch.setattr(standalone, "normalize_platform", lambda _value=None: "linux")
    monkeypatch.setattr(
        standalone,
        "normalize_architecture",
        lambda _value=None, **_kwargs: "x86_64",
    )

    assert commands._cmd_update_standalone() == 1
    assert "No standalone release asset matches this system" in output.getvalue()


def test_standalone_update_success(monkeypatch, tmp_path):
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None)
    asset = {
        "platform": "linux",
        "architecture": "x86_64",
        "name": "jarv-0.15.1-linux-x86_64.tar.gz",
        "download_url": "https://example.test/linux-x64",
        "sha256": "abc",
    }
    calls = []

    monkeypatch.setattr(commands, "console", test_console)
    monkeypatch.setattr(test_console, "status", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(commands, "__version__", "0.15.0")
    monkeypatch.setattr(commands, "UPDATE_FLAG_FILE", tmp_path / "update_available.txt")
    monkeypatch.setattr(standalone, "fetch_release_manifest", lambda: {"version": "0.15.1", "assets": [asset]})
    monkeypatch.setattr(standalone, "normalize_platform", lambda _value=None: "linux")
    monkeypatch.setattr(
        standalone,
        "normalize_architecture",
        lambda _value=None, **_kwargs: "x86_64",
    )

    def install(release_asset):
        calls.append(release_asset.name)
        return "installed"

    monkeypatch.setattr(standalone, "install_standalone_asset", install)

    assert commands._cmd_update_standalone() == 0
    assert calls == ["jarv-0.15.1-linux-x86_64.tar.gz"]
    assert "Updated successfully" in output.getvalue()


def test_windows_standalone_update_stages_handoff(monkeypatch, tmp_path):
    target = tmp_path / "jarv.exe"
    target.write_bytes(b"old")
    staged = []
    asset = standalone.ReleaseAsset(
        version="0.15.1",
        platform="windows",
        architecture="arm64",
        name="jarv-0.15.1-windows-arm64.zip",
        download_url="https://example.test/jarv.zip",
        sha256="ignored",
    )

    def fake_download(_asset, destination: Path):
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("jarv.exe", b"new")
        return destination

    monkeypatch.setattr(standalone, "download_asset", fake_download)
    monkeypatch.setattr(
        standalone,
        "_stage_windows_updater",
        lambda source, handoff_target: staged.append((source, handoff_target)),
    )

    assert standalone.install_standalone_asset(asset, executable_path=target, windows=True) == "staged"
    assert staged
    assert staged[0][1] == target.resolve()
