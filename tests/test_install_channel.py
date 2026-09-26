from pathlib import Path

import pytest

from jarv import commands, install_channel, standalone, uninstall


@pytest.mark.parametrize(
    ("resolved", "kind"),
    [
        ("C:/Users/test/SCOOP/apps/jarv/1.0/jarv.exe", "scoop"),
        ("C:/Users/test/Microsoft/WinGet/Packages/Jarv/jarv.exe", "winget"),
        ("/opt/homebrew/Cellar/jarv/1.0/bin/jarv", "brew"),
    ],
)
def test_resolved_binary_ownership_is_shared(monkeypatch, tmp_path, resolved, kind):
    executable = tmp_path / "bin" / "jarv"
    monkeypatch.setattr(standalone, "is_standalone_install", lambda: True)
    monkeypatch.setattr(install_channel.sys, "executable", str(executable))
    # Model a launcher symlink without requiring Windows symlink privileges.
    monkeypatch.setattr(Path, "resolve", lambda self: Path(resolved))

    channel = install_channel.detect_install_channel()

    assert channel.kind == kind
    assert channel.executable == executable
    assert uninstall.detect_install_channel() == channel
    outcome = commands.perform_update(lambda _stage: pytest.fail("Must hand off to the manager"))
    assert outcome.kind == "manual"
    assert channel.update_command in outcome.detail


def test_direct_binary_stays_standalone_when_resolution_fails(monkeypatch, tmp_path):
    executable = tmp_path / "jarv"
    monkeypatch.setattr(standalone, "is_standalone_install", lambda: True)
    monkeypatch.setattr(install_channel.sys, "executable", str(executable))

    def inaccessible(self):
        raise OSError("Cannot resolve executable")

    monkeypatch.setattr(Path, "resolve", inaccessible)

    assert install_channel.detect_install_channel() == install_channel.InstallChannel(
        "standalone", executable,
    )
