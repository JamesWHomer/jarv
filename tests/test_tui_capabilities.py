"""Tests for the named terminal-capability quirks (jarv.tui_capabilities)."""

import pytest

from jarv import tui_capabilities


def test_is_wsl_detects_env_markers(monkeypatch):
    tui_capabilities.is_wsl.cache_clear()
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    try:
        assert tui_capabilities.is_wsl() is True
    finally:
        tui_capabilities.is_wsl.cache_clear()


def test_is_wsl_detects_interop_marker(monkeypatch):
    tui_capabilities.is_wsl.cache_clear()
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setenv("WSL_INTEROP", "/run/WSL/123_interop")
    try:
        assert tui_capabilities.is_wsl() is True
    finally:
        tui_capabilities.is_wsl.cache_clear()


def test_is_wsl_false_without_markers(monkeypatch):
    from types import SimpleNamespace

    tui_capabilities.is_wsl.cache_clear()
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.setattr(
        tui_capabilities.platform,
        "uname",
        lambda: SimpleNamespace(release="6.1.0-generic"),
    )
    try:
        assert tui_capabilities.is_wsl() is False
    finally:
        tui_capabilities.is_wsl.cache_clear()


@pytest.mark.parametrize("setting, expected", [(None, True), ("0", True), ("1", False)])
def test_supports_erase_eol_setting(monkeypatch, setting, expected):
    if setting is None:
        monkeypatch.delenv("JARV_NO_ERASE_EOL", raising=False)
    else:
        monkeypatch.setenv("JARV_NO_ERASE_EOL", setting)
    assert tui_capabilities.supports_erase_eol() is expected
