"""Tests for the named terminal-capability quirks (jarv.tui_capabilities)."""

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


def test_supports_erase_eol_default():
    assert tui_capabilities.supports_erase_eol() is True
