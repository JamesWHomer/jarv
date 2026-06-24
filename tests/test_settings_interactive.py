import io
from collections import deque
from contextlib import contextmanager

from rich.console import Console

from jarv import settings_interactive
from jarv.config import DEFAULT_CONFIG
from jarv.tui_frame import panel_width


@contextmanager
def noop_context(*_args, **_kwargs):
    yield


class FakeCatalogRefresher:
    def request(self, *_args, **_kwargs):
        return 0

    def cancel_pending(self):
        pass

    def close(self):
        pass


class FakeLive:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.snapshots = []
        FakeLive.instances.append(self)

    def __enter__(self):
        self.refresh()
        return self

    def __exit__(self, *_exc):
        return False

    def refresh(self):
        get_renderable = self.kwargs["get_renderable"]
        self.renderable = get_renderable()
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, color_system=None, width=120)
        console.print(self.renderable)
        self.snapshots.append(output.getvalue())


def _row_index(config, key):
    return next(
        idx
        for idx, row in enumerate(settings_interactive._settings_rows(config))
        if row["key"] == key
    )


def _run_settings_with_keys(monkeypatch, config, keys):
    FakeLive.instances = []
    queued = deque(keys)
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=False, color_system=None, width=120)

    monkeypatch.setattr(settings_interactive, "console", test_console)
    monkeypatch.setattr(settings_interactive, "terminal_size", lambda *, console: (120, 24))
    monkeypatch.setattr(settings_interactive, "Live", FakeLive)
    monkeypatch.setattr(settings_interactive, "refresh_on_resize", noop_context)
    monkeypatch.setattr(settings_interactive, "mouse_capture", noop_context)
    monkeypatch.setattr(settings_interactive, "_ModelCatalogRefresher", FakeCatalogRefresher)

    def read_key_with_repeats(**_kwargs):
        if not queued:
            raise AssertionError("settings loop requested an extra key")
        key = queued.popleft()
        if isinstance(key, tuple):
            return key
        return key, 1

    monkeypatch.setattr(settings_interactive, "_read_key_with_repeats", read_key_with_repeats)

    settings_interactive.run_settings_interactive(config)
    assert not queued
    return FakeLive.instances[-1]


def test_settings_panel_spans_full_terminal_width():
    # /settings shares the flush-to-edge panel width with the other fullscreen views.
    assert panel_width(120) == 120
    assert panel_width(2) == 2
    assert panel_width(1) == 1


def test_settings_esc_exits_from_main_screen(monkeypatch):
    _run_settings_with_keys(monkeypatch, dict(DEFAULT_CONFIG), ["ESC"])


def test_settings_esc_returns_from_unchanged_compact_editor(monkeypatch):
    config = dict(DEFAULT_CONFIG)

    _run_settings_with_keys(
        monkeypatch,
        config,
        [("DOWN", _row_index(config, "base_url")), "ENTER", "ESC", "ESC"],
    )


def test_settings_dirty_compact_editor_requires_second_esc(monkeypatch):
    config = dict(DEFAULT_CONFIG)

    live = _run_settings_with_keys(
        monkeypatch,
        config,
        [("DOWN", _row_index(config, "base_url")), "ENTER", "x", "ESC", "ESC", "ESC"],
    )

    assert any("Esc again to discard" in snapshot for snapshot in live.snapshots)


def test_settings_dirty_multiline_editor_requires_second_esc(monkeypatch):
    config = dict(DEFAULT_CONFIG)

    live = _run_settings_with_keys(
        monkeypatch,
        config,
        [("DOWN", _row_index(config, "system_prompt")), "ENTER", "x", "ESC", "ESC", "ESC"],
    )

    assert any("Esc again to discard" in snapshot for snapshot in live.snapshots)


def test_settings_esc_returns_from_readonly_api_key_editor(monkeypatch):
    config = {**DEFAULT_CONFIG, "provider": "ollama"}

    _run_settings_with_keys(
        monkeypatch,
        config,
        [("DOWN", _row_index(config, "api_key")), "ENTER", "ESC", "ESC"],
    )


def test_settings_esc_returns_from_reset_confirmation(monkeypatch):
    config = dict(DEFAULT_CONFIG)

    _run_settings_with_keys(
        monkeypatch,
        config,
        [("DOWN", _row_index(config, "base_url")), "r", "ESC", "ESC"],
    )
