import io

from rich.console import Console

from jarv import settings_interactive
from jarv.config import DEFAULT_CONFIG
from jarv.tui_frame import panel_width


class FakeCatalogRefresher:
    def request(self, *_args, **_kwargs):
        return 0

    def cancel_pending(self):
        pass

    def close(self):
        pass


def _make_app(monkeypatch, config):
    test_console = Console(file=io.StringIO(), force_terminal=False, color_system=None, width=120)
    monkeypatch.setattr(settings_interactive, "_ModelCatalogRefresher", FakeCatalogRefresher)
    monkeypatch.setattr(settings_interactive, "terminal_size", lambda *, console=None: (120, 24))

    app = settings_interactive.SettingsApp(config, render_console=test_console)

    # The loop is never started in these tests (on_key is driven directly), so
    # _running stays False; record stop() requests to detect a closed screen.
    app.stop_calls = []
    app.stop = lambda result=None: app.stop_calls.append(result)
    return app, test_console


def _render(app, console) -> str:
    output = console.file
    output.seek(0)
    output.truncate(0)
    console.print(app.render())
    return output.getvalue()


def _row_index(config, key):
    return next(
        idx
        for idx, row in enumerate(settings_interactive._settings_rows(config))
        if row["key"] == key
    )


def test_settings_panel_spans_full_terminal_width():
    # /settings shares the flush-to-edge panel width with the other fullscreen views.
    assert panel_width(120) == 120
    assert panel_width(2) == 2
    assert panel_width(1) == 1


def test_settings_esc_exits_from_main_screen(monkeypatch):
    app, _console = _make_app(monkeypatch, dict(DEFAULT_CONFIG))

    app.on_key("ESC", 1)
    assert app.stop_calls


def test_settings_esc_returns_from_unchanged_compact_editor(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    app, _console = _make_app(monkeypatch, config)

    app.on_key("DOWN", _row_index(config, "base_url"))
    app.on_key("ENTER", 1)  # open the editor
    assert app.edit is not None

    app.on_key("ESC", 1)  # unchanged -> back to the list
    assert app.edit is None
    assert not app.stop_calls

    app.on_key("ESC", 1)  # exit the screen
    assert app.stop_calls


def test_settings_dirty_compact_editor_requires_second_esc(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    app, console = _make_app(monkeypatch, config)

    app.on_key("DOWN", _row_index(config, "base_url"))
    app.on_key("ENTER", 1)
    app.on_key("x", 1)  # dirty the buffer
    app.on_key("ESC", 1)  # arms discard rather than leaving

    assert app.edit is not None
    assert "Esc again to discard" in _render(app, console)

    app.on_key("ESC", 1)  # confirm discard -> back to the list
    assert app.edit is None
    assert not app.stop_calls


def test_settings_dirty_multiline_editor_requires_second_esc(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    app, console = _make_app(monkeypatch, config)

    app.on_key("DOWN", _row_index(config, "system_prompt"))
    app.on_key("ENTER", 1)
    app.on_key("x", 1)
    app.on_key("ESC", 1)

    assert app.edit is not None
    assert "Esc again to discard" in _render(app, console)


def test_settings_esc_returns_from_readonly_api_key_editor(monkeypatch):
    config = {**DEFAULT_CONFIG, "provider": "ollama"}
    app, _console = _make_app(monkeypatch, config)

    app.on_key("DOWN", _row_index(config, "api_key"))
    app.on_key("ENTER", 1)
    assert app.edit is not None

    app.on_key("ESC", 1)  # nothing to save -> back to the list
    assert app.edit is None
    assert not app.stop_calls

    app.on_key("ESC", 1)
    assert app.stop_calls


def test_settings_esc_returns_from_reset_confirmation(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    app, _console = _make_app(monkeypatch, config)

    app.on_key("DOWN", _row_index(config, "base_url"))
    app.on_key("r", 1)  # arm the reset confirmation
    assert app.pending_reset is not None

    app.on_key("ESC", 1)  # cancel the reset -> back to the list
    assert app.pending_reset is None
    assert not app.stop_calls

    app.on_key("ESC", 1)
    assert app.stop_calls
