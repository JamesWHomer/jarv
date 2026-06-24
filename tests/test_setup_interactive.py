import io

from rich.console import Console

from jarv import setup_interactive, settings_command
from jarv.config import DEFAULT_CONFIG


class FakeCatalogRefresher:
    def request(self, *_args, **_kwargs):
        return 0

    def cancel_pending(self):
        pass

    def close(self):
        pass


def _make_app(monkeypatch, config, *, step=None):
    saved = []
    test_console = Console(file=io.StringIO(), force_terminal=False, color_system=None, width=120)
    monkeypatch.setattr(setup_interactive, "_ModelCatalogRefresher", FakeCatalogRefresher)
    monkeypatch.setattr(setup_interactive, "terminal_size", lambda *, console=None: (120, 30))
    monkeypatch.setattr(setup_interactive, "save_config", lambda cfg: None)
    monkeypatch.setattr(setup_interactive, "probe_connection", lambda cfg: (True, None))
    monkeypatch.setattr(settings_command, "save_config", lambda cfg: saved.append(dict(cfg)))

    app = setup_interactive.SetupApp(config, step=step, render_console=test_console)

    # The loop is never started in these tests (on_key is driven directly), so
    # _running stays False; record stop() requests to detect a finished wizard.
    app.stop_calls = []
    app.stop = lambda result=None: app.stop_calls.append(result)
    return app, saved, test_console


def _render(app, console) -> str:
    output = console.file
    output.seek(0)
    output.truncate(0)
    console.print(app.render())
    return output.getvalue()


def test_setup_step_rows_cloud_provider():
    rows = setup_interactive._setup_step_rows({**DEFAULT_CONFIG, "provider": "openai"})
    assert [row["key"] for row in rows] == ["provider", "api_key", "model"]


def test_setup_step_rows_local_provider():
    rows = setup_interactive._setup_step_rows({**DEFAULT_CONFIG, "provider": "ollama"})
    assert [row["key"] for row in rows] == ["provider", "model", "base_url"]


def test_welcome_screen_shows_brand_and_hint(monkeypatch):
    app, _saved, console = _make_app(monkeypatch, dict(DEFAULT_CONFIG))

    assert app.phase == "welcome"
    # Advance the brand clock past the hint's entrance window so it's drawn.
    app._brand_started_at -= 5.0
    rendered = _render(app, console)
    assert "jarv ▸ setup" in rendered
    assert "begin" in rendered  # the welcome hint


def test_full_wizard_walks_steps_and_reaches_ready(monkeypatch):
    app, saved, console = _make_app(monkeypatch, dict(DEFAULT_CONFIG))

    # Welcome -> first step.
    app.on_key("ENTER", 1)
    assert app.phase == "step"
    assert "Step 1/3" in _render(app, console)

    # Provider committed -> api_key step shows the "where to get a key" hint.
    app.on_key("ENTER", 1)
    api_key_screen = _render(app, console)
    assert "Step 2/3" in api_key_screen
    assert "https://platform.openai.com/api-keys" in api_key_screen

    # api_key (unchanged) -> model step.
    app.on_key("ENTER", 1)
    assert "Step 3/3" in _render(app, console)

    # Model committed -> ready screen.
    app.on_key("ENTER", 1)
    assert app.phase == "ready"
    assert app.result is not None
    assert app.result["provider"] == "openai"
    assert isinstance(app.result["model"], str) and app.result["model"]
    # provider + model both persisted (api_key left unchanged saves nothing).
    assert len(saved) >= 2

    ready_screen = _render(app, console)
    assert "You're all set!" in ready_screen
    assert "jarv /help" in ready_screen
    assert "Type jarv to start chatting" in ready_screen


def test_ready_enter_finishes_with_config(monkeypatch):
    app, _saved, _console = _make_app(monkeypatch, dict(DEFAULT_CONFIG))
    for _ in range(4):  # welcome + provider + api_key + model
        app.on_key("ENTER", 1)
    assert app.phase == "ready"

    app.on_key("ENTER", 1)
    assert app.stop_calls  # the wizard finished
    assert app.result is not None
    assert app.result["provider"] == "openai"


def test_left_right_navigate_between_steps(monkeypatch):
    app, _saved, console = _make_app(monkeypatch, dict(DEFAULT_CONFIG))
    app.on_key("ENTER", 1)  # welcome -> provider
    assert app.step_idx == 0

    app.on_key("RIGHT", 1)  # commit provider, advance to api_key
    assert app.step_idx == 1
    assert "Step 2/3" in _render(app, console)

    app.on_key("LEFT", 1)  # back to provider
    assert app.step_idx == 0
    assert "Step 1/3" in _render(app, console)

    app.on_key("LEFT", 1)  # no-op on the first step
    assert app.step_idx == 0


def test_esc_backs_out_one_step_then_cancels(monkeypatch):
    app, _saved, console = _make_app(monkeypatch, dict(DEFAULT_CONFIG))
    app.on_key("ENTER", 1)  # welcome -> provider
    app.on_key("ENTER", 1)  # commit provider -> api_key
    assert app.step_idx == 1

    app.on_key("ESC", 1)  # back to provider
    assert app.step_idx == 0
    assert not app.stop_calls

    app.on_key("ESC", 1)  # cancel the whole wizard
    assert app.stop_calls
    assert app.result is None


def test_welcome_esc_cancels(monkeypatch):
    app, _saved, _console = _make_app(monkeypatch, dict(DEFAULT_CONFIG))
    app.on_key("ESC", 1)
    assert app.stop_calls
    assert app.result is None


def test_single_step_saves_and_exits(monkeypatch):
    app, saved, console = _make_app(monkeypatch, dict(DEFAULT_CONFIG), step="provider")

    assert app.phase == "step"
    assert app.single_step is True
    assert "Step" not in _render(app, console)  # single step drops the N/M indicator

    app.on_key("ENTER", 1)
    assert app.stop_calls
    assert app.result is not None
    assert app.result["provider"] == "openai"
    assert saved


def test_single_step_esc_cancels(monkeypatch):
    app, saved, _console = _make_app(monkeypatch, dict(DEFAULT_CONFIG), step="provider")

    app.on_key("ESC", 1)
    assert app.stop_calls
    assert app.result is None
    assert not saved


def test_run_setup_interactive_rejects_unknown_step(monkeypatch):
    test_console = Console(file=io.StringIO(), force_terminal=False, color_system=None, width=120)
    monkeypatch.setattr(setup_interactive, "console", test_console)

    result = setup_interactive.run_setup_interactive(dict(DEFAULT_CONFIG), step="bogus")
    assert result is None
    assert "Unknown setup step" in test_console.file.getvalue()


def test_api_key_submenu_warns_on_bad_format_then_saves(monkeypatch):
    saved = []
    monkeypatch.setattr(settings_command, "save_config", lambda cfg: saved.append(dict(cfg)))

    config = {**DEFAULT_CONFIG, "provider": "openai", "api_keys": {}}
    row = next(
        r for r in settings_command._settings_rows(config) if r["key"] == "api_key"
    )
    edit = settings_command._settings_begin_edit(row, config)
    edit["buffer"] = "not-a-real-key"

    _cfg, _msg, style, done = settings_command._settings_commit_edit(edit, config)
    assert done is False
    assert style == "yellow"
    assert edit.get("format_warned")
    assert not saved

    cfg, _msg, _style, done = settings_command._settings_commit_edit(edit, config)
    assert done is True
    assert cfg["api_keys"]["openai"] == "not-a-real-key"
    assert saved
