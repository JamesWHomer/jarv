import io
import re

from rich.console import Console, Group
from rich.text import Text

from jarv.session_browser_live import SessionBrowserLive


def test_refresh_updates_only_changed_rows_and_resize_repaints_every_row(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, force_interactive=True,
                      legacy_windows=False, color_system="truecolor", width=40, height=6)
    rows = [Text("Sessions"), Text("First conversation", style="on blue"), Text("Second conversation"), Text("Footer")]
    live = SessionBrowserLive(get_renderable=lambda: Group(*rows), console=console,
                              screen=True, auto_refresh=False, redirect_stdout=False, redirect_stderr=False)

    def take_output():
        text = output.getvalue()
        output.seek(0)
        output.truncate(0)
        return text

    try:
        live.start(refresh=True)
        initial = take_output()
        assert re.findall(r"\x1b\[(\d+);1H", initial) == [str(i) for i in range(1, 7)]
        assert all(row.plain in initial for row in rows)

        rows[1] = Text("First conversation")
        rows[2] = Text("Second conversation", style="on blue")
        live.refresh()
        moved = take_output()
        assert re.findall(r"\x1b\[(\d+);1H", moved) == ["2", "3"]
        assert "Sessions" not in moved and "Footer" not in moved
        assert "\n" not in moved  # No bottom-row newline can scroll the screen.

        live.refresh()
        assert take_output() == ""

        rows[2] = Text("Short")
        live.refresh()
        assert "Short" + " " * 35 in take_output()  # Erases the old longer row.

        console.width, console.height = 25, 5
        live.refresh()
        resized = take_output()
        assert re.findall(r"\x1b\[(\d+);1H", resized) == [str(i) for i in range(1, 6)]
        assert all(row.plain in resized for row in rows)
    finally:
        live.stop()
    assert "\x1b[?1049l" in take_output()


def test_style_changes_and_wide_text_survive_incremental_refresh(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, force_interactive=True,
                      legacy_windows=False, color_system="truecolor", no_color=False, width=20, height=4)
    row = Text("会話 e\u0301", style="cyan")
    live = SessionBrowserLive(get_renderable=lambda: row, console=console, screen=True,
                              auto_refresh=False, redirect_stdout=False, redirect_stderr=False)
    try:
        live.start(refresh=True)
        output.seek(0)
        output.truncate(0)
        row.style = "bold green"
        live.refresh()
        changed = output.getvalue()
        assert "会話 e\u0301" in changed
        assert re.findall(r"\x1b\[(\d+);1H", changed) == ["1"]
        assert "\x1b[1;32m" in changed
    finally:
        live.stop()


def test_changed_list_does_not_rewrite_preview_in_the_same_row(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, force_interactive=True,
                      legacy_windows=False, width=50, height=4)
    row = Text("First session".ljust(20)).append(" │ ", style="dim").append("Stable preview", style="bold")
    state = [row]
    live = SessionBrowserLive(get_renderable=lambda: state[0], console=console, screen=True,
                              auto_refresh=False, redirect_stdout=False, redirect_stderr=False)
    try:
        live.start(refresh=True)
        output.seek(0)
        output.truncate(0)
        state[0] = Text("Next session".ljust(20)).append(" │ ", style="dim").append("Stable preview", style="bold")
        live.refresh()
        assert "Next session" in output.getvalue()
        assert "Stable preview" not in output.getvalue()
    finally:
        live.stop()


def test_changing_a_combining_mark_rewrites_its_base_character(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, force_interactive=True,
                      legacy_windows=False, width=20, height=4, no_color=True)
    row = [Text("e", style="bold").append("\u0301", style="cyan")]
    live = SessionBrowserLive(get_renderable=lambda: row[0], console=console, screen=True,
                              auto_refresh=False, redirect_stdout=False, redirect_stderr=False)
    try:
        live.start(refresh=True)
        output.seek(0)
        output.truncate(0)
        row[0] = Text("e", style="bold").append("\u0300", style="cyan")
        live.refresh()
        changed = output.getvalue()
        assert changed.startswith("\x1b[1;1H")
        assert "e\u0300" in re.sub(r"\x1b\[[\d;]*m", "", changed)
    finally:
        live.stop()
