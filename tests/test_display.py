import io
from types import SimpleNamespace

from rich.console import Console

from jarv import display
from jarv.display import (
    clip_middle,
    clip_tail,
    display_output,
    hidden_lines_hint,
    output_display_line_limit,
    output_display_split,
    terminal_size,
    tool_card,
)


def test_truecolor_detected_from_colorterm(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert display._truecolor_color_system() == "truecolor"


def test_truecolor_detected_under_wsl(monkeypatch):
    # WSL terminals render 24-bit colour even though TERM advertises only 256.
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    assert display._truecolor_color_system() == "truecolor"


def test_truecolor_left_to_auto_detection_otherwise(monkeypatch):
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert display._truecolor_color_system() is None


class FakeConsole:
    def __init__(self, width: int = 80, height: int = 24):
        self.width = width
        self.height = height

    @property
    def size(self):
        return SimpleNamespace(width=self.width, height=self.height)


def test_terminal_size_prefers_real_tty_size(monkeypatch):
    console = FakeConsole(width=70, height=20)

    monkeypatch.setattr(display.os, "get_terminal_size", lambda _fd: display.os.terminal_size((100, 40)))

    assert terminal_size(console=console) == (100, 40)


def test_terminal_size_prefers_output_tty_for_rendering(monkeypatch):
    console = FakeConsole(width=70, height=20)
    sizes = {
        0: display.os.terminal_size((140, 40)),
        1: display.os.terminal_size((100, 30)),
        2: display.os.terminal_size((120, 35)),
    }

    monkeypatch.setattr(display.os, "get_terminal_size", lambda fd: sizes[fd])

    assert terminal_size(console=console) == (100, 30)


def test_terminal_size_falls_back_to_console_size(monkeypatch):
    console = FakeConsole(width=70, height=20)

    def fail(_fd):
        raise OSError

    monkeypatch.setattr(display.os, "get_terminal_size", fail)

    assert terminal_size(console=console) == (70, 20)


def test_display_output_shows_head_tail_and_omitted_middle(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(
        display,
        "console",
        Console(file=stream, force_terminal=False, color_system=None, width=120),
    )
    monkeypatch.setattr(display, "terminal_size", lambda **_kwargs: (120, 24))
    output = "\n".join(f"Line {number}" for number in range(1, 41))

    display_output(output)

    rendered = stream.getvalue()
    assert "Line 1\n" in rendered
    assert "Line 5\n" in rendered
    assert "Line 6\n" not in rendered
    assert "Line 38\n" not in rendered
    assert "Line 39\n" in rendered
    assert "Line 40\n" in rendered
    assert "… 33 lines hidden …" in rendered


def test_display_output_does_not_truncate_output_within_screen_budget(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(
        display,
        "console",
        Console(file=stream, force_terminal=False, color_system=None, width=120),
    )
    monkeypatch.setattr(display, "terminal_size", lambda **_kwargs: (120, 24))
    output = "\n".join(f"Line {number}" for number in range(1, 9))

    display_output(output)

    rendered = stream.getvalue()
    assert rendered.count("Line ") == 8
    assert "lines hidden" not in rendered


def test_output_display_limit_is_one_third_of_screen_height(monkeypatch):
    monkeypatch.setattr(display, "terminal_size", lambda **_kwargs: (120, 60))

    assert output_display_line_limit(console=FakeConsole()) == 20


def test_output_display_split_biases_toward_head():
    head_lines, tail_lines = output_display_split(20)

    assert (head_lines, tail_lines) == (13, 6)
    assert head_lines > tail_lines
    assert head_lines + tail_lines + 1 == 20


def test_hidden_lines_hint_variants():
    assert hidden_lines_hint(33).plain == "… 33 lines hidden …"
    assert hidden_lines_hint(1).plain == "… 1 line hidden …"
    assert hidden_lines_hint(5, where="above").plain == "↑ 5 earlier lines hidden"
    assert hidden_lines_hint(2, where="below").plain == "… 2 more lines"
    assert (
        hidden_lines_hint(4, where="above", suffix="full reply will print when done").plain
        == "↑ 4 earlier lines hidden — full reply will print when done"
    )


def test_clip_middle_keeps_both_ends():
    lines = [f"Line {n}" for n in range(1, 41)]

    head, tail, hidden = clip_middle(lines, 8)

    assert head == lines[:5]
    assert tail == lines[-2:]
    assert hidden == 33
    assert clip_middle(lines, 40) == (lines, [], 0)


def test_clip_tail_keeps_newest_lines():
    lines = [f"Line {n}" for n in range(1, 11)]

    tail, hidden = clip_tail(lines, 4)

    assert tail == lines[-4:]
    assert hidden == 6
    assert clip_tail(lines, 10) == (lines, 0)


def test_tool_card_uses_shared_neutral_shell_and_tool_accent():
    stream = io.StringIO()
    test_console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=80,
    )

    test_console.print(tool_card("web_search", "PowerShell documentation"))

    rendered = stream.getvalue()
    assert "\u258e " in rendered
    assert "\u2315 Web search" in rendered
    assert "\u2713 done" not in rendered
    assert "PowerShell documentation" in rendered


def test_tool_cards_use_consistent_terminal_safe_symbols():
    stream = io.StringIO()
    test_console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=80,
    )

    for tool_name in ("run_command", "web_search", "spawn", "read", "edit", "ask_user"):
        test_console.print(tool_card(tool_name, "body"))

    rendered = stream.getvalue()
    assert "> Command" in rendered
    assert "\u2315 Web search" in rendered
    assert "\u21b3 Subagent" in rendered
    assert "\u2261 Read" in rendered
    assert "\u00b1 Edit" in rendered
    assert "? Ask user" in rendered


def test_print_tool_card_shows_failed_and_running_pills():
    stream = io.StringIO()
    test_console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=80,
    )

    test_console.print(
        tool_card("read", "missing.txt", status="failed", status_style="red")
    )
    test_console.print(
        tool_card("run_command", "> sleep 5", status="running 3s", status_style="blue")
    )

    rendered = stream.getvalue()
    assert "✗ failed" in rendered
    assert "● running 3s" in rendered


def test_fullscreen_tool_card_uses_box_and_right_status():
    stream = io.StringIO()
    test_console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=80,
    )

    test_console.print(
        tool_card(
            "web_search",
            "PowerShell documentation",
            display_mode="fullscreen",
        )
    )

    rendered = stream.getvalue()
    assert "\u256d\u2500" in rendered
    assert "\u2713 done" in rendered


def test_print_tool_cards_can_be_separated_by_one_blank_line():
    stream = io.StringIO()
    test_console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=80,
    )

    test_console.print(tool_card("web_search", "DuckDuckGo homepage"))
    test_console.print()
    test_console.print(tool_card("read", "C:\\Windows\\win.ini"))

    rendered = stream.getvalue()
    assert "DuckDuckGo homepage\n\n\u258e \u2261 Read" in rendered
