import io
import time
from types import SimpleNamespace

from rich.console import Console

from jarv import display
from jarv.display import (
    display_output,
    output_display_line_limit,
    output_display_split,
    refresh_on_resize,
    terminal_size,
    tool_card,
)


class FakeConsole:
    def __init__(self, width: int = 80, height: int = 24):
        self.width = width
        self.height = height

    @property
    def size(self):
        return SimpleNamespace(width=self.width, height=self.height)


class FakeLive:
    def __init__(self):
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


def test_terminal_size_prefers_real_tty_size(monkeypatch):
    console = FakeConsole(width=70, height=20)

    monkeypatch.setattr(display.os, "get_terminal_size", lambda _fd: display.os.terminal_size((100, 40)))

    assert terminal_size(console=console) == (100, 40)


def test_terminal_size_falls_back_to_console_size(monkeypatch):
    console = FakeConsole(width=70, height=20)

    def fail(_fd):
        raise OSError

    monkeypatch.setattr(display.os, "get_terminal_size", fail)

    assert terminal_size(console=console) == (70, 20)


def test_refresh_on_resize_refreshes_when_terminal_size_changes(monkeypatch):
    console = FakeConsole()
    live = FakeLive()

    def fail(_fd):
        raise OSError

    monkeypatch.setattr(display.os, "get_terminal_size", fail)

    with refresh_on_resize(live, console=console, interval=0.01):
        time.sleep(0.03)
        assert live.refresh_count == 0

        console.width = 100
        deadline = time.monotonic() + 1
        while live.refresh_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

    assert live.refresh_count >= 1


def test_refresh_on_resize_responds_to_sigwinch(monkeypatch):
    console = FakeConsole()
    live = FakeLive()
    current = {"size": display.os.terminal_size((80, 24))}

    def get_terminal_size(_fd):
        return current["size"]

    class FakeSignal:
        SIGWINCH = 28
        SIG_IGN = object()

        def __init__(self):
            self.handler = None

        def getsignal(self, signum):
            assert signum == self.SIGWINCH
            return self.SIG_IGN

        def signal(self, signum, handler):
            assert signum == self.SIGWINCH
            previous = self.handler
            self.handler = handler
            return previous

    fake_signal = FakeSignal()
    monkeypatch.setattr(display.os, "get_terminal_size", get_terminal_size)
    monkeypatch.setattr(display, "signal", fake_signal)

    with refresh_on_resize(live, console=console, interval=1):
        current["size"] = display.os.terminal_size((120, 40))
        fake_signal.handler(fake_signal.SIGWINCH, None)
        deadline = time.monotonic() + 1
        while live.refresh_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

    assert live.refresh_count >= 1
    assert fake_signal.handler is fake_signal.SIG_IGN


def test_refresh_on_resize_uses_slower_polling_while_size_changes(monkeypatch):
    console = FakeConsole()
    live = FakeLive()
    refresh_times = []

    def fail(_fd):
        raise OSError

    def refresh():
        live.refresh_count += 1
        refresh_times.append(time.monotonic())

    monkeypatch.setattr(display.os, "get_terminal_size", fail)
    live.refresh = refresh

    with refresh_on_resize(live, console=console, interval=0.02, active_interval=0.08):
        console.width = 90
        deadline = time.monotonic() + 1
        while live.refresh_count == 0 and time.monotonic() < deadline:
            time.sleep(0.005)

        assert live.refresh_count == 1

        console.width = 100
        time.sleep(0.04)
        assert live.refresh_count == 1

        deadline = time.monotonic() + 1
        while live.refresh_count < 2 and time.monotonic() < deadline:
            time.sleep(0.005)

        assert live.refresh_count == 2
        assert refresh_times[1] - refresh_times[0] >= 0.06

        time.sleep(0.1)
        console.width = 110
        deadline = time.monotonic() + 1
        while live.refresh_count < 3 and time.monotonic() < deadline:
            time.sleep(0.005)

    assert live.refresh_count == 3


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
    assert "... 33 lines omitted from the middle ..." in rendered


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
    assert "omitted from the middle" not in rendered


def test_output_display_limit_is_one_third_of_screen_height(monkeypatch):
    monkeypatch.setattr(display, "terminal_size", lambda **_kwargs: (120, 60))

    assert output_display_line_limit(console=FakeConsole()) == 20


def test_output_display_split_biases_toward_head():
    head_lines, tail_lines = output_display_split(20)

    assert (head_lines, tail_lines) == (13, 6)
    assert head_lines > tail_lines
    assert head_lines + tail_lines + 1 == 20


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
    assert "\u250c\u2500" in rendered
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
    assert "DuckDuckGo homepage\n\n\u258e \u2193 Read" in rendered
