import time
from types import SimpleNamespace

from jarv import display
from jarv.display import refresh_on_resize, terminal_size


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
