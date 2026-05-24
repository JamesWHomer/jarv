import time
from types import SimpleNamespace

from jarv.display import refresh_on_resize


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


def test_refresh_on_resize_refreshes_when_terminal_size_changes():
    console = FakeConsole()
    live = FakeLive()

    with refresh_on_resize(live, console=console, interval=0.01):
        time.sleep(0.03)
        assert live.refresh_count == 0

        console.width = 100
        deadline = time.monotonic() + 1
        while live.refresh_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

    assert live.refresh_count >= 1
