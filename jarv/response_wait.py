"""Small startup indicator, available before the agent and Markdown are loaded."""

import time

from rich.text import Text


_THINKING_FRAMES = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]


def response_wait_label(has_reasoning: bool) -> str:
    return "Thinking" if has_reasoning else "Waiting"


class ResponseWaitIndicator:
    """Animated response wait line with live elapsed timer."""

    def __init__(self, start_time: float):
        self._start = start_time
        self.has_reasoning = False

    def __rich_console__(self, console, options):
        now = time.perf_counter()
        frame = _THINKING_FRAMES[int(now * 10) % len(_THINKING_FRAMES)]
        label = response_wait_label(self.has_reasoning)
        yield Text(f"{frame}  {label}\u2026  {int(now - self._start)}s")


def start_response_wait(interactive: bool, start_time: float, *, console, live_factory=None):
    if not interactive:
        return None, None
    if live_factory is None:
        from rich.live import Live

        live_factory = Live
    indicator = ResponseWaitIndicator(start_time)
    live = live_factory(indicator, refresh_per_second=4, console=console,
                        auto_refresh=True, transient=True)
    # Live.start() otherwise waits for its first refresh interval (250 ms).
    live.start(refresh=True)
    return indicator, live
