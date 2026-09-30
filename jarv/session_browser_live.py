"""Update only changed terminal rows while browsing sessions."""

from rich.control import Control
from rich.live import Live
from rich.cells import get_character_cell_size


class _ChangedRows:
    def __init__(self, live):
        self.live = live
        self.previous = None
        self.size = None

    def reset(self):
        self.previous = None
        self.size = None

    def __rich_console__(self, console, options):
        # Rich's Screen has already padded the frame to the terminal's exact
        # size. Full-width replacement also removes text that became shorter.
        size = options.size
        lines = console.render_lines(self.live._live_render.renderable, options, pad=False)
        previous = self.previous if size == self.size else None
        changed = False
        for y, line in enumerate(lines):
            old = previous[y] if previous is not None and y < len(previous) else None
            if line == old:
                continue
            start, end, column = 0, len(line), 0
            if old is not None:
                while start < min(len(line), len(old)) and line[start] == old[start]:
                    column += line[start].cell_length
                    start += 1
                if start and any(
                    start < len(parts) and parts[start].text and get_character_cell_size(parts[start].text[0]) == 0
                    for parts in (line, old)
                ):
                    # A changed/deleted combining mark must replace its base
                    # too; otherwise the terminal can retain the old accent.
                    while start:
                        start -= 1
                        column -= line[start].cell_length
                        if line[start].cell_length:
                            break
                suffix = 0
                while suffix < min(len(line), len(old)) - start and line[-suffix - 1] == old[-suffix - 1]:
                    suffix += 1
                end -= suffix
                # Replacing a base character must also redraw any combining
                # marks that share its terminal cell, even if their style is
                # unchanged. Segment boundaries always preserve wide chars.
                while end < len(line) and line[end].text and get_character_cell_size(line[end].text[0]) == 0:
                    end += 1
            changed = True
            yield Control.move_to(column, y).segment
            yield from line[start:end]
        if changed:
            # Absolute positioning clears the pending-wrap state at the last
            # column without a newline that could scroll the bottom row.
            yield Control.home().segment
        self.previous, self.size = lines, size


class SessionBrowserLive(Live):
    """Keep Rich's screen lifecycle, replacing its full-frame refresh writes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.changed_rows = _ChangedRows(self)

    def start(self, refresh=False):
        if not self._started:
            self.changed_rows.reset()
        super().start(refresh=refresh)

    def refresh(self):
        if (self._nested or not self._alt_screen or not self.console.is_interactive
                or self.console.legacy_windows or self.console.is_dumb_terminal):
            return super().refresh()
        with self._lock:
            self._live_render.set_renderable(self.renderable)
            # Cursor-addressed rows deliberately contain no newlines. Rich's
            # normal print cropping would treat them as one long physical row.
            self.console.print(Control(), crop=False, end="")

    def process_renderables(self, renderables):
        # Live.refresh prints one empty Control to trigger its render hook.
        # Unexpected console output and unsupported terminals use Rich's usual
        # path, then force a complete repaint on the next normal refresh.
        refresh = len(renderables) == 1 and isinstance(renderables[0], Control) and not renderables[0].segment.text
        if self._alt_screen and self.console.is_interactive and not self.console.legacy_windows and refresh:
            return [self.changed_rows]
        self.changed_rows.reset()
        return super().process_renderables(renderables)
