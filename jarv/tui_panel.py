"""Shared menu frames and their configurable outer border."""

import json

from rich.padding import Padding
from rich.panel import Panel
from rich.segment import Segment
from rich.text import Text

from .config_schema import get_setting

_menu_border: bool | None = None


def configure_menu_border(enabled: bool) -> None:
    """Apply the border preference to menus without their own config object."""
    global _menu_border
    _menu_border = bool(enabled)


def menu_border_enabled(config: dict | None = None) -> bool:
    if config is not None:
        return bool(get_setting(config, "headsup_border"))
    global _menu_border
    if _menu_border is None:
        # Standalone commands can open menus before load_config runs. Read the
        # preference once without creating/migrating config or session files.
        from .paths import CONFIG_FILE

        try:
            saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            saved = {}
        _menu_border = menu_border_enabled(saved if isinstance(saved, dict) else {})
    return _menu_border


def menu_inner_width(width: int, config: dict | None = None) -> int:
    """Content width with a border and gutter, or the full width without them."""
    return max(1, width - (4 if menu_border_enabled(config) else 0))


def menu_frame_rows(config: dict | None = None, *, footer: bool = False) -> int:
    """Borderless menus share title and metadata instead of reserving a bottom row."""
    return 2 if footer or menu_border_enabled(config) else 1


class MenuPanel(Panel):
    """Borderless menus put metadata in the header and use every content column.

    Set ``footer`` for actual bottom bars, such as editor controls or HUD status.
    """

    def __init__(self, *args, border: bool | None = None, footer: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.border = border
        self.footer = footer

    def __rich_console__(self, console, options):
        border = menu_border_enabled() if self.border is None else self.border
        if border:
            yield from super().__rich_console__(console, options)
            return

        width = max(1, min(options.max_width, self.width or options.max_width))
        height = self.height if self.height is not None else options.height
        body_height = max(0, height - 1 - int(self.footer)) if height is not None else None
        top, _, bottom, _ = Padding.unpack(self.padding)
        body = Padding(self.renderable, (top, 0, bottom, 0)) if top or bottom else self.renderable

        def label_text(value):
            if isinstance(value, str):
                text = Text.from_markup(value)
            else:
                text = value.copy() if value is not None else Text("")
            text.plain = text.plain.replace("\n", " ")
            text.no_wrap = True
            text.overflow = "ellipsis"
            return text

        def label(value, align):
            text = label_text(value)
            text.justify = align
            return console.render_lines(text, options.update(width=width, height=1))[0]

        title = label_text(self.title)
        if not self.footer and self.subtitle:
            title.truncate(width, overflow="ellipsis")
            subtitle = label_text(self.subtitle)
            available = width - title.cell_len - 2
            if available > 0:
                subtitle.truncate(available, overflow="ellipsis")
                title.append(" " * (width - title.cell_len - subtitle.cell_len))
                title.append_text(subtitle)
        lines = [label(title, self.title_align)]
        lines.extend(console.render_lines(
            body, options.update(width=width, height=body_height),
            style=console.get_style(self.style),
        ))
        if self.footer:
            lines.append(label(self.subtitle, self.subtitle_align))
        if height is not None:
            lines = lines[:height]
        for line in lines:
            yield from line
            yield Segment.line()
