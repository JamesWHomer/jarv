import io
import subprocess
import sys

from rich.console import Console

from jarv.agent import InPlaceLive, StreamingMarkdownPreview, TailMarkdown


def test_cli_import_does_not_load_heavy_rendering_or_provider_sdks():
    code = (
        "import sys; "
        "import jarv.cli; "
        "blocked = ['rich.markdown', 'httpx', 'jarv.agent']; "
        "print([name for name in blocked if name in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "[]"


def test_ordinary_message_disambiguation_does_not_import_commands():
    code = (
        "import sys; "
        "import jarv.cli; "
        "jarv.cli._maybe_command('ordinary', []); "
        "print('jarv.commands' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "False"


def test_streaming_markdown_preview_bounds_source_text():
    text = "\n".join(f"line {i}" for i in range(2000))
    preview = TailMarkdown(text, max_lines=8, max_source_chars=200)

    assert len(preview._text) < 400
    assert "line 1999" in preview._text
    assert "line 0" not in preview._text


def test_streaming_markdown_preview_coalesces_fast_deltas():
    class FakeLive:
        def __init__(self):
            self.updates = []

        def update(self, renderable, *, refresh=False):
            self.updates.append((renderable, refresh))

    ticks = iter([0.0, 0.01, 0.02, 0.09, 0.10, 0.11])
    live = FakeLive()
    preview = StreamingMarkdownPreview(
        live,
        max_lines=8,
        refresh_interval=0.08,
        clock=lambda: next(ticks),
    )

    for delta in ("a", "b", "c", "d", "e"):
        preview.append(delta)
    preview.flush(refresh=False)

    assert preview.text == "abcde"
    assert len(live.updates) == 3
    assert [refresh for _, refresh in live.updates] == [True, True, False]
    assert live.updates[-1][0]._text == "abcde"


def test_streaming_markdown_preview_skips_clean_flush():
    class FakeLive:
        def __init__(self):
            self.update_count = 0

        def update(self, _renderable, *, refresh=False):
            self.update_count += 1

    live = FakeLive()
    preview = StreamingMarkdownPreview(
        live,
        max_lines=8,
        clock=lambda: 0.0,
    )

    preview.append("complete")
    preview.flush()

    assert live.update_count == 1


def test_streaming_live_overwrites_rows_without_clearing_entire_block_first():
    output = io.StringIO()
    console = Console(
        file=output,
        force_terminal=True,
        color_system=None,
        width=120,
        height=30,
    )
    live = InPlaceLive(
        TailMarkdown("", 28),
        console=console,
        auto_refresh=False,
        transient=True,
        vertical_overflow="crop",
        redirect_stdout=False,
        redirect_stderr=False,
    )
    live.start(refresh=True)
    output.seek(0)
    output.truncate(0)

    live.update(TailMarkdown("word " * 500, 28), refresh=True)
    repaint = output.getvalue()
    live.stop()

    assert "\x1b[2K" not in repaint
    assert "\x1b[0K" in repaint
