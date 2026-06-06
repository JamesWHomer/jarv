import subprocess
import sys

from jarv.agent import TailMarkdown


def test_cli_import_does_not_load_heavy_rendering_or_provider_sdks():
    code = (
        "import sys; "
        "import jarv.cli; "
        "blocked = ['rich.markdown', 'httpx', 'openai', 'litellm', 'jarv.agent']; "
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
