"""Equivalence and work bounds for the runtime performance hot paths."""
import copy
import random
import re

import pytest
from rich.text import Text

from jarv import context_budget, text_editor
from jarv.headsup import TranscriptEntry
from jarv.tui_frame import window_transcript


def test_text_fast_paths_preserve_unicode_and_markdown():
    from jarv.display import flatten_headings
    from jarv.unicode_safety import sanitize_text

    for value in ("", "ASCII\t\r\n", "界 é 🙂", "bad\ud800text\udfff", "# heading\nbody", "## title\n### next", "plain # inline"):
        assert sanitize_text(value) == value.encode("utf-8", errors="replace").decode("utf-8")
        assert flatten_headings(value) == re.sub(r"^#{1,6}\s+(.+)$", r"**\1**", value, flags=re.MULTILINE)


def _original_compaction(history, target):
    """Pre-optimization algorithm, retained as an independent semantic oracle."""
    modified = False
    while context_budget.estimate_history_tokens("test", history) > target:
        candidates = [(start, end) for start, end in context_budget.iter_turn_ranges(history)[:-1]
                      if history[start].get("type") != "compacted_summary"]
        if not candidates:
            break
        start, end = candidates[0]
        history[start:end] = [{"role": "user", "type": "compacted_summary",
                               "content": context_budget.summarize_turn_items(history[start:end])}]
        modified = True
    return modified


def test_compaction_preserves_exact_context_across_budgets():
    rng = random.Random(42)
    for _ in range(100):
        history = [{"type": "reasoning", "id": "rs_leading", "summary": []}]
        for turn in range(rng.randrange(1, 15)):
            history.extend([
                {"role": "user", "content": "界 question\n" * rng.randrange(80),
                 "type": rng.choice(["message", "message", "compacted_summary"])},
                {"type": "reasoning", "id": f"rs_{turn}", "summary": [{"text": "thought"}],
                 "provider_content": [{"type": "thinking", "signature": "preserve"}]},
                {"type": "function_call", "id": f"fc_{turn}", "call_id": str(turn),
                 "name": "read", "arguments": '{"input":"x"}'},
                {"type": "function_call_output", "call_id": str(turn),
                 "output": rng.choice(["result\n" * rng.randrange(200), [{"type": "input_image", "image_url": "data:image/png;base64,abc"}]])},
                {"role": "assistant", "content": "answer" * rng.randrange(100)},
            ])
        original = copy.deepcopy(history)
        for target in (-1, 0, 100, 1000, context_budget.estimate_history_tokens("test", history), 100000):
            expected, actual = copy.deepcopy(history), copy.deepcopy(history)
            expected_changed = _original_compaction(expected, target)
            changed = context_budget.compact_oldest_turns(actual, model="test", config={},
                                                         instructions="unchanged", tools=[], target_tokens=target)
            assert changed == expected_changed
            assert actual == expected
        assert history == original


def test_compaction_token_work_is_linear(monkeypatch):
    history = [item for _ in range(200) for item in (
        {"role": "user", "content": "x" * 1000}, {"role": "assistant", "content": "y" * 1000})]
    counted = 0
    original = context_budget.estimate_item_tokens

    def count(model, item):
        nonlocal counted
        counted += 1
        return original(model, item)

    monkeypatch.setattr(context_budget, "estimate_item_tokens", count)
    context_budget.compact_oldest_turns(history, model="test", config={}, instructions="", tools=[], target_tokens=1)
    assert counted < 3 * 400


@pytest.mark.parametrize("width", [1, 4, 17, 96])
def test_transcript_viewport_matches_full_render_through_scroll_and_changes(width, headsup_app_factory):
    app = headsup_app_factory()
    app.entries = [TranscriptEntry("assistant", Text("界 é words\n" * (i % 4), style="bold"),
                                   spacer_before=bool(i % 2)) for i in range(30)]
    for change in range(5):
        if change == 1:
            app.entries[-1].renderable.append("appended response")
            app.entries[-1].invalidate()
        elif change == 2:
            app.entries.append(TranscriptEntry("user", Text("new prompt")))
        elif change == 3:
            app.add_notice(Text("feedback\n" * 3))
        elif change == 4:
            app.add_notice(Text("replacement feedback"))
        for rows in (1, 6, 28, 200):
            full = app._transcript_lines(width)
            for offset in (-10, 0, 1, 15, len(full) - rows, len(full) + 100):
                expected, expected_offset = window_transcript(full, rows, offset)
                actual, actual_offset = app._transcript_window(width, rows, offset)
                assert actual == expected
                assert actual_offset == expected_offset
    app.entries = []
    app._notice = None
    assert app._transcript_window(width, 6, 100) == ([Text("")], 0)


def test_transcript_does_not_render_entries_outside_viewport(headsup_app_factory):
    class Hidden:
        def __rich_console__(self, console, options):
            raise AssertionError("offscreen entry rendered")
            yield  # pragma: no cover

    app = headsup_app_factory()
    app.entries = [TranscriptEntry("assistant", Hidden()), TranscriptEntry("user", Text("visible\n" * 30))]
    lines, offset = app._transcript_window(96, 6, 0)
    assert len(lines) == 6
    assert offset == 0


def test_saved_markdown_is_lazy_and_renders_identically():
    from rich.markdown import Markdown
    from jarv.display import flatten_headings, rendered_text_lines
    from jarv.headsup import _HistoryMarkdown

    content = "# Title\n\n- **bold** and 界 é\n- [link](https://example.test)\n\n```python\nprint('test')\n```\n\n| A | B |\n| - | - |\n| 1 | 2 |"
    saved = _HistoryMarkdown(content)
    assert saved._markdown is None
    for width in (10, 40, 96):
        assert rendered_text_lines(saved, width) == rendered_text_lines(Markdown(flatten_headings(content)), width)
        parsed = saved._markdown
        rendered_text_lines(saved, width)
        assert saved._markdown is parsed


def _original_rows(value, width):
    from rich.cells import get_character_cell_size
    width = max(1, width)
    rows, absolute = [], 0
    for line in value.split("\n"):
        start, cells = 0, 0
        for index, char in enumerate(line):
            size = get_character_cell_size(char)
            if size and cells and cells + size > width:
                rows.append((absolute + start, absolute + index))
                start, cells = index, 0
            cells += size
        rows.append((absolute + start, absolute + len(line)))
        if cells >= width:
            rows.append((absolute + len(line), absolute + len(line)))
        absolute += len(line) + 1
    return rows


def test_editor_windows_preserve_wrapping_cursor_and_styling():
    rng = random.Random(42)
    for value in ["", "abc " * 200, "line\n" * 100] + [
        "".join(rng.choices("ab \n\t\r界é🙂", k=150)) for _ in range(40)
    ]:
        for width in (1, 4, 19):
            assert text_editor.visual_rows(value, width) == _original_rows(value, width)
            for cursor in (0, len(value) // 2, len(value)):
                state = {"buffer": value, "cursor": cursor}
                styles = dict(highlight_spans=[(0, 10)], selection_span=(5, 20), selection_style="bold", masked=False)
                full, index = text_editor.render_visual_lines(state, width, **styles)
                for maximum in (None, 0, 1, 6):
                    visible = len(full) if maximum is None else max(1, maximum)
                    start = max(0, min(index - visible + 1, len(full) - visible))
                    lines, row, offset = text_editor.render_visual_line_window(state, width, max_lines=maximum, **styles)
                    assert (lines, row, offset) == (full[start:start + visible], index - start, start)


def test_editor_only_styles_visible_rows(monkeypatch):
    calls = 0
    original = text_editor._append_segment

    def append(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(text_editor, "_append_segment", append)
    value = "line\n" * 10000
    lines, _, _ = text_editor.render_visual_line_window({"buffer": value, "cursor": len(value)}, 80, max_lines=6)
    assert len(lines) == calls == 6
