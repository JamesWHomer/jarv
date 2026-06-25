"""Regression tests for the oneshot inline streaming preview render fixes.

Covers three bugs the rewrite addressed:
  * the char-level source crop and the line-level crop each prepended their own
    "hidden" banner, stacking two hints;
  * the row budget was frozen at the first token, so a mid-stream resize left
    the crop bound stale;
  * the budget could not be pinned for deterministic rendering.
"""

import io

from rich.console import Console

import jarv.agent_ui as agent_ui
from jarv.agent_ui import TailMarkdown


def _render(renderable, *, width: int = 80, height: int = 24) -> str:
    console = Console(
        file=io.StringIO(),
        force_terminal=True,
        color_system=None,
        width=width,
        height=height,
    )
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


def test_tail_markdown_shows_single_hint_not_stacked_banner():
    # Long content with both crops triggered (tiny char cap *and* tiny line
    # budget) must still surface exactly one hint, never the old char-crop
    # banner stacked above the line-crop hint.
    text = "\n\n".join(f"paragraph {i}" for i in range(200))
    out = _render(TailMarkdown(text, max_lines=6, max_source_chars=120))

    assert "streaming content hidden" not in out  # old char-crop banner is gone
    assert out.count("earlier line") == 1  # exactly one line-crop hint
    assert "hidden" in out


def test_tail_markdown_row_budget_tracks_live_terminal_height(monkeypatch):
    sizes = {"value": (80, 10)}
    monkeypatch.setattr(
        agent_ui, "terminal_size", lambda *, console=None: sizes["value"]
    )
    tail = TailMarkdown("", None, reserve_rows=2)
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)

    assert tail._row_budget(console) == 8  # 10 - reserve_rows

    # A mid-stream resize is reflected on the next paint rather than frozen.
    sizes["value"] = (80, 40)
    assert tail._row_budget(console) == 38


def test_tail_markdown_pinned_budget_never_consults_terminal(monkeypatch):
    calls = {"n": 0}

    def _spy(*, console=None):
        calls["n"] += 1
        return (80, 999)

    monkeypatch.setattr(agent_ui, "terminal_size", _spy)
    tail = TailMarkdown("x", max_lines=5)
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None)

    assert tail._row_budget(console) == 5
    assert calls["n"] == 0


def test_tail_markdown_height_holds_steady_across_overflow_threshold(monkeypatch):
    # The block reserves one top row in both regimes, so its rendered height
    # grows to the budget and then holds — no one-row jump as the hint appears.
    monkeypatch.setattr(
        agent_ui, "terminal_size", lambda *, console=None: (80, 8)
    )

    def _height(line_count: int) -> int:
        body = "\n\n".join(f"p{i}" for i in range(line_count))
        rendered = _render(TailMarkdown(body, None))
        return len(rendered.rstrip("\n").splitlines())

    # Just below and well past the overflow threshold should agree on height.
    assert _height(20) == _height(60)
