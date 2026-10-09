"""End-to-end picker state, persistence, and responsive rendering contracts."""

import io
import json
import threading
import time
from collections import deque
from types import SimpleNamespace
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text

from jarv import history, session_browser as browser, session_store, tui_panel
from jarv.command_input import TextInput


@pytest.fixture
def picker(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "SESSIONS_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(history, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(session_store, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(tui_panel, "_menu_border", False)
    data = {"sessions": {}, "terminals": {"terminal": "session-000000000000"}}
    rows = []
    for index, title in enumerate(["Fix remote desktop sizing", "Check memory usage", "Set up Python", "旧い会話"]):
        sid = f"session-{index:012d}"
        path = tmp_path / f"history-{index}.json"
        items = [
            {"role": "user", "content": title},
            {"role": "assistant", "content": "First response."},
            {"role": "user", "content": [{"type": "input_text", "text": "What about the aspect ratio?"}]},
            {"role": "assistant", "content": "Match the host display resolution to the client."},
        ] if index == 0 else [{"role": "user", "content": title}]
        path.write_text(json.dumps(items), encoding="utf-8")
        data["sessions"][sid] = {"label": f"Terminal {index}", "history_file": str(path), "archived": index == 3}
        rows.append(dict(sid=sid, short_id=sid[:14], snippet="", snippet_loaded=False,
                         time_str="5m ago", date_group="Today" if index < 2 else "Yesterday",
                         is_current=index == 0, archived=index == 3))
    history.save_sessions(data)
    data = history.load_sessions()
    screen = browser.SessionBrowserScreen(data=data, sessions=data["sessions"], terminals=data["terminals"],
                                         rows=rows, current_session_id=rows[0]["sid"], background=False)
    monkeypatch.setattr(screen, "_start_prefetch", lambda: None)
    return screen


def render(screen, monkeypatch, width=120, height=24):
    monkeypatch.setattr(browser, "terminal_size", lambda **kwargs: (width, height))
    output = io.StringIO()
    console = Console(file=output, width=width, height=height, color_system=None, legacy_windows=False)
    screen.console = console
    console.print(screen.render())
    return output.getvalue()


@pytest.mark.parametrize("border", [True, False])
@pytest.mark.parametrize("width,height", [(32, 8), (42, 12), (80, 24), (99, 24), (100, 24), (108, 24), (120, 24), (160, 40)])
def test_responsive_picker_preserves_frame_and_essential_controls(picker, monkeypatch, border, width, height):
    monkeypatch.setattr(tui_panel, "_menu_border", border)
    output = render(picker, monkeypatch, width, height)
    lines = output.splitlines()
    assert len(lines) == height
    assert all(cell_len(line) == width for line in lines)
    controls = lines[-2 if border else -1]
    assert "Enter" in controls and "Esc" in controls
    assert "Ctrl+F" in controls
    assert "[current]" in output
    assert "PREVIEW" in output if width >= 100 else "PREVIEW" not in output
    if width >= 80:
        assert "Fix remote desktop sizing" in output
        assert "Archived 1" in output
    if width >= 100:
        layout = picker._list_layout(picker._visible_rows_list())
        assert layout["left_width"] < layout["right_width"]
        assert layout["right_width"] >= 50
        assert "aspect ratio" in output
        assert "Match the host" in output
        assert "First response" not in output


def test_rename_persists_without_changing_terminal_label_and_can_reset(picker, monkeypatch):
    render(picker, monkeypatch)
    sid = picker.selected_sid
    picker.on_key("r", 1)
    picker.on_key(TextInput("[red]Display sizing[/red]"), 1)
    picker.on_key("ENTER", 1)
    saved = history.load_sessions()["sessions"][sid]
    assert saved["title"] == "[red]Display sizing[/red]"
    assert saved["label"] == "Terminal 0"
    assert "[red]Display sizing[/red]" in render(picker, monkeypatch)
    picker.on_key("r", 1)
    picker.on_key("BACKSPACE", 1)
    picker.on_key("ENTER", 1)
    assert "title" not in history.load_sessions()["sessions"][sid]
    assert "Fix remote desktop sizing" in render(picker, monkeypatch)


@pytest.mark.parametrize("literal", ["ENTER", "ESC"])
def test_rename_keeps_pasted_key_names_literal(picker, literal):
    picker.on_key("r", 1)
    sid = picker.rename_sid

    picker.on_key(TextInput(literal), 1)

    assert picker.rename_sid == sid
    assert picker.rename_editor["buffer"] == literal


@pytest.mark.parametrize("literal", ["ENTER", "ESC", "TAB", "DOWN", "CTRL_F"])
def test_search_keeps_pasted_key_names_literal(picker, literal):
    picker.on_key("CTRL_F", 1)

    picker.on_key(TextInput(literal), 1)

    assert picker.search_active
    assert picker.search_query == literal


def test_pasted_text_cannot_trigger_session_actions(picker):
    original = dict(picker.sessions)
    picker.on_key(TextInput("d"), 1)
    picker.on_key(TextInput("d"), 1)

    assert picker.sessions == original
    assert picker.arm_delete_sids is None


def test_rename_cancel_and_failed_save_preserve_existing_title(picker, monkeypatch):
    original = history.SESSIONS_FILE.read_bytes()
    picker.on_key("r", 1)
    picker.on_key(TextInput("Discard this"), 1)
    picker.on_key("ESC", 1)
    assert history.SESSIONS_FILE.read_bytes() == original
    picker.on_key("r", 1)
    picker.on_key(TextInput("Cannot save"), 1)
    monkeypatch.setattr(browser, "save_sessions", lambda data: (_ for _ in ()).throw(OSError("disk full")))
    picker.on_key("ENTER", 1)
    assert "title" not in picker.sessions[picker.selected_sid]
    assert "Couldn't rename" in render(picker, monkeypatch)


def test_transcript_search_shows_match_and_preserves_filter_after_preview(picker, monkeypatch):
    render(picker, monkeypatch)
    picker.on_key("/", 1)
    assert not picker.search_active
    picker.on_key("CTRL_F", 1)
    picker.on_key(TextInput("ASPECT RATIO"), 1)
    assert len(picker._visible_rows_list()) == 1
    output = render(picker, monkeypatch)
    assert "aspect ratio" in output
    assert "1 result" in output
    picker.on_key("ENTER", 1)
    picker.on_key("p", 1)
    assert "aspect ratio" in render(picker, monkeypatch)
    picker.on_key("ESC", 1)
    assert picker.search_query == "ASPECT RATIO"
    assert picker.preview_sid is None
    picker.on_key("ESC", 1)
    assert not picker.search_query


def test_preview_opens_latest_exchange_and_is_read_only(picker, monkeypatch):
    render(picker, monkeypatch)
    before = history.SESSIONS_FILE.read_bytes()
    picker.on_key("p", 1)
    output = render(picker, monkeypatch)
    assert "aspect ratio" in output and "First response" not in output
    picker.on_key("RIGHT", 1)
    assert picker.preview_sid == picker.rows[0]["sid"]
    picker.on_key("LEFT", 1)
    assert picker.preview_sid is None
    assert picker.selected_sid == picker.rows[0]["sid"]
    assert history.SESSIONS_FILE.read_bytes() == before


def test_help_and_empty_search_return_to_the_list(picker, monkeypatch):
    picker.on_key("?", 1)
    assert "shortcuts" in render(picker, monkeypatch)
    picker.on_key("ESC", 1)
    picker.on_key("CTRL_F", 1)
    picker.on_key(TextInput("absent phrase"), 1)
    assert "No conversations match" in render(picker, monkeypatch)
    picker.on_key("ESC", 1)
    assert len(picker._visible_rows_list()) == 3


def test_archived_row_is_readable_and_resume_action_is_explicit(picker, monkeypatch):
    picker.on_key("TAB", 1)
    output = render(picker, monkeypatch)
    assert "旧い会話" in output
    assert "[archived]" in output
    assert "Enter Restore & resume" in output


def test_first_paint_loads_only_visible_histories_and_reuses_preview(picker, monkeypatch):
    for index in range(1000):
        row = dict(picker.rows[1], sid=f"extra-{index}", snippet_loaded=False)
        picker.rows.append(row)
        picker.sessions[row["sid"]] = dict(picker.sessions[picker.rows[1]["sid"]])
    calls = []
    load = browser.load_history
    monkeypatch.setattr(browser, "load_history", lambda path: (calls.append(path), load(path))[1])
    render(picker, monkeypatch)
    first_paint = len(calls)
    assert first_paint <= 24
    render(picker, monkeypatch)
    assert len(calls) == first_paint


def test_close_does_not_delete_files_reused_by_another_terminal(picker, monkeypatch):
    row = picker.rows[0]
    path = picker.sessions[row["sid"]]["history_file"]
    picker.on_key("d", 1)
    picker.on_key("d", 1)
    newer = history.load_sessions()
    newer["sessions"][row["sid"]] = {"history_file": path, "title": "New conversation"}
    history.save_sessions(newer)
    deleted = []
    monkeypatch.setattr(browser, "delete_session_files", deleted.append)
    picker.on_stop()
    assert not deleted
    assert history.load_sessions()["sessions"][row["sid"]]["title"] == "New conversation"


def test_archive_undo_and_deferred_delete_preserve_real_history_files(picker):
    sid = picker.selected_sid
    original = json.loads(Path(picker.sessions[sid]["history_file"]).read_text(encoding="utf-8"))
    picker.on_key("a", 1)
    assert history.load_sessions()["sessions"][sid]["archived"]
    picker.on_key("u", 1)
    restored = Path(picker.sessions[sid]["history_file"])
    normalized = json.loads(restored.read_text(encoding="utf-8"))
    assert [{key: value for key, value in item.items() if key != "id"}
            for item in normalized] == original
    assert history.load_sessions()["terminals"]["terminal"] == sid
    picker.on_key("d", 1)
    picker.on_key("d", 1)
    assert restored.exists()
    assert sid not in history.load_sessions()["sessions"]
    picker.on_key("u", 1)
    assert json.loads(restored.read_text(encoding="utf-8")) == normalized
    assert sid in history.load_sessions()["sessions"]
    picker.on_key("d", 1)
    picker.on_key("d", 1)
    picker.on_stop()
    assert not restored.exists()


def long_conversation(picker):
    path = Path(picker.sessions[picker.selected_sid]["history_file"])
    items = json.loads(path.read_text(encoding="utf-8"))
    items[-1]["content"] = "\n\n".join(f"Preview paragraph {i}: read this part of the conversation." for i in range(50))
    path.write_text(json.dumps(items), encoding="utf-8")


def test_tab_follows_the_displayed_order_and_clears_selection(picker, monkeypatch):
    output = render(picker, monkeypatch)
    assert output.index("Active 3") < output.index("Archived 1") < output.index("All 4")
    picker.on_key(" ", 1)
    for expected in ("archived", "all", "active", "archived"):
        picker.on_key("TAB", 1)
        assert picker.view_mode == expected
        assert not picker.marked_sids
        assert f"[{expected.title()}" in render(picker, monkeypatch)


@pytest.mark.parametrize("border", [False, True])
@pytest.mark.parametrize("focus", ["sessions", "preview"])
def test_selection_background_stays_inside_sessions_pane(picker, monkeypatch, border, focus):
    monkeypatch.setattr(tui_panel, "_menu_border", border)
    render(picker, monkeypatch)
    picker.pane_focus = focus
    picker.on_key(" ", 1) if focus == "sessions" else None
    layout = picker._list_layout(picker._visible_rows_list())
    con = picker.console
    lines = con.render_lines(picker.render(), con.options.update(width=120, height=24), pad=True)
    left_edge = 2 if border else 0
    right_edge = left_edge + layout["left_width"]
    highlights = 0
    for line in lines:
        x = 0
        for segment in line:
            end = x + segment.cell_length
            color = segment.style.bgcolor if segment.style else None
            if color is not None and color.name in ("#17333b", "#15252b"):
                highlights += 1
                assert left_edge <= x < end <= right_edge
            x = end
    assert highlights > 0


@pytest.mark.parametrize("width", [100, 120])
def test_preview_focus_scrolls_without_moving_session_and_remembers_position(picker, monkeypatch, width):
    long_conversation(picker)
    render(picker, monkeypatch, width)
    sid = picker.selected_sid
    list_offset = picker.offset
    picker.on_key("RIGHT", 1)
    assert picker.pane_focus == "preview"
    layout = picker._list_layout(picker._visible_rows_list())
    key = picker._preview_key(sid, layout["right_width"])
    original = picker.preview_positions[key]
    for action in ("DOWN", "PAGEDOWN", "MOUSE_WHEEL_DOWN"):
        before = picker.preview_positions[key]
        picker.on_key(action, 1)
        assert picker.preview_positions[key] > before
        assert picker.selected_sid == sid and picker.offset == list_offset
    scrolled = picker.preview_positions[key]
    assert scrolled > original
    output = render(picker, monkeypatch, width)
    assert "› PREVIEW" in output and "← Sessions" in output
    picker.on_key("LEFT", 1)
    picker.on_key("RIGHT", 1)
    assert picker.preview_positions[key] == scrolled
    picker.on_key("HOME", 1)
    assert picker.preview_positions[key] == 0
    picker.on_key("END", 1)
    assert "Preview paragraph 49" in render(picker, monkeypatch, width)
    picker.on_key("ESC", 1)
    assert picker.pane_focus == "sessions"


def test_preview_does_not_accept_hidden_row_management_shortcuts(picker, monkeypatch):
    render(picker, monkeypatch)
    picker.on_key("RIGHT", 1)
    for key in ("r", "d", "a", " ", "SHIFT_DOWN", "RIGHT"):
        picker.on_key(key, 1)
    assert picker.pane_focus == "preview"
    assert picker.rename_sid is None and picker.arm_delete_sids is None
    assert not picker.marked_sids and not picker.undo_actions
    assert picker.selected_sid == picker.rows[0]["sid"]


def test_fullscreen_returns_to_preview_focus_and_saved_pane_position(picker, monkeypatch):
    long_conversation(picker)
    render(picker, monkeypatch)
    picker.on_key("RIGHT", 1)
    picker.on_key("PAGEDOWN", 1)
    key = picker._preview_key(picker.selected_sid, picker._list_layout(picker._visible_rows_list())["right_width"])
    saved = picker.preview_positions[key]
    picker.on_key("p", 1)
    render(picker, monkeypatch)
    picker.on_key("p", 1)
    assert picker.preview_sid is None and picker.pane_focus == "preview"
    assert picker.preview_positions[key] == saved

    picker.on_key("p", 1)
    picker.on_key("PAGEDOWN", 1)
    expanded = render(picker, monkeypatch)
    passage = next(line.split(":")[0].strip() for line in expanded.splitlines() if "Preview paragraph" in line)
    picker.on_key("p", 1)
    assert passage in render(picker, monkeypatch)


def test_delete_confirmation_shows_only_confirmation_controls(picker, monkeypatch):
    render(picker, monkeypatch)
    previous_height = picker._list_layout(picker._visible_rows_list())["body_height"]
    picker.on_key("d", 1)
    output = render(picker, monkeypatch)
    assert 'Delete "Fix remote desktop sizing" permanently?' in output
    assert "d Confirm" in output and "any other key cancels" in output
    assert "Enter Resume" not in output and "Space Select" not in output
    assert picker._list_layout(picker._visible_rows_list())["body_height"] == previous_height
    picker.on_key("ESC", 1)
    assert "Enter Resume" in render(picker, monkeypatch)
    assert len(picker.rows) == 4


def test_narrow_right_opens_preview_and_resize_returns_to_sessions(picker, monkeypatch):
    render(picker, monkeypatch)
    picker.on_key("RIGHT", 1)
    render(picker, monkeypatch, 99, 24)
    picker.on_resize((99, 24))
    assert picker.pane_focus == "sessions"
    sid = picker.selected_sid
    picker.on_key("RIGHT", 1)
    assert picker.preview_sid == sid
    picker.on_key("LEFT", 1)
    assert picker.preview_sid is None and picker.selected_sid == sid
    picker.on_key("LEFT", 1)
    assert picker.selected_sid == sid


def test_search_from_preview_returns_results_and_tab_still_cycles(picker, monkeypatch):
    render(picker, monkeypatch)
    picker.on_key("RIGHT", 1)
    picker.on_key("/", 1)
    assert not picker.search_active and picker.pane_focus == "preview"
    picker.on_key("CTRL_F", 1)
    assert picker.search_active and picker.pane_focus == "sessions"
    picker.on_key(TextInput("旧い"), 1)
    picker.on_key("TAB", 1)
    assert picker.view_mode == "archived"
    assert not picker.search_active
    output = render(picker, monkeypatch)
    assert "旧い会話" in output and "Ctrl+F Search" in output
    assert "to focus" not in output


def test_search_highlights_across_wrapped_lines():
    from jarv.session_browser_render import highlighted_transcript
    lines = [Text("some earlier content"), Text("Here is the aspect"), Text("ratio setting.")]
    styled, match = highlighted_transcript(lines, "ASPECT RATIO")
    assert match == 1
    assert any(span.start == 12 for span in styled[1].spans)
    assert any(span.start == 0 and span.end == 5 for span in styled[2].spans)
    assert not lines[1].spans  # The cached document remains unmodified.


def wait_for_browser(picker, condition):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        picker._drain_queue()
        picker.on_tick()
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("Browser background work did not complete")


def test_cold_search_and_navigation_never_wait_for_history_io(picker, monkeypatch):
    picker.background = True
    monkeypatch.setattr(picker, "_start_prefetch", lambda: browser.SessionBrowserScreen._start_prefetch(picker))
    entered, release = threading.Event(), threading.Event()
    load = browser.load_history
    main_thread = threading.get_ident()

    def blocked_load(path):
        assert threading.get_ident() != main_thread
        entered.set()
        assert release.wait(5)
        return load(path)

    monkeypatch.setattr(browser, "load_history", blocked_load)
    try:
        picker.on_start()
        assert entered.wait(2)
        render(picker, monkeypatch)
        picker.on_key("DOWN", 1)
        assert picker.selected_sid == picker.rows[1]["sid"]
        picker.on_key("CTRL_F", 1)
        picker.on_key(TextInput("aspect ratio"), 1)
        output = render(picker, monkeypatch)
        assert "aspect ratio" in output
        assert "Searching" in output and "No conversations match" not in output
        release.set()
        wait_for_browser(picker, lambda: len(picker.indexed_paths) == 4)
        assert [r["sid"] for r in picker._visible_rows_list()] == [picker.rows[0]["sid"]]
        render(picker, monkeypatch)
        wait_for_browser(picker, lambda: bool(picker.preview_documents))
        assert "aspect ratio" in render(picker, monkeypatch)
    finally:
        release.set()
        picker.worker.close()


def test_obsolete_preview_is_cancelled_and_cannot_replace_new_selection(picker, monkeypatch):
    picker.background = True
    monkeypatch.setattr(picker, "_prefetch_previews", lambda *_: None)
    entered, release = threading.Event(), threading.Event()
    builds = []

    def delayed_preview(history, width, *, cancelled=None):
        title = history[0]["content"]
        builds.append(title)
        if len(builds) == 1:
            entered.set()
            assert release.wait(5)
            assert cancelled()
        return [Text(title)]

    monkeypatch.setattr(browser, "_history_visual_lines", delayed_preview)
    try:
        render(picker, monkeypatch)
        assert entered.wait(2)
        picker.on_key("DOWN", 1)
        render(picker, monkeypatch)
        picker.on_key("DOWN", 1)
        render(picker, monkeypatch)
        release.set()
        sid = picker.selected_sid
        wait_for_browser(picker, lambda: any(key[0] == sid for key in picker.preview_documents))
        assert builds == ["Fix remote desktop sizing", "Set up Python"]
        assert all(key[0] == sid for key in picker.preview_documents)
        assert "Set up Python" in render(picker, monkeypatch)
    finally:
        release.set()
        picker.worker.close()


def test_search_reuses_matches_and_bounds_excerpts(picker, monkeypatch):
    class CountedText(str):
        checks = 0

        def __contains__(self, query):
            type(self).checks += 1
            return super().__contains__(query)

    sid = picker.selected_sid
    picker.search_text_cache[sid] = "context " * 100000 + "unique needle " + "suffix " * 100000
    picker.search_folded_cache.update({row["sid"]: CountedText(picker.search_text_cache.get(row["sid"], "")) for row in picker.rows})
    picker.search_query = "unique needle"
    assert len(picker._visible_rows_list()) == 1
    checks = CountedText.checks
    for _ in range(5):
        render(picker, monkeypatch)
    # One extra pass is allowed when the initially uncached row title arrives.
    assert CountedText.checks <= checks * 2
    assert all(len(excerpt) < 240 for excerpt in picker.excerpt_cache.values())


@pytest.mark.parametrize("width", [42, 80, 120, 160])
def test_shortcut_hints_take_one_row_with_secondary_actions_on_right(picker, monkeypatch, width):
    render(picker, monkeypatch, width, 24)
    footer = picker._footer_lines(width, picker.rows[0])
    assert len(footer) == 1
    assert "Enter" in footer[0].plain and "Esc" in footer[0].plain
    if width >= 120:
        assert footer[0].plain.endswith("? Help")
        assert "   Space Select" in footer[0].plain


def test_background_preview_reuses_formatting_but_refreshes_changed_history(picker, monkeypatch):
    picker.background = True
    monkeypatch.setattr(picker, "_prefetch_previews", lambda *_: None)
    calls = []
    formatter = browser._history_visual_lines

    def count_formats(*args, **kwargs):
        calls.append(1)
        return formatter(*args, **kwargs)

    monkeypatch.setattr(browser, "_history_visual_lines", count_formats)
    try:
        render(picker, monkeypatch)
        key = picker.preview_request
        wait_for_browser(picker, lambda: key in picker.preview_documents)
        assert len(calls) == 1
        picker.search_query = "aspect ratio"
        render(picker, monkeypatch)
        key = picker.preview_request
        wait_for_browser(picker, lambda: key in picker.preview_documents)
        assert len(calls) == 1
        document, match = picker.preview_documents[key]
        assert match is not None and any(line.spans for line in document)
        path = Path(picker.sessions[picker.selected_sid]["history_file"])
        path.write_text(json.dumps([{"role": "user", "content": "aspect ratio revised"}]), encoding="utf-8")
        picker.preview_checked_at = 0
        picker.on_tick()
        wait_for_browser(picker, lambda: "revised" in picker.search_text_cache[picker.selected_sid])
        assert len(calls) == 2
        assert "revised" in render(picker, monkeypatch)
    finally:
        picker.worker.close()


def test_worker_discards_results_after_browser_closes():
    from jarv.session_browser_work import BrowserWorker

    entered, release = threading.Event(), threading.Event()
    results = []
    worker = BrowserWorker(results.append)

    def blocked(cancelled):
        entered.set()
        assert release.wait(5)
        return "obsolete"

    worker.submit("preview", blocked)
    assert entered.wait(2)
    worker.submit("index", lambda cancelled: "queued")
    worker.close()
    release.set()
    worker.thread.join(timeout=2)
    assert not worker.thread.is_alive()
    assert not results


def test_async_preview_scroll_and_expand_preserve_reading_position(picker, monkeypatch):
    long_conversation(picker)
    picker.background = True
    try:
        render(picker, monkeypatch)
        key = picker.preview_request
        wait_for_browser(picker, lambda: key in picker.preview_documents)
        render(picker, monkeypatch)
        picker.on_key("RIGHT", 1)
        picker.on_key("PAGEDOWN", 1)
        saved = picker.preview_positions[key]
        picker.on_key("p", 1)
        full_key = picker.preview_request
        wait_for_browser(picker, lambda: full_key in picker.preview_documents)
        render(picker, monkeypatch)
        picker.on_key("LEFT", 1)
        assert picker.pane_focus == "preview"
        assert picker.preview_positions[key] == saved
        assert "Preview paragraph" in render(picker, monkeypatch)
    finally:
        picker.worker.close()


def test_indexing_follows_a_session_archived_while_its_file_is_loading(picker, monkeypatch):
    picker.background = True
    sid = picker.selected_sid
    old_path = picker.sessions[sid]["history_file"]
    entered, release = threading.Event(), threading.Event()
    load = browser.load_history

    def delayed_load(path):
        if str(path) == old_path:
            entered.set()
            assert release.wait(5)
        return load(path)

    monkeypatch.setattr(browser, "load_history", delayed_load)
    try:
        picker._request_index(picker.rows[0])
        assert entered.wait(2)
        picker.on_key("a", 1)
        assert picker.sessions[sid]["archived"]
        release.set()
        wait_for_browser(picker, lambda: sid in picker.indexed_paths)
        assert picker.indexed_paths[sid][0] == picker.sessions[sid]["history_file"]
        assert "aspect ratio" in picker.search_folded_cache[sid]
    finally:
        release.set()
        picker.worker.close()


def test_names_are_ready_before_first_frame_without_full_history_indexing(picker, monkeypatch, tmp_path):
    from jarv.session_titles import SessionTitleCache

    picker.background = True
    picker.title_cache = SessionTitleCache(tmp_path / "titles.json",
                                           [meta["history_file"] for meta in picker.sessions.values()])
    monkeypatch.setattr(browser, "terminal_size", lambda **kwargs: (120, 24))
    monkeypatch.setattr(picker, "_start_prefetch", lambda: browser.SessionBrowserScreen._start_prefetch(picker))
    entered, release = threading.Event(), threading.Event()
    load = browser.load_history

    def blocked_load(path):
        entered.set()
        assert release.wait(5)
        return load(path)

    monkeypatch.setattr(browser, "load_history", blocked_load)
    try:
        picker.prepare_titles()
        assert not entered.is_set()
        assert all(row["snippet_loaded"] for row in picker.rows if not row["archived"])
        output = render(picker, monkeypatch)
        assert all(title in output for title in ("Fix remote desktop sizing", "Check memory usage", "Set up Python"))
        picker.on_start()
        assert entered.wait(2)
        release.set()
        wait_for_browser(picker, lambda: all(row["snippet_loaded"] for row in picker.rows))
        cache = SessionTitleCache(tmp_path / "titles.json", [meta["history_file"] for meta in picker.sessions.values()])
        assert all(cache.get(meta["history_file"]) is not None for meta in picker.sessions.values())
    finally:
        release.set()
        picker.worker.close()


def test_title_preparation_reuses_cache_and_keeps_renamed_titles(picker, monkeypatch, tmp_path):
    from jarv.session_titles import SessionTitleCache

    paths = [meta["history_file"] for meta in picker.sessions.values()]
    cache_path = tmp_path / "titles.json"
    cache = SessionTitleCache(cache_path, paths)
    for path in paths:
        cache.read(path)
    cache.save()
    picker.title_cache = SessionTitleCache(cache_path, paths)
    picker.sessions[picker.selected_sid]["title"] = "My custom title"

    def unexpected_read(*args, **kwargs):
        pytest.fail("Opening the menu must not reread cached history files")

    monkeypatch.setattr(Path, "open", unexpected_read)
    monkeypatch.setattr(browser, "terminal_size", lambda **kwargs: (120, 24))
    picker.prepare_titles()
    assert all(row["snippet_loaded"] for row in picker.rows)
    assert picker._title(picker.rows[0]) == "My custom title"
    assert picker._title(picker.rows[1]) == "Check memory usage"


def test_cold_title_reads_are_bounded_to_the_initial_viewport(picker, monkeypatch):
    for index in range(100):
        sid = f"extra-{index}"
        picker.rows.append(dict(sid=sid, archived=False, is_current=False, snippet_loaded=False))
        picker.sessions[sid] = {"history_file": sid}
    picker.selected_sid = "extra-70"
    reads = []
    picker.title_cache = SimpleNamespace(get=lambda path: None,
                                         read=lambda path: reads.append(path) or "A conversation")
    monkeypatch.setattr(browser, "terminal_size", lambda **kwargs: (120, 24))
    picker.prepare_titles()
    assert "extra-70" in reads
    assert len(reads) <= 25
    assert "extra-0" not in reads and "extra-99" not in reads


def test_large_first_prompt_is_cached_after_background_indexing(picker, tmp_path):
    from jarv.session_titles import SessionTitleCache

    path = picker.sessions[picker.selected_sid]["history_file"]
    Path(path).write_text(json.dumps([{"role": "user", "content": "Long first prompt " + "x" * 300000}]), encoding="utf-8")
    cache_path = tmp_path / "titles.json"
    picker.title_cache = SessionTitleCache(cache_path, [path])
    assert picker.title_cache.read(path) is None
    record = picker._read_document(path)
    reopened = SessionTitleCache(cache_path, [path])
    assert reopened.get(path) == record[4]
    assert reopened.get(path).startswith("Long first prompt")


def test_held_arrows_are_delivered_one_step_per_frame_in_order(picker, monkeypatch):
    from jarv import command_input

    keys = ["DOWN"] * 20 + ["UP"] * 20 + ["ENTER"]
    monkeypatch.setattr(command_input, "_PENDING_KEYS", deque(keys))
    observed = [picker._read_browser_key() for _ in keys]
    assert observed == [(key, 1) for key in keys]
    assert not command_input._PENDING_KEYS


def test_held_navigation_uses_prewarmed_previews_without_waiting_for_io(picker, monkeypatch):
    picker.background = True
    entered, release = threading.Event(), threading.Event()
    read_document = picker._read_document

    def blocked_read(path):
        entered.set()
        assert release.wait(5)
        return read_document(path)

    try:
        render(picker, monkeypatch)
        wait_for_browser(picker, lambda: len(picker.preview_documents) == 3)
        monkeypatch.setattr(picker, "_read_document", blocked_read)
        picker.preview_checked_at = 0
        picker.on_tick()
        assert entered.wait(2)
        for key in ["DOWN", "DOWN", "UP", "UP"] * 3:
            picker.on_key(key, 1)
            output = render(picker, monkeypatch)
            assert "Loading preview…" not in output
            expected = {picker.rows[0]["sid"]: "Match the host",
                        picker.rows[1]["sid"]: "user: Check memory usage",
                        picker.rows[2]["sid"]: "user: Set up Python"}[picker.selected_sid]
            assert expected in output
            layout = picker._list_layout(picker._visible_rows_list())
            detail = picker._detail_lines(layout["current"], layout["right_width"], layout["body_height"] - 1)
            assert detail[0].plain == picker._title(layout["current"])
        assert entered.wait(2)
    finally:
        release.set()
        picker.worker.close()
        picker.worker.thread.join(timeout=2)


def test_cold_preview_starts_on_scroll_without_a_settle_delay(picker, monkeypatch):
    picker.background = True
    monkeypatch.setattr(picker, "_prefetch_previews", lambda *_: None)
    entered, release = threading.Event(), threading.Event()
    read_document = picker._read_document

    def blocked_read(path):
        entered.set()
        assert release.wait(5)
        return read_document(path)

    try:
        render(picker, monkeypatch)
        wait_for_browser(picker, lambda: picker.preview_request in picker.preview_documents)
        monkeypatch.setattr(picker, "_read_document", blocked_read)
        picker.on_key("DOWN", 1)
        output = render(picker, monkeypatch)
        assert entered.wait(2)  # No tick, timeout, or pane switch is needed.
        assert "Loading preview…" in output and "Match the host" not in output
        request = picker.preview_request
        version = picker.worker.versions["preview"]
        assert request[0] == picker.selected_sid
        picker.on_key("RIGHT", 1)
        render(picker, monkeypatch)
        assert picker.worker.versions["preview"] == version
        assert picker.pane_focus == "preview"
        release.set()
        wait_for_browser(picker, lambda: request in picker.preview_documents)
        assert "user: Check memory usage" in render(picker, monkeypatch)
    finally:
        release.set()
        picker.worker.close()
        picker.worker.thread.join(timeout=2)


@pytest.mark.parametrize("index", [0, 1])
def test_prewarmed_preview_rejects_a_changed_history_path(picker, monkeypatch, tmp_path, index):
    picker.background = True
    entered, release = threading.Event(), threading.Event()
    read_document = picker._read_document
    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps([{"role": "user", "content": "Replacement conversation"}]), encoding="utf-8")

    def blocked_read(path):
        if path == str(replacement):
            entered.set()
            assert release.wait(5)
        return read_document(path)

    try:
        render(picker, monkeypatch)
        wait_for_browser(picker, lambda: len(picker.preview_documents) == 3)
        picker.sessions[picker.rows[index]["sid"]]["history_file"] = str(replacement)
        monkeypatch.setattr(picker, "_read_document", blocked_read)
        if index:
            picker.on_key("DOWN", 1)
        output = render(picker, monkeypatch)
        assert entered.wait(2)
        assert "Loading preview…" in output
        assert "user: Check memory usage" not in output and "Match the host" not in output
        release.set()
        wait_for_browser(picker, lambda: "Replacement conversation" in picker.search_text_cache.get(picker.selected_sid, ""))
        assert "user: Replacement conversation" in render(picker, monkeypatch)
    finally:
        release.set()
        picker.worker.close()
        picker.worker.thread.join(timeout=2)


def test_preview_prefetch_bounds_memory_and_does_not_rebuild_evicted_neighbours(picker, monkeypatch):
    picker.background = True
    monkeypatch.setattr(picker, "PREVIEW_CACHE_LINES", 16)
    for index in range(30):
        row = dict(picker.rows[1], sid=f"extra-{index}")
        picker.rows.append(row)
        picker.row_by_sid[row["sid"]] = row
        picker.sessions[row["sid"]] = dict(picker.sessions[picker.rows[1]["sid"]])
    builds = []

    def format_preview(history, width, *, cancelled=None):
        builds.append(1)
        return [Text(f"line {n}") for n in range(10)]

    monkeypatch.setattr(browser, "_history_visual_lines", format_preview)
    try:
        render(picker, monkeypatch)
        wait_for_browser(picker, lambda: picker.preview_request in picker.preview_documents and not picker.preview_prefetch)
        assert len(builds) == 11  # Selected session and ten ahead, not all 33.
        assert sum(len(value[1]) for value in picker.preview_cache.values()) <= 16
        assert list(picker.preview_documents) == [picker.preview_request]
        for _ in range(10):
            render(picker, monkeypatch)
            picker._drain_queue()
        assert len(builds) == 11
        assert not picker.preview_prefetch
        picker.on_key("DOWN", 1)
        render(picker, monkeypatch)
        wait_for_browser(picker, lambda: picker.preview_request in picker.preview_documents and not picker.preview_prefetch)
        assert "Loading preview…" not in render(picker, monkeypatch)
        assert len(builds) < 16
    finally:
        picker.worker.close()
        picker.worker.thread.join(timeout=2)


@pytest.mark.parametrize("width,height", [(42, 12), (120, 24)])
def test_archive_notification_survives_navigation_expires_without_moving_layout(picker, monkeypatch, width, height):
    clock = [100.0]
    monkeypatch.setattr(browser, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    render(picker, monkeypatch, width, height)
    original_height = picker._list_layout(picker._visible_rows_list())["body_height"]
    picker.on_key("a", 1)
    message = picker.flash
    for key in ("DOWN", "TAB"):
        picker.on_key(key, 1)
        render(picker, monkeypatch, width, height)
        assert picker.flash == message
        assert picker._list_layout(picker._visible_rows_list())["body_height"] == original_height
    clock[0] += 4.1
    picker.on_tick()
    output = render(picker, monkeypatch, width, height)
    assert picker.flash is None
    assert "u Undo archive" in output
    assert "✓" not in output
    assert picker._list_layout(picker._visible_rows_list())["body_height"] == original_height


def test_undo_feedback_does_not_reannounce_older_archive(picker, monkeypatch):
    render(picker, monkeypatch)
    picker.on_key("a", 1)
    picker.on_key("a", 1)
    picker.on_key("u", 1)
    assert "restored Check memory usage" in render(picker, monkeypatch)
    picker.on_key("DOWN", 1)
    assert "restored Check memory usage" in render(picker, monkeypatch)
    assert "u Undo archive" in render(picker, monkeypatch)
    assert "✓ archived" not in render(picker, monkeypatch)


def test_archive_removes_only_target_rows_and_keeps_nearest_survivor(picker, monkeypatch):
    render(picker, monkeypatch)
    picker.on_key(" ", 1)
    picker.on_key("DOWN", 1)
    picker.on_key("DOWN", 1)
    picker.on_key(" ", 1)
    picker.on_key("a", 1)
    assert picker.selected_sid == picker.rows[1]["sid"]
    assert [r["sid"] for r in picker._visible_rows_list()] == [picker.rows[1]["sid"]]
    assert not picker.marked_sids and not picker.ghost_sids
    output = render(picker, monkeypatch)
    assert "[Active 1]" in output and "Archived 3" in output and "1 of 1" in output
    assert "archived 2 sessions" in output
    picker.on_key("u", 1)
    assert len(picker._visible_rows_list()) == 3
    assert picker.selected_sid == picker.rows[0]["sid"]
    assert picker.marked_sids == {picker.rows[0]["sid"], picker.rows[2]["sid"]}


def test_archiving_last_active_rows_shows_empty_active_view_and_can_undo(picker, monkeypatch):
    render(picker, monkeypatch)
    picker.on_key("SHIFT_DOWN", 2)
    picker.on_key("a", 1)
    output = render(picker, monkeypatch)
    assert "No active conversations." in output and "Archived 4" in output
    assert picker.selected_sid is None and not picker.marked_sids
    picker.on_key("u", 1)
    assert len(picker._visible_rows_list()) == 3
    assert picker.selected_sid == picker.rows[0]["sid"]


def test_restore_leaves_archived_view_and_undo_returns_to_matching_view(picker, monkeypatch):
    picker.on_key("TAB", 1)
    sid = picker.selected_sid
    picker.on_key("a", 1)
    assert not picker._visible_rows_list() and picker.selected_sid is None
    assert "No archived conversations." in render(picker, monkeypatch)
    picker.on_key("TAB", 1)
    picker.on_key("TAB", 1)
    assert picker.view_mode == "active"
    picker.on_key("u", 1)
    assert picker.view_mode == "archived"
    assert picker.selected_sid == sid
    assert picker.rows[3]["archived"]


def test_missing_archive_stays_archived_and_reports_failure(picker, monkeypatch):
    picker.on_key("TAB", 1)
    sid = picker.selected_sid
    Path(picker.sessions[sid]["history_file"]).unlink()
    before = history.SESSIONS_FILE.read_bytes()
    picker.on_key("a", 1)
    assert history.SESSIONS_FILE.read_bytes() == before
    assert picker.sessions[sid]["archived"] and picker.rows[3]["archived"]
    assert picker.selected_sid == sid and not picker.undo_actions
    output = render(picker, monkeypatch)
    assert "Couldn't restore" in output and "files are missing" in output
    assert "marked active" not in output and "✓ restored" not in output


def test_narrow_restore_failure_prioritizes_reason_over_older_undo_hint(picker, monkeypatch):
    picker.on_key("a", 1)
    picker.on_key("TAB", 1)
    Path(picker.sessions[picker.selected_sid]["history_file"]).unlink()
    picker.on_key("a", 1)
    assert "Couldn't restore: files are missing" in render(picker, monkeypatch, 42, 12)


def test_partial_archive_counts_only_successes_and_keeps_failed_selection(picker, monkeypatch):
    render(picker, monkeypatch)
    Path(picker.sessions[picker.rows[1]["sid"]]["history_file"]).unlink()
    picker.on_key("SHIFT_DOWN", 1)
    picker.on_key("a", 1)
    assert picker.undo_actions[-1]["sids"] == [picker.rows[0]["sid"]]
    assert picker.marked_sids == {picker.rows[1]["sid"]}
    assert "Archived 1; couldn't archive 1" in render(picker, monkeypatch)
    picker.on_key("u", 1)
    assert not picker.rows[0]["archived"]
    assert not picker.undo_actions


def test_archive_save_failure_rolls_back_files_metadata_and_notification(picker, monkeypatch):
    sid = picker.selected_sid
    path = Path(picker.sessions[sid]["history_file"])
    original = path.read_bytes()
    metadata = history.SESSIONS_FILE.read_bytes()
    monkeypatch.setattr(browser, "save_sessions", lambda data: (_ for _ in ()).throw(OSError("disk full")))
    picker.on_key("a", 1)
    assert path.read_bytes() == original
    assert history.SESSIONS_FILE.read_bytes() == metadata
    assert not picker.rows[0]["archived"] and not picker.sessions[sid]["archived"]
    assert not picker.undo_actions
    assert picker.terminals["terminal"] == sid
    output = render(picker, monkeypatch)
    assert "Couldn't archive" in output and "disk full" in output


def test_failed_undo_keeps_archive_and_can_be_retried(picker, monkeypatch):
    sid = picker.selected_sid
    picker.on_key("a", 1)
    archived = Path(picker.sessions[sid]["history_file"])
    restore = browser.unarchive_session_files
    monkeypatch.setattr(browser, "unarchive_session_files", lambda *args: None)
    picker.on_key("u", 1)
    assert "Couldn't undo" in render(picker, monkeypatch)
    assert picker.undo_actions[-1]["sids"] == [sid]
    assert picker.rows[0]["archived"] and archived.exists()
    assert "terminal" not in picker.terminals
    monkeypatch.setattr(browser, "unarchive_session_files", restore)
    picker.on_key("u", 1)
    assert not picker.undo_actions and not picker.rows[0]["archived"]
    assert picker.terminals["terminal"] == sid
    assert picker.rows[0]["is_current"]


def test_partial_undo_retries_only_failed_rows(picker, monkeypatch):
    picker.on_key("SHIFT_DOWN", 1)
    picker.on_key("a", 1)
    first, second = [r["sid"] for r in picker.rows[:2]]
    restore = browser.unarchive_session_files
    monkeypatch.setattr(browser, "unarchive_session_files", lambda path, sid: None if sid == second else restore(path, sid))
    picker.on_key("u", 1)
    assert "Undid 1; couldn't undo 1" in render(picker, monkeypatch)
    assert not picker.rows[0]["archived"] and picker.rows[1]["archived"]
    assert picker.undo_actions[-1]["sids"] == [second]
    monkeypatch.setattr(browser, "unarchive_session_files", restore)
    picker.on_key("u", 1)
    assert not picker.undo_actions and not picker.rows[1]["archived"]
    assert picker.terminals["terminal"] == first


def test_undo_archive_does_not_replace_new_terminal_binding(picker):
    picker.on_key("a", 1)
    other = picker.rows[1]["sid"]
    newer = history.load_sessions()
    newer["terminals"]["terminal"] = other
    history.save_sessions(newer)
    picker.on_key("u", 1)
    assert history.load_sessions()["terminals"]["terminal"] == other
    assert not picker.rows[0]["is_current"]


def test_failed_restore_from_fullscreen_preview_has_visible_notification(picker, monkeypatch):
    picker.on_key("TAB", 1)
    render(picker, monkeypatch)
    picker.on_key("p", 1)
    Path(picker.sessions[picker.selected_sid]["history_file"]).unlink()
    picker.on_key("ENTER", 1)
    assert "Archived session files are missing" in render(picker, monkeypatch)
    assert picker.preview_sid is not None and picker.loaded_row is None


def test_delete_notification_keeps_title_after_row_is_removed(picker, monkeypatch):
    render(picker, monkeypatch)
    picker.on_key("d", 1)
    picker.on_key("d", 1)
    assert "deleted Fix remote desktop sizing" in render(picker, monkeypatch)
