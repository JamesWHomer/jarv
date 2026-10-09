"""Tree browsing must preserve prompt identity through folds, search and resize."""

import io
from collections import deque
from copy import deepcopy

import pytest
from rich.cells import cell_len
from rich.console import Console

from conftest import FakeLive, neutralize_tui_modes
from jarv import tree_browser, tui_panel
from jarv.command_input import TextInput
from jarv.history import branches_file_for, load_history, save_branches, save_history
from jarv.session_tree import build_tree, load_session_tree
from jarv.storage import StorageError


def frame(fid, prompt, reply="A helpful reply."):
    return [{"role": "user", "id": fid, "content": prompt}, {"role": "assistant", "content": reply}]


def conversation():
    active = [
        ("a", "Tell me about universities"),
        ("b", "How do their rankings compare?"),
        ("c", "Which is better for computer science?"),
        ("d", "I prefer the city campus"),
        ("e", "That sounds good"),
        ("f", "Tell me about the interview process"),
        ("g", "I have three interviews next week, 45 minutes each."),
    ]
    history = [item for fid, prompt in active for item in frame(fid, prompt)]
    history[-1]["content"] = "**Congratulations on reaching the final stage.**\n\nYou have time for targeted preparation."
    alternatives = [
        ("f", "h", "Thanks"), ("e", "i", "How about another campus?"),
        ("i", "j", "Keep going"), ("j", "k", "What about AI?"),
        ("k", "l", "How about entry requirements?"), ("l", "m", "Which course?"),
        ("m", "n", "One more question"), ("n", "o", "I meant admission scores"),
        ("o", "p", "Lowest admission score?"), ("j", "q", "Other questions"),
        ("q", "r", "What interview?"), ("r", "s", "Understood"),
    ]
    branches = [{"parent_frame_id": parent, "items": frame(fid, prompt)} for parent, fid, prompt in alternatives]
    return history, branches


@pytest.fixture
def screen(monkeypatch):
    size = [120, 24]
    output = io.StringIO()
    console = Console(file=output, width=size[0], height=size[1], color_system=None, legacy_windows=False)
    monkeypatch.setattr(tree_browser, "console", console)
    monkeypatch.setattr(tree_browser, "terminal_size", lambda **kwargs: tuple(size))
    tui_panel.configure_menu_border(False)
    app = tree_browser.TreeBrowserScreen(model=build_tree(*conversation()))
    app.test_size = size
    return app


def render(app):
    output = io.StringIO()
    console = Console(file=output, width=app.test_size[0], height=app.test_size[1], color_system=None, legacy_windows=False)
    console.print(app.render())
    return output.getvalue().splitlines()


def select(app, fid):
    app._select_node(next(i for i, node in enumerate(app.nodes) if node.frame_id == fid))


def selected_id(app):
    return app.nodes[app.selected].frame_id


def search(app, query):
    app.on_key("CTRL_F", 1)
    app.on_key(TextInput(query), 1)
    app.on_key("ENTER", 1)


def test_wide_view_formats_the_complete_selected_exchange(screen):
    text = "\n".join(render(screen))
    assert "19 prompts · 7 on current path" in text
    assert "4 earlier prompts" in text
    assert "TREE" in text and "PREVIEW" in text
    assert "● current" in text and "⑂" not in text
    assert "Congratulations on reaching the final" in text
    assert "**" not in text
    assert "tab preview" in text and "ctrl+f search" in text
    preview = " ".join(line.rsplit(" │ ", 1)[1].strip() for line in text.splitlines() if " │ " in line)
    assert "45 minutes each." in preview


@pytest.mark.parametrize("border", [False, True])
@pytest.mark.parametrize("size", [(20, 4), (36, 6), (42, 12), (80, 24), (119, 24), (120, 24), (160, 32), (250, 40)])
def test_render_stays_inside_terminal_in_all_modes(screen, border, size):
    screen.test_size[:] = size
    tui_panel.configure_menu_border(border)
    for key in (None, "p", "?", "ESC", "ESC", "CTRL_F"):
        if key:
            screen.on_key(key, 1)
        lines = render(screen)
        assert len(lines) == size[1]
        assert all(cell_len(line) == size[0] for line in lines)
        footer = lines[-2] if border else lines[-1]
        assert "esc" in footer.lower()


def test_current_marker_does_not_follow_the_cursor(screen):
    screen.on_key("DOWN", 1)
    assert selected_id(screen) == "h"
    tree = [line.plain for line in screen._tree_lines(75, 20)]
    current = next(line for line in tree if "● current" in line)
    selected = next(line for line in tree if line.startswith("› ") and "Thanks" in line)
    assert not current.startswith("› ")
    assert "● current" not in selected
    row = screen._row(screen.nodes[screen.selected], True, 75)
    assert str(row.style) == "on #17333b"


def test_folded_summary_is_not_a_prompt_action_target(screen):
    before = deepcopy([(node.frame_id, node.items) for node in screen.nodes])
    screen.on_key("HOME", 1)
    assert screen.selected_run
    assert len(screen._selected_row().run) == 4
    for key in ("f", "e", "d", "d"):
        screen.on_key(key, 1)
    assert screen.outcome.action == "cancel"
    assert screen.arm_delete_id is None
    screen.on_key("ENTER", 1)
    assert selected_id(screen) == "a" and not screen.selected_run
    assert len(screen._rows()) == 19
    screen.on_key(" ", 1)
    assert screen.selected_run and len(screen._rows()) == 16
    assert before == [(node.frame_id, node.items) for node in screen.nodes]


def test_parent_navigation_reveals_hidden_parent(screen):
    screen.on_key("LEFT", 1)
    assert selected_id(screen) == "f"
    assert screen.view.folded_runs
    screen.on_key("LEFT", 1)
    assert selected_id(screen) == "e"
    assert not screen.view.folded_runs
    assert any(screen.nodes[row.index].frame_id == "d" for row in screen._rows())


def test_branch_fold_counts_descendants_and_right_reveals_child(screen):
    select(screen, "i")
    screen.on_key(" ", 1)
    assert screen._selected_row().hidden == 10
    assert "10 hidden" in screen._row(screen.nodes[screen.selected], True, 70).plain
    assert all(screen.nodes[row.index].frame_id != "p" for row in screen._rows())
    assert any(screen.nodes[row.index].is_active_leaf for row in screen._rows())
    screen.on_key("RIGHT", 1)
    assert selected_id(screen) == "j"
    assert not screen.view.folded_branches


def test_enter_on_folded_branch_only_jumps_until_confirmed(screen):
    select(screen, "i")
    screen.on_key(" ", 1)
    screen.on_key("ENTER", 1)
    assert selected_id(screen) == "p"
    assert screen.outcome.action == "cancel"
    assert screen._selected_row() is not None
    screen.on_key("ENTER", 1)
    assert screen.outcome == tree_browser.TreeOutcome("open", "p", None)


def test_edit_remains_leaf_only_and_fork_targets_the_selected_parent(screen):
    select(screen, "f")
    screen.on_key("e", 1)
    assert screen.outcome.action == "cancel"
    screen.on_key("f", 1)
    assert screen.outcome == tree_browser.TreeOutcome("fork", "f", None)


def test_search_finds_hidden_prompt_and_retains_ancestors(screen):
    select(screen, "i")
    screen.on_key(" ", 1)
    search(screen, "lowest admission")
    assert selected_id(screen) == "p"
    visible = [screen.nodes[row.index].frame_id for row in screen._rows()]
    assert visible == list("abcdeijklmnop")
    assert len(screen.search_matches) == 1
    assert not any(row.run or row.hidden for row in screen._rows())
    screen.on_key("ESC", 1)
    assert not screen.search_query
    assert selected_id(screen) == "p" and screen._selected_row() is not None


def test_search_indexes_all_assistant_messages_not_just_preview_snippet(screen):
    node = next(node for node in screen.nodes if node.frame_id == "r")
    node.items += [{"role": "assistant", "content": "A unique second response."}]
    search(screen, "unique second")
    assert selected_id(screen) == "r"
    document, match = screen._preview_document(screen.selected, 45)
    assert match is not None
    assert "unique second" in document[match].plain
    assert document[match].spans


def test_no_search_results_cannot_act_on_hidden_selection(screen):
    search(screen, "no such prompt")
    assert screen._rows() == []
    assert "No matching prompts or replies." in "\n".join(render(screen))
    for key in ("ENTER", "f", "e", "d", "d", "p", "TAB"):
        screen.on_key(key, 1)
    assert screen.outcome.action == "cancel"
    assert screen.arm_delete_id is None
    screen.on_key("ESC", 1)
    assert screen._selected_row() is not None


@pytest.mark.parametrize("text", ["q", "f", "d", "ENTER", "DELETE", "LEFT", "first\nsecond"])
def test_search_paste_is_literal_and_esc_clears_it(screen, text):
    screen.on_key("CTRL_F", 1)
    assert screen.text_mode
    screen.on_key(TextInput(text), 1)
    assert screen.search_active
    assert screen.search_query == text.replace("\n", " ")
    assert screen.outcome.action == "cancel"
    screen.on_key("ESC", 1)
    assert not screen.text_mode and not screen.search_query


def test_structural_navigation_can_leave_filtered_results(screen):
    search(screen, "rankings")
    assert selected_id(screen) == "b"
    screen.on_key("RIGHT", 1)
    assert selected_id(screen) == "c"
    assert not screen.search_query and screen._selected_row() is not None


def test_preview_focus_scroll_and_resize_preserve_selection_and_reading_position(screen):
    node = screen.nodes[screen.selected]
    node.items[-1]["content"] = "\n\n".join(f"Paragraph {i}: a useful explanation with enough text to wrap." for i in range(60))
    render(screen)
    screen.on_key("TAB", 1)
    screen.on_key("DOWN", 12)
    assert screen.pane_focus == "preview" and selected_id(screen) == "g"
    _, old_start = screen.preview_positions[("g", "")]
    assert old_start > 0
    layout = screen._layout()
    lines, _, _ = screen._preview_window(screen.selected, layout["right"], layout["body"] - 2)
    anchor = next(line.plain for line in lines if line.plain.startswith("Paragraph"))
    screen.on_key("p", 1)
    expanded = "\n".join(render(screen))
    assert screen.full_preview and anchor[:12] in expanded
    screen.on_key("p", 1)
    assert not screen.full_preview and screen.pane_focus == "preview"
    screen.on_key("TAB", 1)
    screen.on_key("DOWN", 1)
    screen.on_key("UP", 1)
    render(screen)
    assert selected_id(screen) == "g"
    assert screen.preview_positions[("g", "")][1] > 0
    screen.test_size[:] = (80, 24)
    screen.on_resize((80, 24))
    assert screen.pane_focus == "tree"
    screen.on_key("TAB", 1)
    assert screen.full_preview
    assert anchor[:12] in "\n".join(render(screen))
    screen.on_key("ESC", 1)
    assert not screen.full_preview and selected_id(screen) == "g"


def test_preview_does_not_handle_prompt_mutation_shortcuts(screen):
    select(screen, "h")
    screen.on_key("TAB", 1)
    for key in ("f", "e", "d", "d", "ENTER"):
        screen.on_key(key, 1)
    assert screen.outcome.action == "cancel" and screen.arm_delete_id is None
    screen.on_key("p", 1)
    for key in ("f", "e", "d", "d", "ENTER"):
        screen.on_key(key, 1)
    assert screen.outcome.action == "cancel" and screen.arm_delete_id is None


def test_narrow_preview_starts_at_reply_even_after_a_long_prompt(screen):
    screen.test_size[:] = (80, 24)
    screen.nodes[screen.selected].items[0]["content"] = "A very long prompt. " * 200
    text = "\n".join(render(screen))
    assert "Congratulations on reaching the final stage." in text
    screen.on_key("p", 1)
    assert "A very long prompt." in "\n".join(render(screen))


def test_preview_cache_avoids_reformatting_idle_frames(screen, monkeypatch):
    original = tree_browser._history_visual_lines
    calls = []

    def record(items, width):
        calls.append((items[0]["id"], width))
        return original(items, width)

    monkeypatch.setattr(tree_browser, "_history_visual_lines", record)
    render(screen)
    render(screen)
    screen.on_key("TAB", 1)
    render(screen)
    assert len(calls) == 1


def test_help_scrolls_and_returns_without_moving_selection(screen):
    screen.test_size[:] = (42, 12)
    screen.on_key("?", 1)
    first = render(screen)
    screen.on_key("END", 1)
    last = render(screen)
    assert first != last and "close" in " ".join(last)
    screen.on_key("ESC", 1)
    assert not screen.help_open and selected_id(screen) == "g"


def test_delete_folded_subtree_requires_confirmation_and_preserves_live_history(screen, tmp_path):
    history, branches = conversation()
    path = tmp_path / "history-tree.json"
    save_history(history, path)
    save_branches(branches, branches_file_for(path))
    screen.history_file = path
    select(screen, "i")
    screen.on_key(" ", 1)
    screen.on_key("d", 1)
    assert "11 prompts" in screen.flash[0]
    assert len(load_session_tree(path).nodes) == 19
    screen.on_key("ESC", 1)
    assert screen.arm_delete_id is None
    screen.on_key("d", 1)
    screen.on_key("d", 1)
    assert len(screen.nodes) == 8
    assert selected_id(screen) == "e"
    assert load_history(path) == history
    assert "Branch deleted" in screen.flash[0]


def test_failed_delete_stays_in_browser_with_unchanged_tree(screen, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise StorageError("write failed")

    monkeypatch.setattr(tree_browser, "delete_subtree", fail)
    screen.history_file = tmp_path / "history-tree.json"
    select(screen, "i")
    screen.on_key("d", 1)
    screen.on_key("d", 1)
    assert len(screen.nodes) == 19 and selected_id(screen) == "i"
    assert "write failed" in screen.flash[0]
    assert screen.arm_delete_id is None


@pytest.mark.parametrize("size", [(20, 4), (36, 6)])
def test_delete_confirmation_is_visible_in_short_bordered_terminals(screen, size):
    screen.test_size[:] = size
    tui_panel.configure_menu_border(True)
    select(screen, "h")
    screen.on_key("d", 1)
    lines = render(screen)
    assert len(lines) == size[1]
    assert "d " in lines[-2] and "esc" in lines[-2]


def test_pasted_text_cancels_pending_deletion(screen):
    select(screen, "h")
    screen.on_key("d", 1)
    assert screen.arm_delete_id == "h"
    screen.on_key(TextInput("d"), 1)
    assert screen.arm_delete_id is None and screen.flash is None
    assert len(screen.nodes) == 19


def test_live_path_cannot_be_folded_away_or_deleted(screen):
    select(screen, "f")
    screen.on_key(" ", 1)
    assert "current path stays visible" in screen.flash[0]
    screen.on_key("d", 1)
    screen.on_key("d", 1)
    assert len(screen.nodes) == 19 and screen.arm_delete_id is None
    assert any(screen.nodes[row.index].is_active_leaf for row in screen._rows())


def test_unicode_prompts_and_missing_response_render_without_overflow(screen):
    screen.model = build_tree([{"role": "user", "id": "unicode", "content": "界 é 🙂 " * 30}], [])
    screen.selected = 0
    screen._reset_view()
    screen.test_size[:] = (42, 24)
    assert "(no response yet)" in "\n".join(render(screen))
    assert all(cell_len(line) == 42 for line in render(screen))


def test_real_event_loop_browses_search_and_resumes_the_matching_leaf(screen, monkeypatch):
    neutralize_tui_modes(monkeypatch)
    keys = deque(["HOME", "ENTER", "CTRL_F", TextInput("lowest admission"), "ENTER", "p", "ESC", "ENTER"])
    screen._read_key_fn = lambda: (keys.popleft(), 1)
    screen._key_available_fn = lambda: bool(keys)
    screen._terminal_size_fn = lambda **kwargs: tuple(screen.test_size)
    screen._live_factory = lambda get_renderable, console: FakeLive(get_renderable, console)
    screen.run()
    assert screen.outcome == tree_browser.TreeOutcome("open", "p", None)
    assert not keys
