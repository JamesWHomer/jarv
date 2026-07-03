import json
from types import SimpleNamespace

from conftest import make_console

import jarv.session_tree as session_tree
import jarv.tree_browser as tree_browser
import jarv.tree_command as tree_command
from jarv.tree_browser import TreeOutcome


class TtyStdin:
    def isatty(self):
        return True


def _setup(tmp_path, monkeypatch, history_items, *, force_terminal=False):
    history_file = tmp_path / "history-abc.json"
    history_file.write_text(json.dumps(history_items), encoding="utf-8")
    ctx = SimpleNamespace(history_file=history_file)
    monkeypatch.setattr(tree_command, "prepare_session_context", lambda: ctx)
    console, output = make_console(force_terminal=force_terminal)
    monkeypatch.setattr(tree_command, "console", console)
    return ctx, output


def test_tree_plain_renders_prompts_when_not_a_tty(tmp_path, monkeypatch):
    _, output = _setup(
        tmp_path,
        monkeypatch,
        [
            {"role": "user", "content": "first prompt"},
            {"role": "assistant", "content": "reply"},
        ],
    )

    tree_command.cmd_tree()

    rendered = output.getvalue()
    assert "first prompt" in rendered
    assert "Run /tree in an interactive terminal" in rendered


def test_tree_plain_handles_empty_history(tmp_path, monkeypatch):
    _, output = _setup(tmp_path, monkeypatch, [])

    tree_command.cmd_tree()

    assert "No prompts yet" in output.getvalue()


def _run_interactive(tmp_path, monkeypatch, outcome, *, checkout_result=True):
    ctx, output = _setup(
        tmp_path,
        monkeypatch,
        [{"role": "user", "content": "first prompt"}],
        force_terminal=True,
    )
    monkeypatch.setattr(tree_command.sys, "stdin", TtyStdin())
    monkeypatch.setattr("jarv.commands.load_config", lambda: {})
    monkeypatch.setattr(tree_browser, "run_tree_screen", lambda _ctx, _config: outcome)
    checkouts = []

    def fake_checkout(history_file, *, leaf_id):
        checkouts.append((history_file, leaf_id))
        return checkout_result

    monkeypatch.setattr(session_tree, "checkout", fake_checkout)
    tree_command.cmd_tree()
    return ctx, checkouts, output.getvalue()


def test_tree_open_checks_out_selected_leaf(tmp_path, monkeypatch):
    ctx, checkouts, rendered = _run_interactive(
        tmp_path, monkeypatch, TreeOutcome("open", "leaf-1", None)
    )

    assert checkouts == [(ctx.history_file, "leaf-1")]
    assert "Resumed" in rendered


def test_tree_cancel_leaves_history_untouched(tmp_path, monkeypatch):
    _, checkouts, rendered = _run_interactive(
        tmp_path, monkeypatch, TreeOutcome("cancel", None, None)
    )

    assert checkouts == []
    assert "Closed tree" in rendered


def test_tree_edit_prints_prefill(tmp_path, monkeypatch):
    _, checkouts, rendered = _run_interactive(
        tmp_path, monkeypatch, TreeOutcome("edit", "leaf-1", "revised prompt")
    )

    assert len(checkouts) == 1
    assert "Ready to edit" in rendered
    assert "revised prompt" in rendered
