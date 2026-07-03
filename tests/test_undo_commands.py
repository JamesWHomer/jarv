import json
from types import SimpleNamespace

from conftest import make_console

import jarv.undo_commands as undo_commands
from jarv.history import load_history, load_redo_stack, redo_file_for

_EXCHANGES = [
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "first answer"},
    {"role": "user", "content": "second question"},
    {"role": "assistant", "content": "second answer"},
]


def _setup(tmp_path, monkeypatch, history_items):
    history_file = tmp_path / "history-abc.json"
    history_file.write_text(json.dumps(history_items), encoding="utf-8")
    monkeypatch.setattr(
        undo_commands,
        "prepare_session_context",
        lambda: SimpleNamespace(history_file=history_file),
    )
    console, output = make_console()
    monkeypatch.setattr(undo_commands, "console", console)
    return history_file, output


def test_undo_moves_last_exchange_to_redo_stack(tmp_path, monkeypatch):
    history_file, output = _setup(tmp_path, monkeypatch, _EXCHANGES)

    undo_commands.cmd_undo([])

    assert load_history(history_file) == _EXCHANGES[:2]
    stack = load_redo_stack(redo_file_for(history_file))
    assert stack == [_EXCHANGES[2:]]
    assert "Unsent" in output.getvalue()
    assert "second question" in output.getvalue()


def test_undo_on_empty_history_prints_nothing_to_undo(tmp_path, monkeypatch):
    history_file, output = _setup(tmp_path, monkeypatch, [])

    undo_commands.cmd_undo([])

    assert "Nothing to undo" in output.getvalue()
    assert not redo_file_for(history_file).exists()


def test_redo_restores_undone_exchange(tmp_path, monkeypatch):
    history_file, output = _setup(tmp_path, monkeypatch, _EXCHANGES)

    undo_commands.cmd_undo([])
    undo_commands.cmd_redo([])

    assert load_history(history_file) == _EXCHANGES
    assert load_redo_stack(redo_file_for(history_file)) == []
    assert "Restored" in output.getvalue()


def test_redo_with_empty_stack_prints_nothing_to_redo(tmp_path, monkeypatch):
    _, output = _setup(tmp_path, monkeypatch, _EXCHANGES)

    undo_commands.cmd_redo([])

    assert "Nothing to redo" in output.getvalue()


def test_parse_count_clamps_garbage_and_zero():
    assert undo_commands._parse_count([]) == 1
    assert undo_commands._parse_count(["garbage"]) == 1
    assert undo_commands._parse_count(["0"]) == 1
    assert undo_commands._parse_count(["3"]) == 3
