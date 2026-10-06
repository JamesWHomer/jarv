"""Undo, redo, new turns, and archive share one lossless prompt tree."""

import copy
from types import SimpleNamespace

import pytest

from conftest import make_console
from jarv import history, session_store, storage, undo_commands
from jarv.agent import SessionPersistence
from jarv.session_tree import (
    ancestors_and_self, checkout, delete_subtree, load_session_tree,
    preserve_redo_branches,
)


def frame(name, *, legacy=False):
    user = {"role": "user", "content": name}
    if not legacy:
        user["id"] = name
    return [user, {"role": "assistant", "content": "answer " + name}]


def prompts(items):
    return [item["content"] for item in items if item.get("role") == "user"]


@pytest.fixture
def session(tmp_path, monkeypatch):
    path = tmp_path / "sessions" / "history-test.json"
    history.save_history(frame("A") + frame("B") + frame("C"), path)
    history.save_branches([
        {"parent_frame_id": "B", "items": frame("D")},
    ], history.branches_file_for(path))
    monkeypatch.setattr(undo_commands, "prepare_session_context",
                        lambda: SimpleNamespace(history_file=path))
    monkeypatch.setattr(undo_commands, "console", make_console()[0])
    return path


def assert_tree(path, expected=("A", "B", "C", "D")):
    model = load_session_tree(path)
    assert sorted(node.prompt_text for node in model.nodes) == sorted(expected)
    assert len(model.nodes) == len(model.by_id)
    return model


def test_undo_ancestor_then_checkout_retains_original_context(session):
    undo_commands.cmd_undo(["2"])
    model = assert_tree(session)
    assert [node.frame_id for node in ancestors_and_self(model.find("D"))] == ["A", "B", "D"]
    assert checkout(session, leaf_id="D")
    assert prompts(history.load_history(session)) == ["A", "B", "D"]
    assert_tree(session)
    assert not history.load_redo_stack(history.redo_file_for(session))
    assert checkout(session, leaf_id="C")
    assert prompts(history.load_history(session)) == ["A", "B", "C"]


def test_repeated_undo_and_partial_redo_never_duplicate_frames(session):
    undo_commands.cmd_undo([])
    undo_commands.cmd_undo([])
    assert_tree(session)
    undo_commands.cmd_redo([])
    assert_tree(session)
    assert prompts(history.load_history(session)) == ["A", "B"]
    undo_commands.cmd_undo(["2"])
    assert history.load_history(session) == []
    assert_tree(session)
    undo_commands.cmd_redo(["3"])
    assert prompts(history.load_history(session)) == ["A", "B", "C"]
    assert_tree(session)
    assert len(history.load_branches(history.branches_file_for(session))) == 1


def test_legacy_redo_gets_stable_ids_and_parents_once(tmp_path):
    path = tmp_path / "history-legacy.json"
    history.save_history(frame("A", legacy=True), path)
    history.save_redo_stack([frame("C", legacy=True), frame("B", legacy=True)],
                            history.redo_file_for(path))
    loaded = preserve_redo_branches(path)
    model = load_session_tree(path)
    by_prompt = {node.prompt_text: node for node in model.nodes}
    assert [node.prompt_text for node in ancestors_and_self(by_prompt["C"])] == ["A", "B", "C"]
    files = [path, history.redo_file_for(path), history.branches_file_for(path)]
    original = [file.read_bytes() for file in files]
    assert preserve_redo_branches(path) == loaded
    assert [file.read_bytes() for file in files] == original
    redo_ids = {item[0]["id"] for item in history.load_redo_stack(history.redo_file_for(path))}
    assert redo_ids == {by_prompt[name].frame_id for name in ("B", "C")}


def test_normalizes_legacy_branch_parent_aliases_without_redo(tmp_path):
    path = tmp_path / "history-legacy.json"
    history.save_history(frame("A", legacy=True), path)
    history.save_branches([
        {"parent_frame_id": "a0", "items": frame("B", legacy=True)},
        {"parent_frame_id": "b0", "items": frame("C", legacy=True)},
    ], history.branches_file_for(path))
    model = load_session_tree(path)
    by_prompt = {node.prompt_text: node for node in model.nodes}
    assert [node.prompt_text for node in ancestors_and_self(by_prompt["C"])] == ["A", "B", "C"]
    assert all(node.frame_id not in {"a0", "b0", "b1"} for node in model.nodes)
    before = [file.read_bytes() for file in (path, history.branches_file_for(path))]
    load_session_tree(path)
    assert [file.read_bytes() for file in (path, history.branches_file_for(path))] == before


def test_normalization_deduplicates_stable_ids_on_active_path(tmp_path):
    path = tmp_path / "history.json"
    history.save_history(frame("A"), path)
    history.save_branches([
        {"parent_frame_id": "", "items": frame("A")},
        {"parent_frame_id": "A", "items": frame("B")},
        {"parent_frame_id": "A", "items": frame("B")},
    ], history.branches_file_for(path))
    assert_tree(path, ("A", "B"))
    assert len(history.load_branches(history.branches_file_for(path))) == 1


@pytest.mark.parametrize("clear_redo", [False, True])
def test_new_user_save_clears_redo_but_keeps_old_branch_parent(session, clear_redo):
    undo_commands.cmd_undo(["2"])
    persistence = SessionPersistence(incognito=False)
    persistence.session_context = SimpleNamespace(history_file=session)
    persistence.history = preserve_redo_branches(session)
    persistence.history.extend(frame("NEW"))
    persistence.new_user_message = True
    # The default save is the error checkpoint; save_turn also clears redo.
    persistence.save(clear_redo=clear_redo)
    assert not history.load_redo_stack(history.redo_file_for(session))
    assert not persistence.new_user_message
    model = assert_tree(session, ("A", "B", "C", "D", "NEW"))
    assert model.find("B").parent.frame_id == "A"
    assert model.find("NEW").parent.frame_id == "A"


def test_pre_user_checkpoint_preserves_redo(session):
    undo_commands.cmd_undo([])
    persistence = SessionPersistence(incognito=False)
    persistence.session_context = SimpleNamespace(history_file=session)
    persistence.history = preserve_redo_branches(session)
    persistence.save()
    assert len(history.load_redo_stack(history.redo_file_for(session))) == 1


def test_failed_new_user_commit_preserves_redo_and_can_retry(session, monkeypatch):
    undo_commands.cmd_undo([])
    persistence = SessionPersistence(incognito=False)
    persistence.session_context = SimpleNamespace(history_file=session)
    persistence.history = preserve_redo_branches(session)
    persistence.history.extend(frame("NEW"))
    persistence.new_user_message = True
    original = session.read_bytes()
    atomic = storage._atomic

    def fail_journal(path, data):
        if path.name == ".jarv-transaction.json":
            raise OSError("disk full")
        return atomic(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_atomic", fail_journal)
        with pytest.raises(storage.StorageError, match="disk full"):
            persistence.save()
    assert session.read_bytes() == original
    assert persistence.new_user_message
    assert len(history.load_redo_stack(history.redo_file_for(session))) == 1
    persistence.save()
    assert not history.load_redo_stack(history.redo_file_for(session))


def test_deleting_undone_subtree_cannot_be_reversed_by_redo(session):
    undo_commands.cmd_undo(["2"])
    assert delete_subtree(session, node_id="B")
    undo_commands.cmd_redo(["2"])
    assert_tree(session, ("A",))
    assert not history.load_redo_stack(history.redo_file_for(session))


def test_deleting_redo_leaf_retains_preceding_redo_frame(session):
    undo_commands.cmd_undo(["2"])
    assert delete_subtree(session, node_id="C")
    undo_commands.cmd_redo([])
    assert prompts(history.load_history(session)) == ["A", "B"]
    assert_tree(session, ("A", "B", "D"))


def test_undo_all_can_archive_restore_and_resume_branch(session, tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(session_store, "history_file_for_session", lambda sid: session)
    undo_commands.cmd_undo(["3"])
    archived = session_store.archive_session_files(session)
    assert archived is not None
    assert not session.exists()
    assert_tree(archived)
    assert not history.load_redo_stack(history.redo_file_for(archived))
    restored = session_store.unarchive_session_files(archived, "test")
    assert restored == session
    assert checkout(restored, leaf_id="D")
    assert prompts(history.load_history(restored)) == ["A", "B", "D"]
    assert_tree(restored)


def test_archive_moves_branches_created_from_legacy_redo_in_same_commit(tmp_path, monkeypatch):
    path = tmp_path / "sessions" / "history-legacy.json"
    history.save_history(frame("A", legacy=True), path)
    history.save_redo_stack([frame("B", legacy=True)], history.redo_file_for(path))
    monkeypatch.setattr(session_store, "ARCHIVE_DIR", tmp_path / "archive")
    archived = session_store.archive_session_files(path)
    assert archived is not None
    assert_tree(archived, ("A", "B"))
    assert not history.branches_file_for(path).exists()
    assert not history.redo_file_for(path).exists()


def test_normalization_failure_preserves_all_legacy_files(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    history.save_history(frame("A", legacy=True), path)
    redo = history.redo_file_for(path)
    history.save_redo_stack([frame("B", legacy=True)], redo)
    original = (path.read_bytes(), redo.read_bytes())

    def fail(*args):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_atomic", fail)
        with pytest.raises(storage.StorageError, match="disk full"):
            preserve_redo_branches(path)
    assert (path.read_bytes(), redo.read_bytes()) == original
    assert not history.branches_file_for(path).exists()


def test_conflicting_redo_copy_is_not_silently_overwritten(session):
    undone = frame("B")
    history.save_history(frame("A"), session)
    history.save_redo_stack([undone], history.redo_file_for(session))
    conflicting = copy.deepcopy(undone)
    conflicting[-1]["content"] = "different answer"
    history.save_branches([{"parent_frame_id": "A", "items": conflicting}],
                          history.branches_file_for(session))
    with pytest.raises(storage.StorageError, match="Conflicting copies"):
        preserve_redo_branches(session)


def test_preamble_is_preserved_when_all_user_frames_are_undone(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    history.save_history([{"role": "system", "content": "legacy context"}] + frame("A"), path)
    monkeypatch.setattr(undo_commands, "prepare_session_context", lambda: SimpleNamespace(history_file=path))
    monkeypatch.setattr(undo_commands, "console", make_console()[0])
    undo_commands.cmd_undo([])
    assert checkout(path, leaf_id="A")
    loaded = history.load_history(path)
    assert loaded[0]["content"] == "legacy context"
    assert prompts(loaded) == ["A"]


@pytest.mark.parametrize("failure", ["branches", "sidecar", "commit"])
def test_failed_agent_preparation_cannot_save_a_partial_migration(tmp_path, monkeypatch, failure):
    from jarv import agent
    from jarv.config import build_default_config

    path = tmp_path / "history.json"
    history.save_history(frame("A", legacy=True), path)
    branch_path = history.branches_file_for(path)
    history.save_branches([{"parent_frame_id": "a0", "items": frame("B", legacy=True)}], branch_path)
    if failure == "branches":
        branch_path.write_text('{"version":1,"frames":[false]}', encoding="utf-8")
    files = [path, branch_path]
    if failure == "sidecar":
        artifact_path = history.artifact_file_for(path)
        artifact_path.write_text('{broken', encoding="utf-8")
        files.append(artifact_path)
    original = [file.read_bytes() for file in files]
    context = history.SessionContext("test", "test", path, history.utc_now())
    monkeypatch.setattr(agent, "prepare_session_context", lambda **kwargs: context)
    monkeypatch.setattr(agent, "_prepare_client_and_instructions", lambda config, client, **kwargs: (client, ""))
    monkeypatch.setattr(agent, "console", make_console()[0])

    def unexpected_request(*args, **kwargs):
        pytest.fail("Invalid local state must fail before a provider request")

    monkeypatch.setattr(agent, "stream_response", unexpected_request)
    if failure == "commit":
        def fail(*args):
            raise OSError("disk full")
        monkeypatch.setattr(storage, "_atomic", fail)
    result = agent.run_agent("new prompt", build_default_config(), client=object())
    assert result.error
    assert [file.read_bytes() for file in files] == original
