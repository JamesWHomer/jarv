"""Saved-history observations follow successful commits, not later disk reads."""

import copy
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from jarv import agent, history, storage
from jarv.config import DEFAULT_CONFIG
from jarv.provider import StreamDone, TextDelta


def make_persistence(path, observer, *, incognito=False):
    persistence = agent.SessionPersistence(
        incognito=incognito, on_history_saved=observer,
    )
    persistence.session_context = SimpleNamespace(history_file=path)
    persistence.history = [{"role": "user", "content": "hello\ud800"}]
    return persistence


def test_observer_receives_committed_normalized_snapshot(tmp_path):
    path = tmp_path / "history.json"
    observations = []

    def observe(saved_path, saved_history):
        assert json.loads(saved_path.read_text(encoding="utf-8")) == saved_history
        observations.append((saved_path, saved_history))

    persistence = make_persistence(path, observe)
    persistence.save_turn()
    assert observations == [(path, [{"role": "user", "content": "hello?"}])]
    persistence.history[0]["content"] = "later turn"
    assert observations[0][1][0]["content"] == "hello?"


def test_observer_does_not_adopt_external_change_after_commit(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    observations = []
    transaction = storage.transaction

    @contextmanager
    def commit_then_external_edit(target):
        with transaction(target):
            yield
        history.save_history([{"role": "user", "content": "external edit"}], path)

    monkeypatch.setattr(agent, "transaction", commit_then_external_edit)
    persistence = make_persistence(path, lambda *args: observations.append(args))
    persistence.save_turn()
    assert history.load_history(path)[0]["content"] == "external edit"
    assert observations == [(path, [{"role": "user", "content": "hello?"}])]


def test_aborted_transaction_does_not_notify(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    history.save_history([{"role": "user", "content": "original"}], path)
    original = path.read_bytes()
    observations = []
    persistence = make_persistence(path, lambda *args: observations.append(args))
    atomic = storage._atomic

    def fail_journal(target, data):
        if target.name == ".jarv-transaction.json":
            raise OSError("disk full")
        return atomic(target, data)

    monkeypatch.setattr(storage, "_atomic", fail_journal)
    with pytest.raises(storage.StorageError, match="disk full"):
        persistence.save_turn()
    assert path.read_bytes() == original
    assert observations == []


def test_incognito_does_not_save_or_notify(tmp_path):
    path = tmp_path / "history.json"
    observations = []
    persistence = make_persistence(
        path, lambda *args: observations.append(args), incognito=True,
    )
    persistence.save_turn()
    assert not path.exists()
    assert observations == []


@pytest.mark.parametrize("mode", ["saved", "legacy", "incognito", "preparation_failure"])
def test_run_agent_reports_loaded_prefix_before_saved_turn(tmp_path, monkeypatch, mode):
    path = tmp_path / "history.json"
    existing = [{"role": "user", "content": "earlier prompt", "id": "earlier"}]
    if mode == "legacy":
        del existing[0]["id"]
    history.save_history(existing, path)
    context = history.SessionContext("test", "Test", path, datetime.now(timezone.utc))
    events = []

    def observe(kind, saved_path, items):
        events.append((kind, saved_path, copy.deepcopy(items)))

    ui = SimpleNamespace(
        history_loaded=lambda *args: observe("loaded", *args),
        history_saved=lambda *args: observe("saved", *args),
    )
    monkeypatch.setattr(agent, "prepare_session_context", lambda **_: context)
    monkeypatch.setattr(
        agent, "_prepare_client_and_instructions",
        lambda _config, client, **_: (client, "system"),
    )
    monkeypatch.setattr(
        agent, "stream_response",
        lambda *_args, **_kwargs: iter([TextDelta("new reply"), StreamDone(None)]),
    )
    if mode == "preparation_failure":
        def fail_sidecar(_path):
            raise storage.StorageError("cannot load sidecar")

        monkeypatch.setattr(agent, "load_artifact_store", fail_sidecar)

    result = agent.run_agent(
        "new prompt", DEFAULT_CONFIG, client=object(), ui=ui,
        incognito=mode == "incognito",
    )

    if mode in {"saved", "legacy"}:
        assert result.error is None
        assert [event[0] for event in events] == ["loaded", "saved"]
        assert events[0] == ("loaded", path, existing)
        assert events[1] == ("saved", path, history.load_history(path))
        assert events[1][2][0]["id"]
        assert [item["content"] for item in events[1][2] if item.get("role")] == [
            "earlier prompt", "new prompt", "new reply",
        ]
    else:
        assert events == []
        assert history.load_history(path) == existing
        assert result.error == ("cannot load sidecar" if mode == "preparation_failure" else None)
