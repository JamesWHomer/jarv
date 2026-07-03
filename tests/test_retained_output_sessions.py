from pathlib import Path

import jarv.history as history
import jarv.session_store as session_store
from jarv.history import reads_file_for


def test_reads_sidecar_follows_archive_restore_and_delete(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    archive_dir = tmp_path / "archive"
    sessions_dir.mkdir()
    monkeypatch.setattr(history, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(session_store, "ARCHIVE_DIR", archive_dir)

    history_path = sessions_dir / "history-abc.json"
    history_path.write_text('[{"role":"user","content":"hello"}]', encoding="utf-8")
    reads_path = reads_file_for(history_path)
    reads_path.write_text('{"cmd_test":{"content":"output"}}', encoding="utf-8")

    archived_history = session_store.archive_session_files(history_path)

    assert archived_history is not None
    archived_reads = next(archive_dir.glob("reads-*.json"))
    assert archived_reads.exists()
    assert not reads_path.exists()

    restored_history = session_store.unarchive_session_files(
        archived_history,
        "session-id",
    )

    assert restored_history is not None
    restored_reads = reads_file_for(restored_history)
    assert restored_reads.read_text(encoding="utf-8") == (
        '{"cmd_test":{"content":"output"}}'
    )

    session_store.delete_session_files(restored_history)

    assert not restored_history.exists()
    assert not restored_reads.exists()
