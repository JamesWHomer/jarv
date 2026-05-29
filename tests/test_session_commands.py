import json
from pathlib import Path

from jarv import session_commands


def test_compact_tokens_keeps_session_rows_short():
    assert session_commands._compact_tokens(0) == "0 tok"
    assert session_commands._compact_tokens(999) == "999 tok"
    assert session_commands._compact_tokens(12_340) == "12.3k tok"
    assert session_commands._compact_tokens(1_200_000) == "1.2m tok"


def test_cwd_label_prefers_leaf_directory_for_windows_paths():
    assert session_commands._cwd_label(r"C:\Users\ubers\Desktop\jarv") == "jarv"
    assert session_commands._cwd_label("/home/ubers/projects/jarv") == "jarv"
    assert session_commands._cwd_label("") == ""


def test_session_metadata_widths_are_progressive():
    assert session_commands._session_metadata_widths(20) == (0, 0, 20)
    assert session_commands._session_metadata_widths(40) == (10, 0, 30)
    assert session_commands._session_metadata_widths(60) == (10, 18, 32)


def test_session_total_tokens_reads_usage_sidecar(tmp_path):
    history_path = tmp_path / "history-abc.json"
    usage_path = tmp_path / "usage-abc.json"
    history_path.write_text("[]", encoding="utf-8")
    usage_path.write_text(
        json.dumps({"totals": {"total_tokens": 12345}}),
        encoding="utf-8",
    )

    total = session_commands._session_total_tokens(
        {"history_file": str(history_path)},
        "session-id",
    )

    assert total == 12345
