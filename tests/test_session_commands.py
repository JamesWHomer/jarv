from jarv import session_commands


def test_cwd_label_prefers_leaf_directory_for_windows_paths():
    assert session_commands._cwd_label(r"C:\Users\ubers\Desktop\jarv") == "jarv"
    assert session_commands._cwd_label("/home/ubers/projects/jarv") == "jarv"
    assert session_commands._cwd_label("") == ""


def test_session_metadata_widths_are_progressive():
    assert session_commands._session_metadata_widths(20) == (0, 20)
    assert session_commands._session_metadata_widths(40) == (18, 22)
    assert session_commands._session_metadata_widths(60) == (18, 42)
