from jarv import session_commands


def test_session_row_widths_preserve_message_space_on_small_screens():
    assert session_commands._session_row_widths(40) == (13, 7, 16)
    assert session_commands._session_row_widths(58) == (28, 7, 19)


def test_session_row_widths_give_extra_space_to_message_on_large_screens():
    assert session_commands._session_row_widths(110) == (28, 7, 71)
