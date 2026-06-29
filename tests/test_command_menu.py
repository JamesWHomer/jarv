from jarv.command_menu import filter_entries, menu_entries
from jarv.command_registry import COMMANDS


def test_menu_entries_follow_registry_order_and_hide_session_alias():
    names = [entry.name for entry in menu_entries()]

    expected = [name for name, meta in COMMANDS.items() if meta.menu] + ["exit"]
    assert names == expected
    assert "session" not in names
    assert "sessions" in names


def test_menu_entries_include_exit_as_final_entry():
    entries = menu_entries()

    exit_entry = entries[-1]
    assert exit_entry.display == "/exit"
    assert exit_entry.takes_rest is False
    assert exit_entry.summary == "Leave heads-up mode"


def test_menu_entries_carry_display_summary_and_arg_hint():
    by_name = {entry.name: entry for entry in menu_entries()}

    assert by_name["settings"].display == "/settings"
    assert by_name["settings"].summary == "Open common controls"
    assert by_name["settings"].arg_hint == ""
    assert by_name["set"].arg_hint == "<key> <value>"
    assert by_name["set"].takes_rest is True
    assert by_name["usage"].arg_hint == "[session|day|week|month|all]"


def test_filter_empty_query_returns_all_entries():
    entries = menu_entries()
    assert filter_entries(entries, "") == entries


def test_filter_orders_prefix_matches_before_substring_matches():
    result = [entry.name for entry in filter_entries(menu_entries(), "se")]

    # Prefix matches (registry order) come first, then substring matches.
    assert result == ["setup", "sessions", "set", "settings", "unset"]


def test_filter_is_case_insensitive():
    assert filter_entries(menu_entries(), "SE") == filter_entries(menu_entries(), "se")


def test_filter_no_match_returns_empty():
    assert filter_entries(menu_entries(), "zzz") == []


def test_filter_preserves_registry_order_within_prefix_tier():
    result = [entry.name for entry in filter_entries(menu_entries(), "s")]

    # The names that start with "s", in declaration order, lead the results;
    # "session" stays hidden.
    assert result[:4] == ["setup", "sessions", "set", "settings"]
    assert "session" not in result
