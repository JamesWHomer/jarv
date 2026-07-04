from jarv.command_menu import argument_entries, filter_entries, menu_entries
from jarv.command_registry import COMMANDS


def test_menu_entries_follow_registry_order_and_hide_session_alias():
    names = [entry.name for entry in menu_entries()]

    expected = [name for name, meta in COMMANDS.items() if meta.menu] + ["exit"]
    assert names == expected
    assert "session" not in names
    assert "sessions" in names


def test_menu_entries_lead_with_everyday_commands():
    # Declaration order is the menu order: the empty-query view should surface
    # daily-driver commands, not one-time reference commands like /about.
    names = [entry.name for entry in menu_entries()]

    assert names[0] == "settings"
    assert names.index("usage") < names.index("about")
    assert names.index("new") < names.index("update")


def test_menu_entries_include_exit_as_final_entry():
    entries = menu_entries()

    exit_entry = entries[-1]
    assert exit_entry.display == "/exit"
    assert exit_entry.insert == "/exit"
    assert exit_entry.runs_on_enter is True
    assert exit_entry.summary == "Leave heads-up mode"
    assert "quit" in exit_entry.aliases


def test_menu_entries_carry_display_summary_and_arg_hint():
    by_name = {entry.name: entry for entry in menu_entries()}

    assert by_name["settings"].display == "/settings"
    assert by_name["settings"].summary == "Open common controls"
    assert by_name["settings"].arg_hint == ""
    assert by_name["set"].arg_hint == "<key> <value>"
    assert by_name["usage"].arg_hint == "[session|day|week|month|all]"


def test_menu_entries_complete_to_runnable_or_open_ended_text():
    by_name = {entry.name: entry for entry in menu_entries()}

    # Parameterless commands complete to the bare command and run on Enter.
    assert by_name["settings"].insert == "/settings"
    assert by_name["settings"].runs_on_enter is True
    # Commands with possible arguments gain a trailing space and defer running.
    assert by_name["usage"].insert == "/usage "
    assert by_name["usage"].runs_on_enter is False


def test_argument_entries_for_usage_periods():
    entries = argument_entries("usage")

    assert [entry.name for entry in entries] == ["session", "day", "week", "month", "all"]
    day = entries[1]
    assert day.display == "day"
    assert day.insert == "/usage day"
    assert day.runs_on_enter is True


def test_argument_entries_for_set_expect_a_value_after_the_key():
    entries = argument_entries("set")

    assert entries
    from jarv.config import DEFAULT_CONFIG

    assert [entry.name for entry in entries] == list(DEFAULT_CONFIG.keys())
    for entry in entries:
        assert entry.insert == f"/set {entry.name} "
        assert entry.runs_on_enter is False


def test_argument_entries_for_unset_run_on_enter():
    entries = argument_entries("unset")

    assert entries
    for entry in entries:
        assert entry.insert == f"/unset {entry.name}"
        assert entry.runs_on_enter is True


def test_argument_entries_for_setup_cover_every_step():
    from jarv.setup import SETUP_STEPS

    entries = argument_entries("setup")

    assert {entry.name for entry in entries} == set(SETUP_STEPS)
    assert entries[0].name == "provider"


def test_argument_entries_empty_for_commands_without_choices():
    assert argument_entries("help") == []
    assert argument_entries("btw") == []
    assert argument_entries("nonsense") == []


def test_filter_empty_query_returns_all_entries():
    entries = menu_entries()
    assert filter_entries(entries, "") == entries


def test_filter_orders_prefix_matches_before_substring_matches():
    result = [entry.name for entry in filter_entries(menu_entries(), "se")]

    # Prefix matches (registry order) come first, then substring matches.
    assert result == ["settings", "sessions", "set", "setup", "unset"]


def test_filter_is_case_insensitive():
    assert filter_entries(menu_entries(), "SE") == filter_entries(menu_entries(), "se")


def test_filter_no_match_returns_empty():
    assert filter_entries(menu_entries(), "zzz") == []


def test_filter_preserves_registry_order_within_prefix_tier():
    result = [entry.name for entry in filter_entries(menu_entries(), "s")]

    # The names that start with "s", in declaration order, lead the results;
    # "session" stays hidden.
    assert result[:4] == ["settings", "sessions", "set", "setup"]
    assert "session" not in result


def test_suggest_commands_finds_close_spellings():
    from jarv.command_registry import suggest_commands

    assert "history" in suggest_commands("histry")
    assert suggest_commands("sett")[0] == "settings"
    assert suggest_commands("") == []
    assert suggest_commands("zzzzzz") == []


def test_filter_matches_hidden_aliases_but_shows_the_canonical_entry():
    result = filter_entries(menu_entries(), "qu")

    assert [entry.name for entry in result] == ["exit"]
    assert result[0].display == "/exit"

    result = filter_entries(menu_entries(), "session")
    assert [entry.name for entry in result] == ["sessions"]
