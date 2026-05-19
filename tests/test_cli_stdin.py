import io
import sys
import unittest
from unittest.mock import patch

import jarv.cli as cli


class PipedStringIO(io.StringIO):
    def isatty(self):
        return False


class TtyStringIO(io.StringIO):
    def isatty(self):
        return True


class CliStdinTests(unittest.TestCase):
    def test_compose_query_uses_args_only_when_no_stdin(self):
        self.assertEqual(cli._compose_query(["review", "this"]), "review this")

    def test_compose_query_uses_stdin_only(self):
        self.assertEqual(cli._compose_query([], "alpha\nbeta\n"), "alpha\nbeta")

    def test_compose_query_attaches_stdin_to_prompt(self):
        query = cli._compose_query(["summarize", "this"], "alpha\nbeta\n")

        self.assertTrue(query.startswith("summarize this\n\nInput from stdin:"))
        self.assertIn("```text\nalpha\nbeta\n```", query)

    def test_read_piped_stdin_truncates_to_limit(self):
        text, truncated = cli._read_piped_stdin(5, PipedStringIO("abcdef"))

        self.assertEqual(text, "abcde")
        self.assertTrue(truncated)

    def test_read_piped_stdin_rejects_binary_text(self):
        with self.assertRaises(ValueError):
            cli._read_piped_stdin(100, PipedStringIO("abc\x00def"))

    def test_main_combines_prompt_and_piped_stdin_for_agent(self):
        config = {
            "provider": "openai",
            "model": "test-model",
            "max_stdin_chars": 200000,
            "check_updates": False,
        }

        with (
            patch.object(sys, "argv", ["jarv", "summarize", "this"]),
            patch.object(sys, "stdin", PipedStringIO("hello from pipe")),
            patch.object(cli, "load_config", return_value=config),
            patch("jarv.config.is_setup_complete", return_value=True),
            patch.object(cli, "validate_config", return_value=True),
            patch("jarv.provider.resolve_api_key", return_value="key"),
            patch("jarv.provider.create_client", return_value=object()),
            patch("jarv.agent.run_agent") as run_agent,
        ):
            cli.main()

        query = run_agent.call_args.args[0]
        self.assertIn("summarize this", query)
        self.assertIn("Input from stdin:", query)
        self.assertIn("hello from pipe", query)

    def test_main_ignores_stdin_for_slash_commands(self):
        stdin = PipedStringIO("do not read this")

        with (
            patch.object(sys, "argv", ["jarv", "/history"]),
            patch.object(sys, "stdin", stdin),
            patch.object(cli, "_run_slash_command", return_value=True) as run_slash,
        ):
            cli.main()

        run_slash.assert_called_once_with("/history", [])
        self.assertEqual(stdin.tell(), 0)

    def test_main_does_not_prompt_for_bare_command_alias_when_stdin_is_piped(self):
        config = {
            "provider": "openai",
            "model": "test-model",
            "max_stdin_chars": 200000,
            "check_updates": False,
        }

        with (
            patch.object(sys, "argv", ["jarv", "history"]),
            patch.object(sys, "stdin", PipedStringIO("actual stdin")),
            patch.object(cli, "_maybe_command") as maybe_command,
            patch.object(cli, "load_config", return_value=config),
            patch("jarv.config.is_setup_complete", return_value=True),
            patch.object(cli, "validate_config", return_value=True),
            patch("jarv.provider.resolve_api_key", return_value="key"),
            patch("jarv.provider.create_client", return_value=object()),
            patch("jarv.agent.run_agent") as run_agent,
        ):
            cli.main()

        maybe_command.assert_not_called()
        query = run_agent.call_args.args[0]
        self.assertIn("history", query)
        self.assertIn("actual stdin", query)

    def test_main_preserves_heads_up_mode_without_args_or_stdin(self):
        config = {
            "provider": "openai",
            "model": "test-model",
            "max_stdin_chars": 200000,
        }

        with (
            patch.object(sys, "argv", ["jarv"]),
            patch.object(sys, "stdin", TtyStringIO("")),
            patch.object(cli, "load_config", return_value=config),
            patch("jarv.config.is_setup_complete", return_value=True),
            patch.object(cli, "validate_config", return_value=True),
            patch("jarv.provider.resolve_api_key", return_value="key"),
            patch("jarv.provider.create_client", return_value=object()),
            patch.object(cli, "run_heads_up_mode") as heads_up,
        ):
            cli.main()

        heads_up.assert_called_once()


if __name__ == "__main__":
    unittest.main()
