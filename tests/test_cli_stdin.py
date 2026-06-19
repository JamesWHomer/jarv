import io
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import jarv.cli as cli
from jarv.headsup import HeadsupApp


class PipedStringIO(io.StringIO):
    def isatty(self):
        return False


class TtyStringIO(io.StringIO):
    def isatty(self):
        return True


class CliStdinTests(unittest.TestCase):
    def _headsup_app(self, config=None, client=object(), args=None, handle_slash=None, maybe_command=None, module=None):
        ready = threading.Event()
        ready.set()
        return HeadsupApp(
            config or {"model": "test"},
            client,
            args=args,
            agent_loader=({"module": module or SimpleNamespace()}, ready),
            handle_slash=handle_slash or (lambda command, rest, config, client, args, hint: (config, client)),
            maybe_command=maybe_command or (lambda _first, _rest: None),
        )

    def test_parser_accepts_provider_override(self):
        args = cli._build_parser().parse_args(["--provider", "ANTHROPIC", "hello"])

        self.assertEqual(args.provider, "anthropic")
        self.assertEqual(args.query, ["hello"])

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

    def test_read_piped_stdin_replaces_lone_surrogates(self):
        text, truncated = cli._read_piped_stdin(100, PipedStringIO("abc\udc8fdef"))

        self.assertEqual(text, "abc?def")
        self.assertFalse(truncated)
        text.encode("utf-8")

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
            patch("jarv.provider.create_client", return_value=object()) as create_client,
            patch("jarv.agent.run_agent") as run_agent,
        ):
            cli.main()

        query = run_agent.call_args.args[0]
        self.assertIn("summarize this", query)
        self.assertIn("Input from stdin:", query)
        self.assertIn("hello from pipe", query)
        create_client.assert_not_called()
        self.assertIsNone(run_agent.call_args.kwargs["client"])

    def test_main_applies_provider_override_without_running_setup(self):
        config = {
            "provider": "openai",
            "model": "test-model",
            "max_stdin_chars": 200000,
            "check_updates": False,
        }

        with (
            patch.object(sys, "argv", ["jarv", "--provider", "anthropic", "hello"]),
            patch.object(sys, "stdin", TtyStringIO("")),
            patch.object(cli, "load_config", return_value=config),
            patch("jarv.config.is_setup_complete", return_value=False),
            patch.object(cli, "cmd_setup") as setup,
            patch.object(cli, "validate_config", return_value=True),
            patch("jarv.provider.resolve_api_key", return_value="key") as resolve_key,
            patch("jarv.agent.run_agent") as run_agent,
        ):
            cli.main()

        setup.assert_not_called()
        runtime_config = run_agent.call_args.args[1]
        self.assertEqual(runtime_config["provider"], "anthropic")
        self.assertEqual(config["provider"], "openai")
        self.assertEqual(resolve_key.call_args.args[0]["provider"], "anthropic")

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

    def test_main_returns_update_failure_exit_code(self):
        with (
            patch.object(sys, "argv", ["jarv", "/update"]),
            patch("jarv.commands.cmd_update", return_value=1),
            self.assertRaises(SystemExit) as raised,
        ):
            cli.main()

        self.assertEqual(raised.exception.code, 1)

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
            patch("jarv.provider.create_client", return_value=object()) as create_client,
            patch("jarv.agent.run_agent") as run_agent,
        ):
            cli.main()

        maybe_command.assert_not_called()
        query = run_agent.call_args.args[0]
        self.assertIn("history", query)
        self.assertIn("actual stdin", query)
        create_client.assert_not_called()
        self.assertIsNone(run_agent.call_args.kwargs["client"])

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

    def test_main_exits_130_when_one_shot_turn_is_cancelled(self):
        config = {
            "provider": "openai",
            "model": "test-model",
            "max_stdin_chars": 200000,
            "check_updates": False,
        }

        with (
            patch.object(sys, "argv", ["jarv", "cancel", "me"]),
            patch.object(sys, "stdin", TtyStringIO("")),
            patch.object(cli, "load_config", return_value=config),
            patch("jarv.config.is_setup_complete", return_value=True),
            patch.object(cli, "validate_config", return_value=True),
            patch("jarv.provider.resolve_api_key", return_value="key"),
            patch("jarv.agent.run_agent", return_value=SimpleNamespace(cancelled=True)),
        ):
            with self.assertRaises(SystemExit) as raised:
                cli.main()

        self.assertEqual(raised.exception.code, 130)

    def test_heads_up_mode_reloads_runtime_after_config_mutating_command(self):
        config = {"provider": "openai", "model": "old-model"}
        refreshed = {"provider": "openai", "model": "new-model"}
        args = SimpleNamespace(
            provider=None,
            model=None,
            effort=None,
            timeout=None,
            system=None,
        )
        calls = []

        def handle_slash(command, rest, current_config, current_client, current_args, unknown_help_hint):
            calls.append((command, rest, current_config, current_client, current_args, unknown_help_hint))
            return refreshed, "new-client"

        app = self._headsup_app(
            config,
            client="old-client",
            args=args,
            handle_slash=handle_slash,
        )
        app._handle_query("/set model new-model")

        self.assertEqual(calls, [("/set", ["model", "new-model"], config, "old-client", args, True)])
        self.assertEqual(app.config, refreshed)
        self.assertEqual(app.client, "new-client")

    def test_heads_up_mode_handles_jarv_prefixed_slash_command(self):
        calls = []

        def handle_slash(command, rest, current_config, current_client, current_args, unknown_help_hint):
            calls.append((command, rest))
            return current_config, current_client

        app = self._headsup_app(handle_slash=handle_slash)
        app._handle_query("jarv /set model new-model")

        self.assertEqual(calls, [("/set", ["model", "new-model"])])

    def test_heads_up_mode_skips_reload_for_unknown_slash_command(self):
        args = SimpleNamespace(
            provider=None,
            model=None,
            effort=None,
            timeout=None,
            system=None,
        )
        calls = []

        def handle_slash(command, rest, current_config, current_client, current_args, unknown_help_hint):
            calls.append((command, rest))
            return current_config, current_client

        app = self._headsup_app(args=args, handle_slash=handle_slash)
        app._handle_query("/bogus")

        self.assertEqual(calls, [("/bogus", [])])

    def test_heads_up_mode_restores_cancelled_prompt(self):
        run_agent_calls = []

        def run_agent(query, config, client, **kwargs):
            run_agent_calls.append((query, config, client, kwargs))
            return SimpleNamespace(cancelled=True, prompt="draft prompt")

        app = self._headsup_app(module=SimpleNamespace(run_agent=run_agent))
        app._handle_query("draft prompt")

        self.assertEqual(len(run_agent_calls), 1)
        self.assertTrue(run_agent_calls[0][3]["heads_up"])
        self.assertIsNotNone(run_agent_calls[0][3]["ui"])
        self.assertEqual(app.editor["buffer"], "draft prompt")


if __name__ == "__main__":
    unittest.main()
