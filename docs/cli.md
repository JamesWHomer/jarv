# CLI flags

Flags override settings for one invocation without changing saved config. Dedicated flags take precedence over `--config`; repeated `--config` keys use the last value. Most flags work in both one-shot and heads-up mode. `--output-format`, `--quiet`, `--verbose`, and `--non-interactive` require a prompt or non-empty stdin and never open heads-up mode. Runtime flags cannot be combined with slash commands.

| Flag | Short | Description |
| --- | --- | --- |
| `--provider PROVIDER` | | Override the provider (`openai`, `anthropic`, `gemini`, etc.) |
| `--model MODEL` | `-m` | Override the model (e.g. `gpt-5.4-mini`) |
| `--effort EFFORT` | `-e` | Override reasoning effort with a value supported by the selected model |
| `--timeout SECONDS` | | Override shell command timeout/check-in seconds |
| `--system PROMPT` | `-s` | Override the system prompt |
| `--new` | | Start a fresh session (ignore prior history, but still save) |
| `--incognito` | | Don't load or save session history |
| `--session ID` | | Use or create a named session without rebinding the terminal; archived sessions must be restored first |
| `--config KEY=VALUE` | `-c` | Repeatable, validated setting override; lists/maps use JSON |
| `--cwd PATH` | `-C` | Working directory for commands, files, and project context |
| `--base-url URL` | | Override the provider API endpoint |
| `--service-tier TIER` | | `standard`, `flex`, `priority`, or `ultrafast`, where supported by the active provider/model |
| `--command-safety LEVEL` | | Override command/edit approval policy: `all`, `risky`, or `none` |
| `--tools LIST` | | Allow only these comma-separated tools, including for subagents |
| `--no-tools` | | Disable all agent tools |
| `--max-turns N` | | Maximum agent model turns shared by the root and all subagents |
| `--run-timeout SECONDS` | | Cancel the entire agent run at a deadline, including active commands and subagents |
| `--non-interactive` | | Never prompt for setup, clarification, or approval; fail if user input is required |
| `--output-format FORMAT` | | Clean final `text`, one `json` result, or streaming `jsonl` events on stdout; diagnostics go to stderr |
| `--prompt-file PATH` | | Read a UTF-8 prompt file instead of a positional prompt; piped stdin can still be attached |
| `--system-file PATH` | | Read a UTF-8 system prompt file instead of `--system` |
| `--no-project-context` | | Skip project instructions and git context |
| `--no-update-check` | | Skip background update checks |
| `--no-color` | | Disable colour (also supported through `NO_COLOR`) |
| `--quiet` | `-q` | Clean one-shot answer with progress suppressed; errors remain on stderr |
| `--verbose` | | Clean one-shot answer with runtime details and progress on stderr |
| `--version` | | Print the version and exit |
| `--help` | `-h` | Print CLI flag help and exit |

```bash
jarv --provider anthropic -m claude-sonnet-4-6 "summarise this repo"
jarv -m gpt-5.4-mini "summarise this repo"
jarv --effort high "refactor the auth module"
jarv --new "start fresh without prior context"
jarv --incognito "one-off task without saved conversation history"
jarv --timeout 120 --system "You are a poet" "write me a haiku"
git diff | jarv --non-interactive --output-format json --no-tools "review this patch"
jarv -C ./my-project --session nightly --max-turns 20 --run-timeout 300 "check the project"
jarv --prompt-file review.txt --system-file reviewer.txt --tools read,web_search
jarv -c audit=false -c max_tool_output_chars=40000 --command-safety all "investigate"
```

For Astra Ultrafast, run `jarv --provider openai --model gpt-6-astra --service-tier ultrafast "your task"`, or select **Processing tier → ultrafast** in `/settings` after selecting Astra. Standard remains the default. Ultrafast uses the existing HTTP streaming transport and keeps your reasoning effort unchanged.

Jarv currently offers Ultrafast only for `gpt-6-astra` at the direct OpenAI Responses endpoint (`https://api.openai.com/v1`). It uses API billing, with token rates at 6x Standard, including the applicable cache and long-context rates. See [OpenAI's Ultrafast guide](https://developers.openai.com/api/docs/guides/ultrafast-mode) and [API pricing](https://developers.openai.com/api/docs/pricing?latest-pricing=ultrafast). Custom gateways and regional endpoints are not enabled for this tier.

Subagents inherit the selected tier. The command auditor uses Standard because it calls Chat Completions. Changing to an incompatible model or endpoint resets a saved Ultrafast preference to Standard with a notice; explicitly requesting an incompatible combination fails validation. Provider errors are surfaced without changing tiers. Usage records retain requested and served tiers; when the served tier is missing, Ultrafast cost remains unknown unless the provider reports a cost.

`--new`, `--incognito`, and `--session` are mutually exclusive. So are `--tools`/`--no-tools`, `--quiet`/`--verbose`, and `--system`/`--system-file`. File arguments are resolved relative to the directory where Jarv was launched, before `--cwd` is applied. Prompt-file contents are always treated as a prompt, even when they begin with a slash command name.

Incognito runs do not load or save conversation history, artifacts, retained outputs, or usage records. In heads-up mode, each incognito prompt also starts without earlier conversation context; history/session commands and `/usage` are unavailable. Config, catalog/update caches, clipboard files, provider requests, and actions performed by tools are outside this history setting.

`--tools` replaces the saved disabled-tool selection for this invocation. Available names are `run_command`, `web_search`, `read`, `edit`, `spawn`, and `ask_user`; subagents keep their internal `finish` tool. To restrict a run to reading, omit both `run_command` and `edit` from the allowlist.

`--non-interactive` preserves the selected safety policy. Commands that need human approval fail with exit code 3; it never implicitly grants approval. Under `risky`, the auditor can still approve a command if auditor auto-approval is enabled. `ask_user` also ends an unattended run with code 3. `--max-turns` counts agent response rounds, including subagent rounds and interactive-command continuations; transport retries, audits, and history compaction do not count. `--run-timeout` uses cooperative cancellation and allows resource cleanup to finish. In heads-up mode, these limits restart for each submitted prompt.

Machine output uses a final result object with `type`, `status`, `text`, `error`, `session_id`, `turns`, and `exit_code`. `status` is `success`, `error`, `cancelled`, `input_required`, or `limit`. JSONL additionally emits `start`, `turn_start`, `text_delta`, `tool_call`, `tool_result`, and `retry` events. Tool events identify the agent and call ID; text deltas identify the root turn. On `retry`, discard that turn's previous deltas; the final result's `text` is authoritative. Progress and errors never enter protocol stdout. Invalid arguments rejected by the parser use normal stderr usage messages before the output protocol starts.

Exit codes: **0** success, **1** agent/provider failure or run limit, **2** invalid invocation, **3** required input unavailable, **130** cancellation. Without explicit output/verbosity flags, existing terminal rendering is preserved.
