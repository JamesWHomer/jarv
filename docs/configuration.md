# Configuration

Settings live in `~/.jarv/config.json` (created when config is first loaded). Use `/settings` for the common controls, `/set` and `/unset` for individual values, or edit the JSON file directly for maps and lists. Unlike `--config`, `/set` does not parse JSON maps/lists. Invalid JSON is reported and preserved; Jarv does not replace it with defaults.

| Key | Default | Description |
| --- | --- | --- |
| `provider` | `"openai"` | API provider: `openai`, `anthropic`, `gemini`, `openrouter`, `groq`, `deepseek`, `together`, `fireworks`, `ollama`, `lm_studio`, `vllm`. |
| `api_key` | `""` | Legacy single API key field (migrated to `api_keys`). |
| `api_keys` | `{}` | Per-provider API keys. Falls back to provider env vars when empty. |
| `base_url` | `""` | Custom API base URL. Overrides the provider default. |
| `model` | `"gpt-5.4-mini"` | Model name passed to the API. |
| `service_tiers` | `{}` | Per-provider processing tier: `standard`, `flex`, `priority`, or `ultrafast`. Missing providers use `standard`; unsupported tiers are not offered. Ultrafast requires direct OpenAI Astra. |
| `reasoning_effort` | `""` | Model-supported reasoning effort. Empty uses the provider/model default; `none` explicitly disables reasoning only where supported. |
| `context_budget_ratio` | `0.75` | Share of the context window used for input. |
| `context_compaction_threshold` | `0.85` | Fill ratio that triggers history compaction. |
| `context_output_reserve_ratio` | `0.15` | Context window share reserved for model output. |
| `context_window_fallback` | `128000` | Context window when neither the provider nor the models.dev catalog knows the model. |
| `max_stdin_chars` | `200000` | Maximum piped stdin characters attached to a one-shot prompt. |
| `max_tool_output_chars` | `20000` | Maximum generic tool output characters returned to the model. It also supplies the default head/tail budget for `run_command`. |
| `project_context_max_chars` | `16000` | Maximum project-context file characters injected into the system prompt (longer files are truncated head+tail). |
| `disabled_tools` | `[]` | Tool names omitted from root agents and subagents. Configure these from the Tools section in `/settings`. |
| `interactive_commands` | `false` | Experimental. Hold a command that is waiting on stdin open and let the model type into it. |
| `command_timeout` | `60` | Seconds before non-interactive shell commands are killed, or before interactive commands check in again. |
| `interactive_max_rounds` | `40` | Model interaction rounds allowed for one interactive command before Jarv kills the process. Only used when `interactive_commands` is on. |
| `persistent_shell` | `true` | Reuse PowerShell for noninteractive Windows commands with an isolated execution context per command. Disable to always start a fresh process. |
| `web_timeout` | `15` | Total seconds allowed for a web search, including queueing, retries and pagination, or a URL request, including redirects and body transfer. |
| `web_search_interval` | `1` | Minimum seconds between DuckDuckGo requests across agents in this process. Only one search request runs at a time. |
| `web_search_max_pages` | `5` | Maximum result pages fetched per search; at most two additional attempts are allowed for transient failures. |
| `command_safety` | `"risky"` | Command approval policy: `all` (human approval for every command), `risky` (review flagged commands, allowing auditor auto-approval), `none` (no approval gate). |
| `audit` | `true` | LLM auditor for flagged commands. |
| `auditor_auto_approve` | `true` | Let the auditor auto-approve commands it deems safe under `command_safety=risky`. With `all`, human approval is always required. |
| `auditor_model` | `""` | Auditor model. Empty uses the active `model`. |
| `max_subagent_depth` | `4` | Maximum nesting depth for spawned subagents. |
| `subagent_thread_pool_max_workers` | `8` | Max parallel subagents per `spawn` call. |
| `subagent_timeout` | `600` | Maximum runtime in seconds for one `spawn` batch before unfinished subagents are cancelled. |
| `check_updates` | `true` | Background check on one-shot prompt runs, except `--quiet` (non-blocking; successful checks throttled to once per 24h; PyPI for Python installs, GitHub Releases for direct standalone installs; Scoop, WinGet, and Homebrew manage their own update availability). |
| `read_only_command_display` | `"fullscreen"` | Display mode for `/help`, `/about`, `/usage`, and `/config`: temporary `fullscreen` view or permanent `print` output. |
| `tool_call_display` | `"auto"` | Tool-call layout: `auto` selects `print` for one-shot runs and `fullscreen` in heads-up mode; explicit modes override it. |
| `tool_output_display_lines` | `"auto"` | Display-only output line budget per tool card: `auto` uses one-third of terminal height in `print` layout and one-half in `fullscreen`, with a minimum of 3; an integer of at least 3 fixes the budget. |
| `turn_summary` | `false` | Show one summary after Jarv finishes responding to your prompt in one-shot and heads-up mode. Turning this off preserves the field selections below. |
| `turn_summary_tokens` | `true` | Include input, output, and total tokens summed across successful model requests in the turn. |
| `turn_summary_cache` | `true` | Include provider-reported cached input tokens summed across the turn. |
| `turn_summary_reasoning` | `true` | Include provider-reported reasoning output tokens summed across the turn. |
| `turn_summary_speed` | `true` | Include the final response's server-reported generation tok/s when available, otherwise estimated text streaming speed excluding the initial wait. Short or unmeasurable streams show unavailable. |
| `turn_summary_time` | `true` | Include total model request time across the turn, excluding tool execution. |
| `turn_summary_session` | `false` | Include the running token total for the saved session. |
| `turn_summary_cost` | `false` | Include running session cost, labelled when estimated or incomplete. |
| `headsup_border` | `true` | Show outer frames in heads-up mode and all menus. Turn off to remove borders and side padding while keeping headers, footers, and the heads-up input box. |
| `headsup_intro_logo` | `true` | Show the rainbow JARV logo, wave, and welcome hint in new heads-up sessions. Toggle **Rainbow JARV** in Display settings. |
| `headsup_intro_stars` | `true` | Show twinkling stars in new heads-up sessions. Toggle **Welcome stars** in Display settings. Turn both welcome settings off for a blank area. |
| `colour` | `true` | Render in colour. Set to `false` to keep only bold, dim, and underline. The `NO_COLOR` environment variable also disables colour. |
| `system_prompt` | `"You are Jarv..."` | System instructions sent with each request. |
| `project_context` | `true` | Read `JARV.md`/`AGENTS.md`/`CLAUDE.md` and git branch, status, and recent commits into the system prompt. |

Processing tier choices depend on the active provider. Jarv offers Standard, Flex, and Priority for OpenAI, OpenRouter, and Gemini; OpenAI also offers Ultrafast for the direct Astra route described in the [CLI reference](cli.md). Anthropic offers Standard and Priority; Jarv maps Priority to Anthropic's `auto` tier. Other providers remain on Standard. Apart from Ultrafast's explicit model/endpoint check, the provider API determines whether a selected tier is available for the model and account.

Legacy `print_usage_after_model` and `print_usage_after_agent` settings migrate automatically to the unified turn summary. Model stats enable the token and timing fields; print usage enables token, session, and cost fields. Existing `turn_summary*` values take precedence. The summary appears once at the end of the full turn, after tool calls and internal continuations finish.
