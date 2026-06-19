# jarv

A multi-provider AI-powered CLI agent that can run shell commands, fan out work to parallel subagents, and keep track of conversation history across terminal sessions.

```bash
jarv                                    # start an interactive session
jarv whats the meaning of life?         # one-shot question
jarv commit all these files             # let it run commands to do the job
jarv refactor the auth module           # complex tasks get split across subagents
```

## Install

Jarv can be installed as a standalone binary or as a Python package. After installing, run `jarv /setup` to choose a provider, enter an API key, and pick a model.

```powershell
# WinGet package in progress:
# winget install JamesWHomer.Jarv
irm https://github.com/JamesWHomer/jarv/releases/latest/download/install.ps1 | iex
```

```bash
# Homebrew tap in progress:
# brew install JamesWHomer/tap/jarv
curl -fsSL https://github.com/JamesWHomer/jarv/releases/latest/download/install.sh | sh
uv tool install jarv
pipx install jarv
pip install jarv
```

Python package installs require **Python 3.10+**. Keys can also be set via provider-specific environment variables or `jarv /set`.

To upgrade:

```bash
jarv /update
```

![jarv update demo](https://github.com/user-attachments/assets/5dbbb528-6b6f-4d04-bc2d-d68652e67fc9)

## Usage

### One-shot mode

Pass a prompt as arguments. Jarv answers (running commands if needed) and exits.

```bash
jarv what process is using port 8080?
jarv find all TODO comments in src/
```

Jarv also accepts piped stdin as one-shot input. If you pass both stdin and prompt arguments, the arguments are treated as the instruction and stdin is attached as context.

```bash
git diff | jarv review this patch
cat README.md | jarv summarize this
rg TODO . | jarv group these by subsystem
```

### Heads-up mode

Run `jarv` with no arguments to enter an interactive prompt loop.

```
jarv> what files changed today?
jarv> now run the tests
jarv> /history
jarv> /new
```

- Type a prompt and press Enter.
- Slash commands start with `/` — type `/help` to list them.
- During a response, Esc or Ctrl+C stops further work, checkpoints the turn in history/context, and restores the prompt for editing. Use `/undo` to remove the turn.
- At the prompt, Esc or Ctrl+C clears existing text; press either again on an empty prompt to exit.
- You can also exit with `exit`, `quit`, or `/exit`.

![jarv interactive session](https://github.com/user-attachments/assets/d56b491b-4498-43f9-a93b-cfee7aee4791)

### Flags

Flags override config values for a single run and work in both one-shot and heads-up mode.

| Flag | Short | Description |
| --- | --- | --- |
| `--provider PROVIDER` | | Override the provider (`openai`, `anthropic`, `gemini`, etc.) |
| `--model MODEL` | `-m` | Override the model (e.g. `gpt-5.4-mini`) |
| `--effort EFFORT` | `-e` | Override reasoning effort with a value supported by the selected model |
| `--timeout SECONDS` | | Override command timeout in seconds |
| `--system PROMPT` | `-s` | Override the system prompt |
| `--new` | | Start a fresh session (ignore prior history, but still save) |
| `--incognito` | | Don't load or save session history |
| `--version` | | Print the version and exit |

```bash
jarv --provider anthropic -m claude-sonnet-4-6 "summarise this repo"
jarv -m gpt-5.4-mini "summarise this repo"
jarv --effort high "refactor the auth module"
jarv --new "start fresh without prior context"
jarv --incognito "one-off task, leave no trace"
jarv --timeout 120 --system "You are a poet" "write me a haiku"
```

## How it works

Jarv uses a multi-provider tool-calling agent loop (OpenAI Responses API, Anthropic Messages, Gemini, and OpenAI-compatible endpoints). The root model can call five tools:

| Tool | Purpose |
| --- | --- |
| `run_command` | Execute a shell command and return stdout, stderr, and exit code |
| `web_search` | Search the web through DuckDuckGo's public HTML endpoint |
| `read` | Page through command output, artifacts, URLs, or local files |
| `spawn` | Fan out work to parallel subagents, each with their own tool access |
| `ask_user` | Ask you a question and wait for a reply |

Each tool can be enabled or disabled from `jarv /settings`. Disabled tools are not sent to the model and are also unavailable to spawned subagents. The subagent-only `finish` tool remains enabled so child agents can always return their result.

On Windows, commands run through PowerShell. On other platforms, they run through the system shell.

`run_command` accepts optional `head_chars` and `tail_chars` parameters that control how much of the beginning and end of command output is returned to the model. Omitted values split `max_tool_output_chars` between the two sides. Explicit values override that configured limit for the command. When output is longer than the requested head and tail, Jarv retains the full result under a session-scoped `cmd_<id>` and reports the exact omitted offset and size so the model can retrieve it with `read`.

In the terminal, command output uses at most one-third of the screen height. Truncated output is biased roughly 2:1 toward the first lines, followed by the omitted-middle count and the final lines. Jarv also shows the resolved `head_chars` and `tail_chars` for each command.

`read(input, offset, size)` uses Unicode character offsets for text. `offset` defaults to 0 and `size` defaults to `max_tool_output_chars`; an explicit size is returned without generic tool-output truncation. Inputs resolve as retained command IDs, visible artifact labels, HTTP(S) URLs, or local file paths. Relative paths use the current working directory. Local and HTTP(S) PDFs with embedded text are extracted with page markers; scanned/image-only PDFs are not OCR'd. Consecutive reads requested in one model response run concurrently, with results returned in call order.

When `read` sees a direct local or HTTP(S) image (`png`, `jpeg`, `webp`, and provider-supported `gif`) and the active model advertises image input support in Jarv's cached provider/OpenRouter catalog, Jarv returns it as native multimodal tool output for that provider. Image reads ignore `offset` and `size`, are capped at 10 MiB, and are not resized or transcoded. If the active model route does not advertise image capability, Jarv returns a text result telling the model that image reads are unavailable instead of sending base64 text.

Web search and reads require no extra API key. `web_search` accepts any positive `max_results` plus a non-negative `offset`, following DuckDuckGo result pages as needed. URL reads preserve HTTP(S) hyperlinks as absolute URLs in extracted text, do not execute JavaScript, cap text and PDF responses at 2 MiB, and mark every returned page as untrusted content. Public, private, and localhost HTTP(S) addresses are supported, including custom ports.

### Command safety

Before executing a shell command, jarv can prompt you for confirmation. The `command_safety` config key controls this:

| Level | Behavior |
| --- | --- |
| `risky` (default) | Prompts for confirmation when a command matches dangerous patterns — recursive deletion, privilege escalation, network exfiltration, disk formatting, credential access, force pushes, and more. |
| `all` | Every command requires your explicit approval before running. |
| `none` | Commands run immediately with no confirmation prompt. |

Set the level in the settings menu (`jarv /settings`) or at any time with:

```bash
jarv /set command_safety risky    # default — confirm dangerous commands
jarv /set command_safety all      # confirm everything
jarv /set command_safety none     # no prompts
```

### Subagent orchestration

When the model calls `spawn`, Jarv runs N child agents in parallel. Each child operates independently — running commands, reasoning through subtasks — and terminates by calling `finish` with a detailed report and a short summary. The parent agent can then read any child's full output via `read`.

- **Parallel by default** — all children in a `spawn` call run concurrently in a thread pool.
- **Artifacts** — each child's output is stored as a named artifact. The parent (or siblings that declare a dependency) can fetch the full content.
- **Recursive** — children can themselves spawn further children, up to `max_subagent_depth` levels deep (default 4). Children are sterile by default; the parent must explicitly allow further spawning.
- **Transcript scope** — child-agent transcripts are discarded. Root history stores the parent `spawn`/`read` tool calls and returned outputs.
- **Session-scoped** — artifacts persist for the active session and are available on later prompts until you start a new session or archive.

The terminal shows a live progress panel as children run, with a green checkmark or red cross as each finishes.

## Commands

![jarv slash commands](https://github.com/user-attachments/assets/ed723bce-229e-47db-82a4-c26af2c623ad)

| Command | Description |
| --- | --- |
| `/help` | Show all commands |
| `/about` | Detailed info and examples |
| `/set <key> <value>` | Set a config value |
| `/unset <key>` | Reset a config key to default |
| `/settings` | Open the interactive settings menu |
| `/config` | Show raw config values |
| `/setup` | Run the setup wizard |
| `/new` | Start a fresh session on the next prompt |
| `/archive` | Archive session history and sidecars |
| `/sessions` | Browse sessions (interactive when in a TTY) |
| `/sessions <id>` | Load a specific session by ID prefix |
| `/history` | Show recent conversation history |
| `/undo [n]` | Remove last *n* exchanges (default 1) |
| `/redo [n]` | Restore last *n* undone exchanges (default 1) |
| `/usage` | Show token usage, cost, and context breakdown for the current session |
| `/usage day` / `/usage week` / `/usage month` | Show system-wide usage for the last 24h, 7d, or 30d |
| `/usage --all [--since 24h]` | Show system-wide usage across Jarv sessions |
| `/update` | Update Jarv to the latest version for the active install channel |

All commands work both as `jarv /command` (one-shot) and inside heads-up mode. Read-only commands (`/help`, `/about`, `/usage`, and `/config`) use a temporary display by default in interactive terminals; change `read_only_command_display` in `/settings` to print them permanently instead.

Agent tool calls have a separate `tool_call_display` setting. `auto` uses `print` for one-shot runs and `fullscreen` in heads-up mode. `print` is resize-safe and left-aligned; `fullscreen` uses bordered cards with right-aligned status.

## Sessions

Each terminal is automatically bound to its own session. Jarv identifies terminals using environment variables (`WT_SESSION`, `TERM_SESSION_ID`, `TMUX`, `STY`) with a parent-process fallback, so history persists across runs in the same terminal.

- `/new` starts a fresh session on the next prompt without archiving the current session.
- `/sessions` opens an interactive browser (arrow keys to navigate, Enter to load, `a` to archive, `d` to delete, `p` to preview, `Tab` to switch views, Ctrl+F to search).
- `/history` opens an interactive transcript where Up/Down scroll and Left/Right jump to the previous or next chat/reply.
- `/undo` and `/redo` let you step through recent exchanges.

![jarv session undo](https://github.com/user-attachments/assets/957729b0-3925-461d-a65c-25cf030ffaa6)

## Config

Settings live in `~/.jarv/config.json` (created on first run). Use `/settings` for the common controls, or edit the file directly with `/set` and `/unset`.

| Key | Default | Description |
| --- | --- | --- |
| `provider` | `"openai"` | API provider (`openai`, `anthropic`, `gemini`, `openrouter`, etc.). |
| `api_key` | `""` | Legacy single API key field (migrated to `api_keys`). |
| `api_keys` | `{}` | Per-provider API keys. Falls back to provider env vars when empty. |
| `base_url` | `""` | Custom API base URL. Overrides the provider default. |
| `model` | `"gpt-5.4-mini"` | Model name passed to the API. |
| `service_tiers` | `{}` | Per-provider processing tier: `standard`, `flex`, or `priority`. Missing providers use `standard`; unsupported tiers are not offered. |
| `reasoning_effort` | `""` | Model-supported reasoning effort. Empty uses the provider/model default; `none` explicitly disables reasoning only where supported. |
| `max_history` | `40` | Max stored history items sent as model context (item cap before token trimming). Does not delete saved history. |
| `context_budget_ratio` | `0.75` | Share of the context window used for input. |
| `context_compaction_threshold` | `0.85` | Fill ratio that triggers history compaction. |
| `context_output_reserve_ratio` | `0.15` | Context window share reserved for model output. |
| `context_window_fallback` | `128000` | Context window when model metadata is unknown. |
| `max_stdin_chars` | `200000` | Maximum piped stdin characters attached to a one-shot prompt. |
| `max_tool_output_chars` | `20000` | Maximum generic tool output characters returned to the model. It also supplies the default head/tail budget for `run_command`. |
| `disabled_tools` | `[]` | Tool names omitted from root agents and subagents. Configure these from the Tools section in `/settings`. |
| `command_timeout` | `60` | Seconds before a shell command is killed. |
| `web_timeout` | `15` | Seconds before a web search or URL read is killed. |
| `command_safety` | `"risky"` | Command confirmation level: `all` (confirm every command), `risky` (confirm dangerous commands only), `none` (no confirmation). |
| `audit` | `true` | LLM auditor for flagged commands. |
| `auditor_auto_approve` | `true` | Let the auditor auto-approve commands it deems safe. |
| `auditor_model` | `""` | Auditor model. Empty uses the active `model`. |
| `max_subagent_depth` | `4` | Maximum nesting depth for spawned subagents. |
| `subagent_thread_pool_max_workers` | `8` | Max parallel subagents per `spawn` call. |
| `check_updates` | `true` | Background update check on startup (non-blocking, throttled to once per 24h; PyPI for Python installs, GitHub Releases for standalone installs). |
| `read_only_command_display` | `"fullscreen"` | Display mode for `/help`, `/about`, `/usage`, and `/config`: temporary `fullscreen` view or permanent `print` output. |
| `tool_call_display` | `"auto"` | Tool-call layout: `auto` selects `print` for one-shot runs and `fullscreen` in heads-up mode; explicit modes override it. |
| `print_usage_after_agent` | `false` | Print a compact token usage line after each completed agent run. |
| `system_prompt` | `"You are Jarv..."` | System instructions sent with each request. |

Processing tier choices depend on the active provider. OpenAI, OpenRouter, and Gemini offer all three choices where the selected model supports them. Anthropic offers Standard and Priority; its Priority mode uses committed Priority capacity when available and otherwise falls back to Standard. Other providers remain on Standard.

## Local files

All state is stored in `~/.jarv/` (on Windows, `%USERPROFILE%\.jarv\`):

```
~/.jarv/
├── config.json                      # settings and optional API key
├── sessions.json                    # terminal → session mappings
├── sessions/
│   ├── history-<hash>.json          # conversation history
│   ├── artifacts-<hash>.json        # subagent artifacts
│   ├── reads-<hash>.json            # retained command outputs
│   ├── usage-<hash>.json            # session token usage totals
│   └── redo-<hash>.json             # undo/redo stack
├── usage.json                       # future system-wide token usage ledger
└── archive/                         # archived sessions
```

`max_history` counts stored items, not exchanges or tokens. User messages, assistant messages, reasoning items, function calls, and function call outputs each count as one item.

System-wide usage tracking starts from the version that records `~/.jarv/usage.json`; older session totals are not backfilled into time-window reports. Cost tracking is request-based and grouped by provider and processing tier. Provider-reported cost is used when available; otherwise Jarv matches the selected model to pricing from OpenRouter's public model catalog and labels the calculation as estimated. The OpenRouter catalog is refreshed when provider model choices are refreshed, and its input, cached-input, and output rates are shown with model choices. Unknown and contract-priced requests are shown separately instead of being priced as Standard.

## Dependencies

| Package | Role |
| --- | --- |
| [httpx](https://pypi.org/project/httpx/) | Direct provider API transports |
| [pypdf](https://pypi.org/project/pypdf/) | Lazy-loaded embedded-text extraction for PDF reads |
| [rich](https://pypi.org/project/rich/) | Terminal styling, live rendering, markdown |

## License

[Elastic License 2.0 (ELv2)](LICENSE) — free to use, modify, and redistribute. You may not offer jarv as a hosted/managed service.
