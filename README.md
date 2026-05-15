# jarv

An OpenAI-powered CLI agent that can run shell commands, fan out work to parallel subagents, and keep track of conversation history across terminal sessions.

```bash
jarv                                    # start an interactive session
jarv whats the meaning of life?         # one-shot question
jarv commit all these files             # let it run commands to do the job
jarv refactor the auth module           # complex tasks get split across subagents
```

## Install

Requires **Python 3.10+** and an **OpenAI API key**.

```bash
pip install jarv
jarv /setup
```

The setup wizard will walk you through entering your API key and choosing a model. The key can also be set via the `OPENAI_API_KEY` environment variable or `jarv /set api_key ...`.

To upgrade:

```bash
jarv /update
```

## Usage

### One-shot mode

Pass a prompt as arguments. Jarv answers (running commands if needed) and exits.

```bash
jarv what process is using port 8080?
jarv find all TODO comments in src/
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
- Exit with `exit`, `quit`, `/exit`, or Ctrl+C.

### Flags

Flags override config values for a single run and work in both one-shot and heads-up mode.

| Flag | Short | Description |
| --- | --- | --- |
| `--model MODEL` | `-m` | Override the model (e.g. `gpt-4o`) |
| `--effort EFFORT` | `-e` | Override reasoning effort (`low` / `medium` / `high`) |
| `--timeout SECONDS` | | Override command timeout in seconds |
| `--system PROMPT` | `-s` | Override the system prompt |
| `--new` | | Start a fresh session (ignore prior history, but still save) |
| `--incognito` | | Don't load or save session history |
| `--version` | | Print the version and exit |

```bash
jarv -m gpt-4o "summarise this repo"
jarv --effort high "refactor the auth module"
jarv --new "start fresh without prior context"
jarv --incognito "one-off task, leave no trace"
jarv --timeout 120 --system "You are a poet" "write me a haiku"
```

## How it works

Jarv wraps the OpenAI Responses API with a tool-calling agent loop. The model can call three tools:

| Tool | Purpose |
| --- | --- |
| `run_command` | Execute a shell command and return stdout, stderr, and exit code |
| `spawn` | Fan out work to parallel subagents, each with their own tool access |
| `read_artifact` | Retrieve the full output of a completed subagent |

On Windows, commands run through PowerShell. On other platforms, they run through the system shell.

### Subagent orchestration

When the model calls `spawn`, Jarv runs N child agents in parallel. Each child operates independently — running commands, reasoning through subtasks — and terminates by calling `finish` with a detailed report and a short summary. The parent agent can then read any child's full output via `read_artifact`.

- **Parallel by default** — all children in a `spawn` call run concurrently in a thread pool.
- **Artifacts** — each child's output is stored as a named artifact. The parent (or siblings that declare a dependency) can fetch the full content.
- **Recursive** — children can themselves spawn further children, up to `max_subagent_depth` levels deep (default 4). Children are sterile by default; the parent must explicitly allow further spawning.
- **Scoped per query** — the artifact store resets with each new top-level prompt.

The terminal shows a live progress panel as children run, with a green checkmark or red cross as each finishes.

## Commands

| Command | Description |
| --- | --- |
| `/help` | Show all commands |
| `/about` | Detailed info and examples |
| `/set <key> <value>` | Set a config value |
| `/unset <key>` | Reset a config key to default |
| `/config` | Show current settings |
| `/setup` | Run the setup wizard |
| `/new` | Start a fresh session on the next prompt |
| `/archive` | Archive session history and artifacts |
| `/sessions` | Browse sessions (interactive when in a TTY) |
| `/sessions <id>` | Load a specific session by ID prefix |
| `/history` | Show recent conversation history |
| `/undo [n]` | Remove last *n* exchanges (default 1) |
| `/redo [n]` | Restore last *n* undone exchanges (default 1) |
| `/usage` | Show token usage, cost, and context breakdown |
| `/update` | Update Jarv to the latest version |

All commands work both as `jarv /command` (one-shot) and inside heads-up mode.

## Sessions

Each terminal is automatically bound to its own session. Jarv identifies terminals using environment variables (`WT_SESSION`, `TERM_SESSION_ID`, `TMUX`, `STY`) with a parent-process fallback, so history persists across runs in the same terminal.

- `/new` starts a fresh session on the next prompt without archiving the current session.
- `/sessions` opens an interactive browser (arrow keys to navigate, Enter to load, `a` to archive, `d` to delete, `p` to preview, `Tab` to switch views, Ctrl+F to search).
- `/undo` and `/redo` let you step through recent exchanges.

## Config

Settings live in `~/.jarv/config.json` (created on first run). Edit the file directly or use `/set` and `/unset`.

| Key | Default | Description |
| --- | --- | --- |
| `api_key` | `""` | OpenAI API key. Falls back to `OPENAI_API_KEY` env var if empty. |
| `model` | `"gpt-5.4-mini"` | Model name passed to the API. |
| `reasoning_effort` | `""` | Reasoning effort level. Leave empty to disable. |
| `max_history` | `40` | Number of recent messages kept as context. |
| `command_timeout` | `60` | Seconds before a shell command is killed. |
| `max_subagent_depth` | `4` | Maximum nesting depth for spawned subagents. |
| `subagent_thread_pool_max_workers` | `8` | Max parallel subagents per `spawn` call. |
| `check_updates` | `true` | Background update check on startup (non-blocking, throttled to once per 24h). |
| `system_prompt` | `"You are Jarv..."` | System instructions sent with each request. |

## Local files

All state is stored in `~/.jarv/` (on Windows, `%USERPROFILE%\.jarv\`):

```
~/.jarv/
├── config.json                      # settings and optional API key
├── sessions.json                    # terminal → session mappings
├── sessions/
│   ├── history-<hash>.json          # conversation history
│   ├── artifacts-<hash>.json        # subagent artifacts
│   ├── usage-<hash>.json            # token usage totals
│   └── redo-<hash>.json             # undo/redo stack
└── archive/                         # archived sessions
```

## Dependencies

| Package | Role |
| --- | --- |
| [openai](https://pypi.org/project/openai/) | Responses API client |
| [rich](https://pypi.org/project/rich/) | Terminal styling, live rendering, markdown |
| [litellm](https://pypi.org/project/litellm/) | Token counting, model pricing, context window metadata |

## License

Elastic License 2.0
