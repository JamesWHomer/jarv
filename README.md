# jarv

Super simple OpenAI-powered CLI agent.

Jarv uses the OpenAI Responses API, keeps a short local conversation history, and can run shell commands when the model decides they are useful. Command output is shown in your terminal and sent back to the model so it can continue the task.

```bash
jarv
jarv whats the meaning of life?
jarv what did the fox say?
jarv bring up the man page for the uhhh exponent function?
jarv commit all these files
```

## Heads-Up Mode

Run `jarv` with no prompt to start heads-up mode: an interactive prompt loop for repeated questions and tasks.

```bash
jarv
```

In heads-up mode:

- Type a prompt and press Enter to send it.
- Keep sending prompts without rerunning `jarv`.
- Commands start with `/` (e.g. `/clear`, `/history`). Type `/help` to see all commands.
- Type `exit`, `quit`, or `/exit`, or press Ctrl+C, to leave.

## Install

Requirements:

- Python 3.10+
- An OpenAI API key

Install directly from GitHub with pip:

```bash
pip install git+https://github.com/JamesWHomer/jarv.git
```

Then add your OpenAI API key:

```bash
jarv /set api_key YOUR_OPENAI_API_KEY
```

Alternatively, clone the repo and install it locally first:

```bash
git clone https://github.com/JamesWHomer/jarv.git
cd jarv
pip install -e .
jarv /set api_key YOUR_OPENAI_API_KEY
```

You can verify the install with:

```bash
jarv /help
```

The first run that needs config will create `~/.jarv/config.json` (on Windows, `%USERPROFILE%\.jarv\config.json`). You can also set the `OPENAI_API_KEY` environment variable instead of saving the key in Jarv config.

To upgrade later:

```bash
jarv /update
```

or:

```bash
pip install --upgrade git+https://github.com/JamesWHomer/jarv.git
```

## Config (`~/.jarv/config.json`)

Default config:

```json
{
  "api_key": "",
  "model": "gpt-5.4-mini",
  "reasoning_effort": "",
  "max_history": 40,
  "command_timeout": 60,
  "max_subagent_depth": 4,
  "subagent_thread_pool_max_workers": 8,
  "system_prompt": "You are Jarv, a helpful CLI assistant..."
}
```

Config notes:

- `api_key` is read from config first. If it is empty, Jarv uses `OPENAI_API_KEY`.
- `model` is the OpenAI model name to use.
- `reasoning_effort` is sent as `{ "effort": "..." }` when non-empty. Leave it empty to disable.
- `max_history` is the number of recent messages Jarv keeps as context.
- `command_timeout` is the number of seconds before a shell command is killed.
- `max_subagent_depth` is the maximum recursion depth for spawned subagents. The root agent is depth 0; each `spawn` call adds one level. Defaults to 4.
- `subagent_thread_pool_max_workers` is the maximum number of subagents that can run in parallel within a single `spawn` call. Defaults to 8.
- `system_prompt` is sent to the model along with basic system info like OS, current working directory, shell, and session context.

You can edit the JSON file directly or use `jarv /set` / `jarv /unset`.

## Commands

| Command | Description |
| --- | --- |
| `jarv` | Start heads-up mode: an interactive prompt loop for repeated prompts |
| `jarv <anything>` | Ask Jarv a question or give it a task |
| `jarv /set <key> <value>` | Set a config value |
| `jarv /unset <key>` | Reset a config key to its default |
| `jarv /clear` | Archive this terminal's session and start a fresh one |
| `jarv /load` | Load the most recently used session into this terminal |
| `jarv /load <id>` | Load a specific session into this terminal |
| `jarv /history` | Show recent conversation history |
| `jarv /config` | Show current settings |
| `jarv /update` | Update Jarv to the latest version from GitHub |
| `jarv /cleanup` | Delete orphaned session files from `~/.jarv/sessions/` |
| `jarv /about` | Show detailed information about Jarv |
| `jarv /help` | Show help (`jarv help` also works) |

## Files

Jarv stores local state in `~/.jarv/`:

- `config.json` — settings and optional API key
- `sessions.json` — terminal/session metadata
- `last_sha.txt` — last seen GitHub commit SHA for update checks
- `sessions/history-<hash>.json` — conversation history for each session
- `sessions/artifacts-<hash>.json` — artifact store for each session
- `archive/` — sessions archived by `jarv /clear`

## History and sessions

Each terminal is bound to exactly one session at a time. By default a new terminal gets its own session (the id is derived from a terminal fingerprint). History for a session lives in `~/.jarv/sessions/history-<hash>.json`.

Jarv detects terminals from environment values such as `WT_SESSION`, `TERM_SESSION_ID`, `TMUX`, or `STY`, with a parent-process fallback.

- `jarv /clear` archives the current session's history and artifacts, then starts a fresh session on the next prompt.
- `jarv /load` binds this terminal to the most recently used session.
- `jarv /load <id>` binds this terminal to a specific session id.
- `jarv /cleanup` removes session files in `~/.jarv/sessions/` that no longer have a matching entry in `sessions.json`.

## Subagent orchestration

For complex tasks, the model can fan out work to parallel subagents using the `spawn` tool. Each subagent runs the same agent loop independently, can run shell commands, and terminates by calling `finish` with a full report (`longform`) and a short summary (`tldr`).

- **Parallel execution** — all children in a single `spawn` call run concurrently in a thread pool.
- **Artifacts** — each child's output is stored as a named artifact. The parent (and any sibling that declares a dependency) can fetch the full content via `read_artifact`.
- **Depth limit** — subagents can themselves spawn children up to `max_subagent_depth` levels deep (default 4). By default children are *sterile* (cannot spawn further) unless the parent explicitly sets `sterile: false`.
- **Scoped per query** — the artifact store is created fresh for each top-level query; artifacts do not persist across separate `jarv` invocations.

The terminal shows a summary of each spawn batch as children finish, with a green ✓ for success and red ✗ for failure.

## Notes

- Jarv may run commands if the model requests them. Review tasks you give it accordingly.
- On Windows, commands are executed through PowerShell. On other platforms, commands run through the system shell.
