# Tools and safety

Jarv uses a multi-provider tool-calling agent loop (OpenAI Responses API, Anthropic Messages, Gemini, and OpenAI-compatible endpoints). The root model can call six tools:

| Tool | Purpose |
| --- | --- |
| `run_command` | Execute a shell command (and, with interactive commands enabled, answer its stdin prompts) |
| `web_search` | Search the web through DuckDuckGo's public HTML endpoint |
| `read` | Page through command output, artifacts, URLs, or local files |
| `edit` | Make an exact string replacement in an existing text file |
| `spawn` | Fan out work to parallel subagents, each with their own tool access |
| `ask_user` | Ask you a question and wait for a reply |

Each tool can be enabled or disabled from `jarv /settings`. Disabled tools are not sent to the model and are also unavailable to spawned subagents. The subagent-only `finish` tool remains enabled so child agents can always return their result.

On Windows, noninteractive commands reuse a PowerShell process per agent to avoid repeated startup. Each command gets a fresh execution context: working directory and environment carry forward, while ordinary variables, functions, and preferences reset. Output appears while commands run. The first command pays the startup cost; subsequent commands reuse the process. Cancellation, timeouts, or a crashed worker cause the next command to start a new one from the last saved directory and environment.

Interactive commands, explicit shell flow control (`exit`, `return`, `break`, `continue`), `Add-Type`, and explicit background launches such as `Start-Process` use the fresh-process runner. Jarv also falls back if the worker cannot start. Unexpected child processes left in a reusable worker are terminated before it is replaced; use `-c persistent_shell=false` for scripts that intentionally leave background processes running. A command already sent to a worker is never automatically retried after a failure. Turn **Reuse PowerShell** off in `/settings` to always use fresh processes.

On Windows, commands run through PowerShell. On other platforms, they run through the system shell.

`run_command` returns a head and tail of the output, sized by its `head_chars`/`tail_chars` arguments (omitted values split `max_tool_output_chars` between the two sides, with a combined ceiling of 200,000 characters). When this view is truncated, Jarv retains the captured output under a session-scoped `cmd_<id>` for later `read` calls. Capture keeps at most 2,000,000 characters per stream, preserving the head and tail; text dropped at this limit cannot be recovered with `read`. Retention is also bounded to 128 outputs and 8,000,000 total characters, evicting the oldest entries when needed.

Interactive commands are an experimental feature, disabled by default; turn them on with the **Interactive commands** toggle in `jarv /settings` (`interactive_commands`). While they are off, every command runs to completion or is killed at `command_timeout`, and nothing can be typed into a process that blocks on a prompt.

With them on, a command that stays alive after its output goes idle is treated as waiting for stdin: the model's next response is sent to the process instead of printed as chat, and the loop repeats until the command exits or is cancelled. Each step shows only the new output since the previous interaction. During this loop, `command_timeout` becomes a check-in interval rather than a kill timer — Jarv asks the model what to do next instead of terminating the process.

By default, tool output uses a line budget of one-third of the terminal height in `print` layout and one-half in `fullscreen` layout, with a minimum of three lines. Set `tool_output_display_lines` to an integer of at least 3 to pin that budget in both layouts. This controls display only; `max_tool_output_chars` controls the model's default output budget. Jarv also shows the resolved `head_chars` and `tail_chars` for each command.

`read(input, offset, size)` pages through retained command output, artifacts, HTTP(S) URLs, and local files using Unicode character offsets, with at most 200,000 characters per page. PDFs with embedded text are extracted with page markers (scanned/image-only PDFs are not OCR'd). Local PDFs are limited to 20 MiB; extraction stops at 1,000 pages or 2,000,000 text characters. Consecutive `read` and `web_search` calls in one model response are scheduled concurrently, including mixed batches; results return to the model in call order. DuckDuckGo requests share a queue across all agents in the process; URL reads remain parallel. Commands and edits run sequentially within each agent.

Image reads (`png`, `jpeg`, `webp`, and provider-supported `gif`) are returned as native multimodal input when Jarv recognizes image support for the selected model and route. Images are limited to 10 MiB and ignore `offset`/`size`. Local providers and OpenRouter's `auto`/`free` routes are currently excluded; Gemini image tool results require a Gemini 3 model and do not accept GIF. Unsupported routes return a text notice.

Web search and URL reads need no extra API key. `web_search` accepts 1–20 results per call and a non-negative result offset. URL reads preserve links as absolute URLs, don't execute JavaScript, and mark fetched pages as untrusted content. Text and PDF web responses are capped at 2 MiB; images use the 10 MiB limit above. These limits apply to both transferred and decompressed data. Gzip and deflate compression are supported; other content encodings return a tool error.

URL requests have one `web_timeout` budget covering connection setup, redirects, and the entire response transfer, including responses that keep sending small chunks. Expiry reports a timeout without cancelling sibling tools. JSON is pretty-printed only within bounded input, nesting, and output limits; otherwise its original text remains available through normal read pagination.

DuckDuckGo requests run one at a time, spaced at least `web_search_interval` seconds apart (default 1). Each search permits up to two retries for network failures, timeouts, HTTP 429, and temporary server errors (500/502/503/504), using delays of about 2 then 5 seconds with jitter. A valid `Retry-After` delay takes precedence and pauses other searches too. A human-verification page pauses DuckDuckGo searches in the process for at least 60 seconds (longer if requested by `Retry-After`) and returns a clear blocked message; new calls during the pause fail promptly. Genuine empty results succeed with zero matches; unrecognized pages report an unexpected-response error.

The `web_timeout` budget (default 15 seconds) includes waiting, retries, and all pages. Cancellation interrupts queueing, backoff, and active requests. Searches fetch at most `web_search_max_pages` result pages (default 5), with up to two additional retry attempts. Results gathered before a later failure or limit are returned with a **Partial results** notice; a failure before reaching the requested offset reports its actual cause. Searches are not cached, so requesting an offset walks earlier pages again within these limits.

`edit` accepts existing UTF-8 files up to 5,000,000 bytes and checks that the replacement will stay within the same limit before preparing or writing it. Oversized replacements leave the file unchanged. Diff previews are capped at 60 lines and 12,000 characters; result previews are capped at 4,000 characters. Use `read` to inspect more of the edited file.

## Project context

At the start of each request, jarv looks for a project instructions file — `JARV.md`, then `AGENTS.md`, then `CLAUDE.md` — starting in the working directory and walking up to the git root (outside a repository, only the working directory is checked). The first match is injected into the system prompt, together with the current git branch, clean/dirty status, and the last five commits, so the model starts aware of the project it is in.

Disable with `project_context`; cap the injected file size with `project_context_max_chars`. Git info is skipped silently when git is not installed or the directory is not a repository.

## Command safety

Before executing a shell command, jarv can prompt you for confirmation. The `command_safety` config key controls this:

Approval previews show every script line in execution order, including blank lines, assignments, and directory changes. Risky lines are highlighted without hiding the surrounding code. In fullscreen mode the complete preview is scrollable; inline auditor status appears after the full script has been printed into terminal scrollback.

Terminal controls embedded in commands, output, diffs, replies, and saved transcripts are displayed as visible escapes. This affects human-readable display only; original tool arguments, stored content, and JSON output are preserved.

| Level | Behavior |
| --- | --- |
| `risky` (default) | Flags commands matching dangerous patterns — recursive deletion, privilege escalation, network exfiltration, disk formatting, credential access, force pushes, and more. The enabled auditor can auto-approve them; otherwise Jarv asks for confirmation. |
| `all` | Every command requires your explicit approval before running, even when the auditor recommends approval. |
| `none` | Commands run immediately with no confirmation prompt. |

The same levels gate `edit` calls: risky edits (files outside the working directory, hidden files, secrets, system paths) show a diff preview for approval under `risky`, every edit does under `all`. The LLM auditor applies only to shell commands; flagged edits require human approval.

Set the level in the settings menu (`jarv /settings`) or at any time with:

```bash
jarv /set command_safety risky    # default — review flagged commands; auditor may auto-approve
jarv /set command_safety all      # confirm everything
jarv /set command_safety none     # no prompts
```

## Subagent orchestration

When the model calls `spawn`, Jarv runs N child agents in parallel. Each child operates independently — running commands, reasoning through subtasks — and terminates by calling `finish` with a detailed report and a short summary. The parent agent can then read any child's full output via `read`.

- **Parallel by default** — a `spawn` call accepts up to 16 children, with at most `subagent_thread_pool_max_workers` running at once (default 8); the rest queue.
- **Unique artifact labels** — use a new child label across successive and nested `spawn` calls in a session. Colliding batches are rejected before any child starts, preserving earlier reports.
- **Artifacts** — each successful child's output is stored as a named artifact. The parent can read it or pass access to later children through `deps`. Dependencies must already be visible to the parent; children cannot depend on another child in the same batch.
- **Recursive** — children can themselves spawn further children, up to `max_subagent_depth` levels deep (default 4). Children are sterile by default; the parent must explicitly allow further spawning.
- **Bounded** — a `spawn` batch cancels unfinished children after `subagent_timeout` seconds (default 600) instead of waiting forever.
- **Transcript scope** — child-agent transcripts are discarded. Root history stores the parent `spawn`/`read` tool calls and returned outputs.
- **Session-scoped** — artifacts persist with their session across prompts. Starting a new session leaves them with the old session; archiving moves them with its history, and restoring brings them back. Incognito artifacts stay in memory for the current run only.

The terminal shows a live progress panel as children run, with a green checkmark or red cross as each finishes.
