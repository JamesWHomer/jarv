# Usage

## One-shot mode

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

![jarv one-shot and piped stdin](https://github.com/JamesWHomer/jarv/releases/download/readme-assets/oneshot.webp)

## Heads-up mode

Run `jarv` with no arguments to enter an interactive prompt loop.

```
jarv> what files changed today?
jarv> now run the tests
jarv> /history
jarv> /new
```

- Type a prompt and press Enter.
- Slash commands start with `/` — type `/help` to list them.
- Ctrl+V (or Alt+V if your terminal owns Ctrl+V, e.g. Windows Terminal) attaches a copied image or image file as an `[Image #N]` chip for image-capable models.
- During a response, Esc or Ctrl+C stops further work, checkpoints the turn in history/context, and restores the prompt for editing. Use `/undo` to remove the turn.
- At the prompt, Esc or Ctrl+C clears existing text; press either again on an empty prompt to exit.
- You can also exit with `exit`, `quit`, `/exit`, or `/quit`.

See the [CLI reference](cli.md) for invocation flags and machine-readable output, and [slash commands](commands.md) for interactive controls.

## Turn summaries

In `/settings` → Turn summary, enable **Turn summary** to show one line after Jarv finishes responding to your prompt in one-shot and heads-up mode. Tool calls and internal continuations belong to that same turn; the summary appears once at the end. The master switch defaults to off. Choose what appears with the individual toggles in the same section:

- **Token counts**: input, output, and total tokens across the turn's successful model requests.
- **Cached tokens** and **Reasoning tokens**: provider-reported totals across the turn, independently selectable.
- **Output speed**: generation tok/s for the final response.
- **Model time**: total time spent on successful model requests during the turn, excluding tool execution.
- **Session tokens** and **Session cost**: running saved-session totals, including subagent and auditor usage.

Token counts, cache, reasoning, speed, and time are selected by default; session tokens and cost are off. Turning the master switch off preserves your selections. You can also configure these with `/set`:

```text
/set turn_summary true
/set turn_summary_cache false
/set turn_summary_session true
/set turn_summary_cost true
```

The summary adds up the main agent's successful model requests, using provider token counts, including reasoning tokens where reported. Input context sent again after a tool call counts again in the turn's input total. **Model time** adds each successful request's elapsed time, including connection, queueing, input processing, and reasoning. Replayed requests are timed separately from failed attempts; tool execution is excluded.

**Output speed** measures the final response's generation and is labelled accordingly when a turn uses multiple model requests. It prefers provider-reported generation timing and labels it `tok/s (server)`. For example, [Groq reports completion time](https://console.groq.com/docs/api-reference) and [llama.cpp-compatible servers can report generation timings](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md). Timing metadata must actually be included in the response; many APIs omit it.

Without server timing, a text-only response can show `~… tok/s (stream)`. This estimate measures arrivals from the first nonempty text chunk to the last, excluding the initial wait and final metadata delivery. It uses provider-reported output counts with reasoning tokens removed, and apportions the first chunk's share by text length because chunks do not identify individual tokens. Network buffering and chunk sizes still affect it, so it is not an exact server decoding rate. A single chunk, a window shorter than 100 ms, fewer than one estimated token after the first chunk, mixed text/tool output, incomplete recovered text, or counts that cannot separate reasoning make this estimate unavailable. Server timing can still provide a rate for those responses.

When the provider omits token usage, available text/context counts are estimated and labelled. These heuristic counts are not used to claim a generation speed.

When using an explicit one-shot output format (`--output-format`) or `--verbose`, summaries go to stderr so stdout keeps its requested format. `--quiet` suppresses summaries. Turn token counts, model time, and output speed also work in incognito mode; saved-session token and cost totals are unavailable there.

The old **Model turn stats** and **Print usage** settings migrate automatically to this single summary, preserving explicitly configured new options. See [configuration](configuration.md) for all field keys.
