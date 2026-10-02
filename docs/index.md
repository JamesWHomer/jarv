# Jarv

**An AI agent that behaves like a shell tool.** Pipe into it, script it, and point it at a supported provider — cloud or local. Jarv runs commands, edits files, searches the web, and fans work out to parallel subagents, from a standalone binary or a Python package with four direct runtime dependencies.

```bash
git diff | jarv review this patch       # pipe anything in, like any other unix tool
jarv what process is using port 8080?   # one-shot: answers (running commands if needed), then exits
jarv commit all these files             # let it run commands to do the job
jarv                                    # start an interactive session
```

![jarv interactive session](https://github.com/JamesWHomer/jarv/releases/download/readme-assets/hero.webp)

## Why jarv?

The official vendor CLIs (Claude Code, Codex CLI, Gemini CLI) are strong interactive coding agents for their own models. Jarv makes different trade-offs:

- **Scriptable first.** One-shot mode and piped stdin are core, not an afterthought: `rg TODO . | jarv group these by subsystem` works the way you'd expect a Unix tool to. Use it in scripts, aliases, and pipelines.
- **Any model, including local.** OpenAI, Anthropic, Gemini, OpenRouter, Groq, DeepSeek, Together, and Fireworks — or fully local with Ollama, LM Studio, and vLLM. Switch per run with `--provider`/`-m`. No lock-in, no subscription.
- **Lightweight.** A standalone binary with its Python runtime bundled, or a small Python package with four direct runtime dependencies. No Node runtime.
- **A terminal agent, not just a coding agent.** Each terminal window is bound to its own persistent session, so jarv doubles as a general assistant that remembers context per window — with undo/redo, a forkable history tree, and usage tracking built in.

## Start here

- [Install Jarv and choose a provider](getting-started.md).
- [Use prompts, pipelines, and interactive mode](usage.md).
- [Look up CLI flags](cli.md) or [slash commands](commands.md).
- [Configure tools and command approval](tools.md#command-safety).
- [Browse and manage sessions](sessions.md).

Jarv is open source under the [MIT License](https://github.com/JamesWHomer/jarv/blob/main/LICENSE).
