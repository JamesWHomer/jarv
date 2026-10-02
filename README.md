# jarv

**An AI agent that behaves like a shell tool.** Pipe into it, script it, and point it at a supported provider — cloud or local. Jarv runs commands, edits files, searches the web, and delegates work to parallel subagents.

[Documentation](https://github.com/JamesWHomer/jarv/blob/main/docs/index.md) · [CLI reference](https://github.com/JamesWHomer/jarv/blob/main/docs/cli.md) · [Releases](https://github.com/JamesWHomer/jarv/releases)

```bash
git diff | jarv review this patch
jarv what process is using port 8080?
jarv --no-tools "Explain this error"
jarv
```

![jarv interactive session](https://github.com/JamesWHomer/jarv/releases/download/readme-assets/hero.webp)

## Why Jarv?

- **Scriptable:** use one-shot prompts, piped input, and JSON output in scripts and pipelines.
- **Multi-provider:** choose OpenAI, Anthropic, Gemini, OpenRouter, Groq, DeepSeek, Together, or Fireworks, or run local models with Ollama, LM Studio, or vLLM.
- **Lightweight:** install a standalone binary with its Python runtime bundled, or a Python package with four direct runtime dependencies. No Node runtime.
- **Persistent sessions:** keep context per terminal, browse conversation history, and fork earlier prompts. Undo and redo change conversation history; they do not reverse tool actions.

## Install

Choose one installation method.

**Windows PowerShell:**

```powershell
irm https://github.com/JamesWHomer/jarv/releases/latest/download/install.ps1 | iex
```

**macOS or Linux:**

```bash
curl -fsSL https://github.com/JamesWHomer/jarv/releases/latest/download/install.sh | sh
```

**Python 3.10+ with uv:**

```bash
uv tool install jarv
```

`pipx install jarv` and `pip install jarv` are also supported. See the [installation guide](https://github.com/JamesWHomer/jarv/blob/main/docs/getting-started.md) for API keys, updates, and uninstalling.

## First run

```bash
jarv /setup
jarv --no-tools "Say hello"
jarv
```

The setup wizard selects a provider, API key when needed, and model. The one-shot example checks the connection without enabling tools; `jarv` by itself opens an interactive session. Local providers need a running model server and do not require an API key.

## Documentation

The Markdown guides in [`docs/`](https://github.com/JamesWHomer/jarv/tree/main/docs) are the main source of detailed documentation:

- [Getting started](https://github.com/JamesWHomer/jarv/blob/main/docs/getting-started.md) — installation, provider setup, and first use.
- [Usage](https://github.com/JamesWHomer/jarv/blob/main/docs/usage.md) — one-shot prompts, pipelines, and interactive mode.
- [Tools and safety](https://github.com/JamesWHomer/jarv/blob/main/docs/tools.md) — command approval, file access, and subagents.
- [Sessions](https://github.com/JamesWHomer/jarv/blob/main/docs/sessions.md) — conversation history, branches, and undo/redo.
- [Reference](https://github.com/JamesWHomer/jarv/blob/main/docs/cli.md) — CLI flags, [slash commands](https://github.com/JamesWHomer/jarv/blob/main/docs/commands.md), and [configuration](https://github.com/JamesWHomer/jarv/blob/main/docs/configuration.md).

## Development

See the [development guide](https://github.com/JamesWHomer/jarv/blob/main/docs/development.md) for running tests and the [documentation guide](https://github.com/JamesWHomer/jarv/blob/main/docs/documentation.md) for local previews and Read the Docs setup.

## License

[MIT License](https://github.com/JamesWHomer/jarv/blob/main/LICENSE).
