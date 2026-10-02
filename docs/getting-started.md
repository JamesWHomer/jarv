# Getting started

Jarv can be installed as a standalone binary or as a Python package. After installing, run `jarv /setup` to choose a provider, enter an API key, and pick a model.

```powershell
irm https://github.com/JamesWHomer/jarv/releases/latest/download/install.ps1 | iex
```

```bash
curl -fsSL https://github.com/JamesWHomer/jarv/releases/latest/download/install.sh | sh
uv tool install jarv
pipx install jarv
pip install jarv
```

Python package installs require **Python 3.10+**. Use `jarv /setup key` to save the active provider's key, or set its environment variable: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `TOGETHER_API_KEY`, or `FIREWORKS_API_KEY`. Saved per-provider keys take precedence over the legacy `api_key` field and environment variables. Local providers do not require an API key.

Choose one install method. The standalone scripts install to `%LOCALAPPDATA%\Programs\Jarv` on Windows and `~/.local/bin` on macOS/Linux. The Windows script updates your user PATH; the shell script prints a PATH command if needed. Release builds cover x86-64 and ARM64 on Windows, macOS, and Linux (the Linux ARM asset is named `aarch64`).

To upgrade:

```bash
jarv /update
```

Scoop, WinGet, and Homebrew installs show the owning package manager's update command to run after exiting Jarv. `/update` installs the update for direct standalone and non-editable Python installs; background checks only notify you. Editable source installs are left untouched.

![jarv update demo](https://github.com/JamesWHomer/jarv/releases/download/readme-assets/update.webp)

## Uninstall

Jarv can detect how it was installed and either uninstall the standalone binary or show the exact package-manager command to run:

```bash
jarv /uninstall
```

The standalone uninstall scripts also work when the binary is unavailable:

```powershell
irm https://github.com/JamesWHomer/jarv/releases/latest/download/uninstall.ps1 | iex
```

```bash
curl -fsSL https://github.com/JamesWHomer/jarv/releases/latest/download/uninstall.sh | sh
```

Package-manager commands are `winget uninstall JamesWHomer.Jarv`, `brew uninstall jarv`, `scoop uninstall jarv`, `uv tool uninstall jarv`, `pipx uninstall jarv`, or `python -m pip uninstall jarv`.

With `jarv /uninstall`, your data stays in `~/.jarv`; add `--purge` to remove it and cached clipboard images too. Add `--yes` to skip confirmation when deleting the standalone binary or purging data (required for those actions in non-interactive shells). For a package-manager install, `--purge` deletes data immediately after confirmation; you still run the printed uninstall command yourself.

The standalone scripts remove the binary directly and ask separately before purging data; they do not accept `--yes`. For `uninstall.ps1`, parameters can't pass through `| iex` — use:

```powershell
& ([ScriptBlock]::Create((irm https://github.com/JamesWHomer/jarv/releases/latest/download/uninstall.ps1))) -Purge
```

For the shell script, download it before requesting a purge so its confirmation can read terminal input:

```bash
curl -fsSL https://github.com/JamesWHomer/jarv/releases/latest/download/uninstall.sh -o uninstall-jarv.sh
sh uninstall-jarv.sh --purge
```

If you used a custom installation directory, pass `-InstallDir PATH` to the PowerShell script or `--dir PATH` to the shell script.

## First run

```bash
jarv /setup
jarv --no-tools "Say hello"
jarv
```

The setup wizard selects a provider, API key when needed, and model. The one-shot example checks the connection without enabling tools. Running `jarv` by itself opens an interactive session.

For local models, choose `ollama`, `lm_studio`, or `vllm` in setup and make sure your local model server is running. Use `jarv /setup base_url` if it listens at a custom address.

Continue with [Usage](usage.md), or read about [models and providers](models.md).
