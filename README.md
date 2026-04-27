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

Run `jarv` with no prompt to start heads-up mode, where you can keep sending prompts without rerunning the command each time.

In heads-up mode, type a prompt and press Enter. Type `exit` or `quit`, or press Ctrl+C, to leave.

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
jarv set api_key YOUR_OPENAI_API_KEY
```

Alternatively, clone the repo and install it locally first:

```bash
git clone https://github.com/JamesWHomer/jarv.git
cd jarv
pip install -e .
jarv set api_key YOUR_OPENAI_API_KEY
```

You can verify the install with:

```bash
jarv help
```

The first run that needs config will create `~/.jarv/config.json` (on Windows, `%USERPROFILE%\.jarv\config.json`). You can also set the `OPENAI_API_KEY` environment variable instead of saving the key in Jarv config.

To upgrade later:

```bash
jarv update
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
  "system_prompt": "You are Jarv, a helpful CLI assistant..."
}
```

Config notes:

- `api_key` is read from config first. If it is empty, Jarv uses `OPENAI_API_KEY`.
- `model` is the OpenAI model name to use.
- `reasoning_effort` is sent as `{ "effort": "..." }` when non-empty. Leave it empty to disable.
- `max_history` is the number of recent messages Jarv keeps as context and saves in `~/.jarv/history.json`.
- `command_timeout` is the number of seconds before a shell command is killed.
- `system_prompt` is sent to the model along with basic system info like OS, current working directory, and shell.

You can edit the JSON file directly or use `jarv set` / `jarv unset`.

## Commands

| Command | Description |
| --- | --- |
| `jarv` | Start heads-up mode for repeated prompts |
| `jarv <anything>` | Ask Jarv a question or give it a task |
| `jarv set <key> <value>` | Set a config value |
| `jarv unset <key>` | Reset a config key to its default |
| `jarv clear` | Clear conversation history |
| `jarv history` | Show recent conversation history |
| `jarv config` | Show current settings |
| `jarv update` | Update Jarv to the latest version from GitHub |
| `jarv about` | Show detailed information about Jarv |
| `jarv help` | Show help |

## Files

Jarv stores local state in `~/.jarv/`:

- `config.json` - settings and optional API key
- `history.json` - recent conversation history
- `last_sha.txt` - last seen GitHub commit SHA for update checks

## Notes

- Jarv may run commands if the model requests them. Review tasks you give it accordingly.
- On Windows, commands are executed through PowerShell. On other platforms, commands run through the system shell.
