# jarv

Super simple OpenAI-powered CLI agent.

```bash
jarv whats the meaning of life?
jarv what did the fox say?
jarv bring up the man page for the uhhh exponent function?
jarv commit all these files
```

Jarv can run shell commands when the model decides they are useful. Command output is shown in the terminal and returned to the model for follow-up.

## Install

```bash
pip install -e .
```

First run creates `~/.jarv/config.json`; add your OpenAI API key there, or set the `OPENAI_API_KEY` environment variable.

## Config (`~/.jarv/config.json`)

```json
{
  "api_key": "sk-...",
  "model": "gpt-4o-mini",
  "reasoning_effort": "",
  "max_history": 40,
  "command_timeout": 60,
  "system_prompt": "You are Jarv, a helpful CLI assistant..."
}
```

Config notes:

- `reasoning_effort` may be `low`, `medium`, `high`, or empty to disable. It is only sent for reasoning-capable models.
- `command_timeout` is the number of seconds before a shell command is killed.
- `api_key` in the config is overridden by `OPENAI_API_KEY` if the config value is empty.

## Commands

| Command | Description |
|---|---|
| `jarv <anything>` | Ask Jarv a question or give it a task |
| `jarv set <key> <value>` | Set a config value |
| `jarv unset <key>` | Reset a config key to its default |
| `jarv clear` | Clear conversation history |
| `jarv history` | Print recent conversation |
| `jarv config` | Show current settings |
| `jarv update` | Update Jarv from GitHub |
| `jarv help` | Show help |
