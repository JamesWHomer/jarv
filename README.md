Super simple CLI agent.

`jarv whats the meaning of life?`

`jarv what did the fox say?`

`jarv bring up the man page for the uhhh exponent function?`

`jarv delete all those photos of annoying orange`

`jarv commit all these files`

## Install

```bash
pip install -e .
```

First run creates `~/.jarv/config.json` — add your OpenAI API key there,
or set the `OPENAI_API_KEY` environment variable.

## Config (`~/.jarv/config.json`)

```json
{
  "api_key": "sk-...",
  "model": "gpt-4o-mini",
  "max_history": 40,
  "system_prompt": "You are Jarv, a helpful CLI assistant..."
}
```

## Commands

| Command | Description |
|---|---|
| `jarv <anything>` | Ask jarv a question or give it a task |
| `jarv clear` | Clear conversation history |
| `jarv history` | Print recent conversation |
| `jarv config` | Show current settings |
| `jarv help` | Show help |
