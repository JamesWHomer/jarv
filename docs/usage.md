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
