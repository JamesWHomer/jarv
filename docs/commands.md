# Slash commands

![jarv slash commands](https://github.com/JamesWHomer/jarv/releases/download/readme-assets/commands.webp)

| Command | Description |
| --- | --- |
| `/help` | Show all commands |
| `/about` | Detailed info and examples |
| `/set <key> <value>` | Set a config value |
| `/unset <key>` | Reset a config key to default |
| `/settings` | Open the interactive settings menu |
| `/config` | Show raw config values |
| `/setup [provider\|key\|model\|base_url]` | Run the setup wizard or jump to one step |
| `/new` | Start a fresh session on the next prompt |
| `/resume` | Resume the most recent unarchived session used in this directory |
| `/archive` | Archive session history and sidecars |
| `/sessions` | Browse sessions (interactive when in a TTY) |
| `/sessions <id>` | Load a specific session by ID prefix |
| `/history` | Show recent conversation history |
| `/tree` | Browse the session as a tree — fork, edit, or resume any prompt |
| `/undo [n]` | Remove last *n* exchanges (default 1) |
| `/redo [n]` | Restore last *n* undone exchanges (default 1) |
| `/btw <question>` | Heads-up only: ask an aside, then save the completed exchange as a branch outside the main context |
| `/usage` | Interactive usage screen — spend vs the previous period, tokens, requests, context headroom, a spend-over-time chart, and share-of-spend by model. `←/→` (or `1-5` / `s t w m a`) switches scope live |
| `/usage <session\|day\|week\|month\|all>` | Open straight to a scope (`day`/`today`, `week`, `month` = rolling 24h, 7d, 30d; `all` = full system-wide history) |
| `/update` | Update Jarv to the latest version for the active install channel |
| `/uninstall [--purge] [--yes]` | Uninstall Jarv or show its package-manager uninstall command |

Except for `/btw` and the heads-up exit commands, commands work both as `jarv /command` and inside heads-up mode. `/session` aliases `/sessions`; `jarv help` also opens help. Read-only commands (`/help`, `/about`, `/usage`, and `/config`) use a temporary display by default in interactive terminals; change `read_only_command_display` in `/settings` to print them permanently instead. Without an interactive terminal, `/history`, `/tree`, and `/usage` print a static view, and `/sessions` lists the five most recent sessions, including archived ones.

`/btw` returns to the preceding exchange after a successful aside. If it is the first exchange, is cancelled, or fails, it stays on the active path. In incognito mode it runs as an ordinary prompt.

Agent tool calls have a separate `tool_call_display` setting. `auto` uses `print` for one-shot runs and `fullscreen` in heads-up mode. `print` is resize-safe and left-aligned; `fullscreen` uses bordered cards with right-aligned status.

In heads-up mode, command results and errors share one feedback area: each notice replaces the previous one, and sending a message clears it. Feedback stays visible while you type, reflows when you resize the terminal, and can be scrolled when long. In an empty session it overlays the space below the Jarv logo without moving or restarting the welcome animation. Session changes load the conversation before showing the result; messages printed around full-screen views also appear when you return. This feedback is only for display and is not saved as conversation history or sent to the model. The footer is reserved for keyboard hints and feedback about editing the prompt.

**Colour** in `/settings` defaults to **on**. Set it to **off** (or set `colour` to `false`) to render everything without colour — bold, dim, and underline are kept, so the layout still reads. Setting `NO_COLOR` in the environment does the same thing without changing your config. Existing `monochrome` preferences are migrated automatically.

Turn off **Menu borders** in the Display section of `/settings`, or run `/set headsup_border false`, to remove outer frames in heads-up mode and all menus (settings and editors, setup, sessions and previews, tree, usage, and read-only screens). Borderless frames have no side padding, so content uses the full width. Menu metadata shares the header, freeing the bottom row for content and keeping usage hints at the bottom. Editor controls, heads-up status, and the heads-up input box remain visible. The settings screen updates immediately; `/set headsup_border true` restores the borders and padding.
