# Slash commands: browse /help, then drive /settings — navigate, cycle a value, exit.
Require jarv
Output demos/output/commands.webp

Set Shell "powershell"
Set FontFamily "Berkeley Mono"
Set FontSize 20
Set Width 1600
Set Height 868
Set Padding 16
Set Theme {"name": "github-dark", "background": "#212830", "foreground": "#c9d1d9", "cursor": "#c9d1d9", "selection": "#1f6feb", "black": "#484f58", "red": "#ff7b72", "green": "#3fb950", "yellow": "#d29922", "blue": "#58a6ff", "magenta": "#bc8cff", "cyan": "#39c5cf", "white": "#b1bac4", "brightBlack": "#6e7681", "brightRed": "#ffa198", "brightGreen": "#56d364", "brightYellow": "#e3b341", "brightBlue": "#79c0ff", "brightMagenta": "#d2a8ff", "brightCyan": "#56d4dd", "brightWhite": "#f0f6fc"}
Set TypingSpeed 40ms

Sleep 1s
Type "jarv"
Enter
# Wait for the idle splash before typing (cold-start keystroke-loss guard).
Wait+Screen@60s /type a message to begin/
Sleep 2s
# Typing "/" opens the command palette, which filters live as the name is typed;
# dwell on it so the autocomplete popup reads before /help runs.
Type "/help"
Sleep 1500ms
Enter
# The waits below double as typing verification: on the flaky cold-start runs
# where keystrokes are lost, the screen never changes and the tape fails loudly
# on a timeout instead of producing a splash-only recording.
Wait+Screen@30s /\/redo/
Sleep 4s
# Scroll through the command list slowly enough to follow.
Down@500ms 6
Sleep 3s
Escape
Sleep 2s
Type "/settings"
Sleep 1200ms
Enter
Wait+Screen@30s /showing 1-/
Sleep 3s
# Row 5 is "Reasoning effort" (account: provider, api key, tier, base url; then model).
Down@500ms 5
Sleep 2s
# Enter on a choice row cycles the value in place and flashes "saved ...".
# record-all.ps1 re-asserts reasoning_effort before every take and restores the
# user's setting when done, so cycling it here is safe.
Enter
Wait+Screen@15s /saved Reasoning effort/
Sleep 3s
Enter
Sleep 2500ms
# Wander further down the list — runtime, audit, and command-review sections.
Down@350ms 9
Sleep 3s
Escape
Sleep 2500ms
