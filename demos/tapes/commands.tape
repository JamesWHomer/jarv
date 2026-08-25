# Slash commands: browse /help, then change a setting in /settings.
Require jarv
Output demos/output/commands.webp
Source demos/tapes/_common.tape

Sleep 500ms
Type "jarv"
Enter
# Wait for the idle splash before typing (cold-start keystroke-loss guard).
Wait+Screen@60s /type a message to begin/
Sleep 1s
# Typing "/" opens the command palette, which filters live as the name is typed;
# dwell on it so the autocomplete popup reads before /help runs.
Type "/help"
Sleep 1s
Enter
# The waits below double as typing verification: on the flaky cold-start runs
# where keystrokes are lost, the screen never changes and the tape fails loudly
# on a timeout instead of producing a splash-only recording.
Wait+Screen@30s /\/redo/
Sleep 1500ms
# Scroll a little way into the command list, slowly enough to follow.
Down@350ms 4
Sleep 1200ms
Escape
Sleep 700ms
Type "/settings"
Sleep 1s
Enter
Wait+Screen@30s /showing 1-/
Sleep 1500ms
# Row 5 is "Reasoning effort" (account: provider, api key, tier, base url; then
# model). Enter on a choice row cycles the value in place and flashes "saved ...".
# record.ps1/record-all.ps1 restore the user's setting afterwards, so cycling it
# on camera is safe.
Down@400ms 5
Sleep 1s
Enter
Wait+Screen@15s /saved Reasoning effort/
Sleep 1500ms
Escape
Sleep 700ms
