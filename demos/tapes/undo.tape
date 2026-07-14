# Sessions: per-terminal history with /undo and /redo.
Require jarv
Output demos/output/undo.webp

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
Type "in one short sentence, why is the sky blue?"
Sleep 700ms
Enter
# "Started responding in Ns" precedes the final answer at low/none reasoning
# effort; then wait for the idle footer so the whole answer is on screen.
Wait+Screen@240s /Started responding/
Wait+Screen@240s /\/exit quit/
Sleep 4s
# /undo drops the exchange back out of the transcript; linger so it's clearly
# the reply vanishing on camera, not a cut.
Type "/undo"
Sleep 700ms
Enter
Sleep 5s
# /redo restores it — hold again so the answer visibly comes back.
Type "/redo"
Sleep 700ms
Enter
Sleep 5s
