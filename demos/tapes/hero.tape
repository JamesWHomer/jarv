# Hero shot: interactive heads-up session — ask, watch it work, get an answer.
Require jarv
Output demos/output/hero.webp

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
# Wait for the idle splash before typing: on a cold start (fresh .pyc compile)
# the input loop can lag several seconds behind the first paint and keystrokes
# typed into that window are lost.
Wait+Screen@60s /type a message to begin/
Sleep 2s
Type "what changed in this repo in the last day? keep it to a few bullets"
Sleep 700ms
Enter
# "Started responding in Ns" precedes the final answer at low/none reasoning
# effort (record-all.ps1 forces it; at high effort "Thought for" replaces it).
# This first marker also doubles as typing verification: a dropped cold-start
# keystroke leaves the screen unchanged, so the tape fails loudly on timeout.
Wait+Screen@240s /Started responding/
# Then hold until the turn is fully done. The idle footer's "/exit quit" only
# renders once the transcript has content AND the agent has stopped, so waiting
# on it guarantees the whole answer is on screen before we linger — rather than
# a fixed sleep that could clip a reply still streaming in.
Wait+Screen@240s /\/exit quit/
Sleep 6s
