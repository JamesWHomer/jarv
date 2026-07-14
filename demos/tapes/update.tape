# /update: self-update for the active install channel.
Require jarv
Output demos/output/update.webp

Set Shell "powershell"
Set FontFamily "Berkeley Mono"
Set FontSize 20
Set Width 1600
Set Height 428
Set Padding 16
Set Theme {"name": "github-dark", "background": "#212830", "foreground": "#c9d1d9", "cursor": "#c9d1d9", "selection": "#1f6feb", "black": "#484f58", "red": "#ff7b72", "green": "#3fb950", "yellow": "#d29922", "blue": "#58a6ff", "magenta": "#bc8cff", "cyan": "#39c5cf", "white": "#b1bac4", "brightBlack": "#6e7681", "brightRed": "#ffa198", "brightGreen": "#56d364", "brightYellow": "#e3b341", "brightBlue": "#79c0ff", "brightMagenta": "#d2a8ff", "brightCyan": "#56d4dd", "brightWhite": "#f0f6fc"}
Set TypingSpeed 40ms

Sleep 1s
Type "jarv /update"
Sleep 700ms
Enter
# Guard the returning-prompt match against the pre-command prompt, then hold on
# the result long enough to read the channel/version line.
Sleep 5s
Wait@120s /^> *$/
Sleep 5s
