# Scriptability: one-shot prompts and piped stdin, like any other unix tool.
Require jarv
Output demos/output/oneshot.webp

Set Shell "powershell"
Set FontFamily "Berkeley Mono"
Set FontSize 20
Set Width 1600
Set Height 868
Set Padding 16
Set Theme {"name": "github-dark", "background": "#212830", "foreground": "#c9d1d9", "cursor": "#c9d1d9", "selection": "#1f6feb", "black": "#484f58", "red": "#ff7b72", "green": "#3fb950", "yellow": "#d29922", "blue": "#58a6ff", "magenta": "#bc8cff", "cyan": "#39c5cf", "white": "#b1bac4", "brightBlack": "#6e7681", "brightRed": "#ffa198", "brightGreen": "#56d364", "brightYellow": "#e3b341", "brightBlue": "#79c0ff", "brightMagenta": "#d2a8ff", "brightCyan": "#56d4dd", "brightWhite": "#f0f6fc"}
Set TypingSpeed 40ms

Sleep 1s
Type "jarv what process is listening on port 8080?"
Sleep 700ms
Enter
# Let jarv paint over the shell first, so the returning-prompt match ("^> $")
# can't latch onto the pre-command prompt still on screen. Then read the answer.
Sleep 5s
Wait@240s /^> *$/
Sleep 4s
Type "cat README.md | jarv summarize this project in one sentence"
Sleep 700ms
Enter
Sleep 5s
Wait@240s /^> *$/
Sleep 5s
