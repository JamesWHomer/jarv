# Shared look for every demo: GitHub-dark, Berkeley Mono, 1600x868.
# Each tape pulls this in with `Source demos/tapes/_common.tape` (VHS applies Set
# commands in order, so a tape can override any of these afterwards — update.tape
# uses a shorter Height). Keep Require/Output/choreography in the tapes; a sourced
# tape may not declare its own Output.
Set Shell "powershell"
# Fallback chain, not a bare name: xterm.js measures cell width from whatever the
# browser resolves, so a missing font doesn't just change the typeface — it
# renders every glyph adrift in its cell. Berkeley Mono is a paid font; without
# it the demos record in Consolas.
Set FontFamily "Berkeley Mono,Consolas,monospace"
Set FontSize 20
Set Width 1600
Set Height 868
Set Padding 16
Set Theme {"name": "github-dark", "background": "#212830", "foreground": "#c9d1d9", "cursor": "#c9d1d9", "selection": "#1f6feb", "black": "#484f58", "red": "#ff7b72", "green": "#3fb950", "yellow": "#d29922", "blue": "#58a6ff", "magenta": "#bc8cff", "cyan": "#39c5cf", "white": "#b1bac4", "brightBlack": "#6e7681", "brightRed": "#ffa198", "brightGreen": "#56d364", "brightYellow": "#e3b341", "brightBlue": "#79c0ff", "brightMagenta": "#d2a8ff", "brightCyan": "#56d4dd", "brightWhite": "#f0f6fc"}
Set TypingSpeed 40ms
