# Scriptability: one-shot prompts and piped stdin, like any other unix tool.
Require jarv
Output demos/output/oneshot.webp
Source demos/tapes/_common.tape
# Two commands, ~16 rows of transcript — crop the frame instead of leaving half
# of a full-height terminal empty.
Set Height 520

Sleep 500ms
Type "jarv what process is listening on port 8080?"
Sleep 500ms
Enter
# Let jarv paint over the shell first, so the returning-prompt match ("^> $")
# can't latch onto the pre-command prompt still on screen. Then read the answer.
Sleep 2s
Wait@240s /^> *$/
Sleep 2500ms
Type "cat README.md | jarv summarize this project in one sentence"
Sleep 500ms
Enter
Sleep 2s
Wait@240s /^> *$/
Sleep 2500ms
