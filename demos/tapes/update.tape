# /update: self-update for the active install channel.
Require jarv
Output demos/output/update.webp
Source demos/tapes/_common.tape
# Two lines of output — crop the frame to match (~8 rows).
Set Height 240

Sleep 500ms
Type "jarv /update"
Sleep 500ms
Enter
# Guard the returning-prompt match against the pre-command prompt, then hold on
# the result long enough to read the channel/version line.
Sleep 4s
Wait@120s /^> *$/
Sleep 2500ms
