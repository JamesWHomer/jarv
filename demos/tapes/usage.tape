# /usage: interactive spend/token screen, arrow keys switch scope live.
Require jarv
Output demos/output/usage.webp
Source demos/tapes/_common.tape

Sleep 500ms
Type "jarv /usage"
Sleep 500ms
Enter
# Dwell on the Session scope once it paints, then arrow through the rest:
# Right cycles Session -> Today -> Week -> Month -> All.
Wait+Screen@60s /scope/
Sleep 2s
Right
Sleep 1500ms
Right
Sleep 1500ms
Right
Sleep 1500ms
Right
Sleep 2s
Escape
Sleep 800ms
