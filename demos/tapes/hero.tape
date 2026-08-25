# Hero shot: interactive heads-up session — ask, watch it work, get an answer.
Require jarv
Output demos/output/hero.webp
Source demos/tapes/_common.tape

Sleep 500ms
Type "jarv"
Enter
# Wait for the idle splash before typing: on a cold start (fresh .pyc compile)
# the input loop can lag several seconds behind the first paint and keystrokes
# typed into that window are lost.
Wait+Screen@60s /type a message to begin/
Sleep 1s
# Short, self-bounding question: one git call, three lines of answer.
Type "what changed in the last 3 commits? one line each"
Sleep 500ms
Enter
# The first status line before the answer: "Thought for Ns" when the model
# returned reasoning, "Started responding in Ns" when it didn't — which of the
# two shows up depends on the model, so match either. It also doubles as typing
# verification: a dropped cold-start keystroke leaves the screen unchanged, so
# the tape fails loudly on timeout.
Wait+Screen@240s /(Thought for|Started responding)/
# Sync on the tool card finishing rather than guessing how long the git call
# takes — only the short closing answer is left after this.
Wait+Screen@120s /✓ done/
# The idle footer ("/exit quit") is the one on-screen proof the turn is over, but
# jarv doesn't repaint when it ends: the footer keeps advertising "cancel turn"
# until the next event, so waiting on it alone deadlocks. Nudge the input box (a
# space, immediately erased) to force the repaint — twice, in case the first
# lands while the answer is still streaming.
Sleep 3s
Type " "
Backspace
Sleep 2500ms
Type " "
Backspace
Wait+Screen@120s /\/exit quit/
Sleep 1500ms
