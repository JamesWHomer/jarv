# Sessions: per-terminal history with /undo and /redo.
Require jarv
Output demos/output/undo.webp
Source demos/tapes/_common.tape

Sleep 500ms
Type "jarv"
Enter
# Wait for the idle splash before typing (cold-start keystroke-loss guard).
Wait+Screen@60s /type a message to begin/
Sleep 1s
Type "in one short sentence, why is the sky blue?"
Sleep 500ms
Enter
# "Thought for Ns" or "Started responding in Ns" (see hero.tape) precedes the
# answer. The idle footer needs a nudge to repaint once the turn ends — see the
# same dance in hero.tape.
Wait+Screen@240s /(Thought for|Started responding)/
Sleep 3s
Type " "
Backspace
Sleep 2500ms
Type " "
Backspace
Wait+Screen@120s /\/exit quit/
Sleep 1500ms
# /undo drops the exchange back out of the transcript; linger so it's clearly
# the reply vanishing on camera, not a cut.
Type "/undo"
Sleep 500ms
Enter
Sleep 2500ms
# /redo restores it — hold again so the answer visibly comes back.
Type "/redo"
Sleep 500ms
Enter
Sleep 2500ms
