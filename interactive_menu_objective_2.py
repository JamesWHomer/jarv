import sys
import time


def out(text="", *, end="\n"):
    print(text, end=end, flush=True)


def read_char():
    return sys.stdin.read(1)


def read_key():
    ch = read_char()
    if ch == "\x1b":
        second = read_char()
        if second == "[":
            third = read_char()
            return {
                "A": "UP",
                "B": "DOWN",
                "C": "RIGHT",
                "D": "LEFT",
            }.get(third, "ESC_SEQ")
        return "ESC"
    if ch == "\t":
        return "TAB"
    if ch in ("\n", "\r"):
        return "ENTER"
    if ch == "":
        return "EOF"
    return ch


def wait_for_enter():
    while True:
        key = read_key()
        if key == "ENTER":
            return True
        if key == "EOF":
            return False
        out(f"Expected ENTER, got {key}. Try again.")


def tab_gate():
    out()
    out("Step 2 - Focus gate")
    out("Instruction: send TAB exactly twice, then ENTER.")
    tabs = 0
    while True:
        key = read_key()
        if key == "TAB":
            tabs += 1
            out(f"Tab count: {tabs}/2")
            continue
        if key == "ENTER":
            if tabs == 2:
                out("Focus gate passed.")
                return True
            out(f"Focus gate failed: expected 2 tabs before ENTER, got {tabs}.")
            tabs = 0
            out("Try again: send TAB exactly twice, then ENTER.")
            continue
        if key == "EOF":
            return False
        out(f"Focus gate ignored {key}.")


def arrow_menu():
    out()
    out("Step 3 - Arrow menu")
    out("Instruction: use DOWN twice, then ENTER, to select Gamma.")
    choices = ["Alpha", "Beta", "Gamma"]
    index = 0
    while True:
        out(f"Selected: {choices[index]}")
        key = read_key()
        if key == "DOWN":
            index = (index + 1) % len(choices)
            continue
        if key == "UP":
            index = (index - 1) % len(choices)
            continue
        if key == "ENTER":
            if choices[index] == "Gamma":
                out("Arrow menu passed: Gamma selected.")
                return True
            out(f"Arrow menu failed: selected {choices[index]}, expected Gamma.")
            out("Resetting to Alpha. Use DOWN twice, then ENTER.")
            index = 0
            continue
        if key == "EOF":
            return False
        out(f"Arrow menu ignored {key}.")


def line_gate():
    out()
    out("Step 4 - Typed value")
    out("Instruction: type launch-42 and press ENTER.")
    while True:
        out("Value: ", end="")
        value = sys.stdin.readline()
        if value == "":
            return False
        value = value.strip()
        if value == "launch-42":
            out("Typed value passed.")
            return True
        out(f"Typed value failed: got {value!r}. Try again.")


def escape_gate():
    out()
    out("Step 5 - Modal")
    out("Instruction: dismiss the modal with ESC.")
    while True:
        key = read_key()
        if key == "ESC":
            out("Modal dismissed.")
            return True
        if key == "EOF":
            return False
        out(f"Modal still open. Expected ESC, got {key}.")


def eof_gate():
    out()
    out("Step 6 - Close input")
    out("Instruction: send EOF to finish cleanly.")
    key = read_key()
    if key == "EOF":
        out("Objective complete: waited for sync, pressed ENTER, used TAB, arrows, typed launch-42, dismissed with ESC, and closed with EOF.")
        return True
    out(f"Objective incomplete: expected EOF, got {key}.")
    return False


def main():
    out("Jarv Interactive Menu Test 2")
    out("Objective:")
    out("1. Wait for SYNC READY, then press ENTER.")
    out("2. Send TAB exactly twice, then ENTER.")
    out("3. Use DOWN twice, then ENTER, to select Gamma.")
    out("4. Type launch-42 and press ENTER.")
    out("5. Send ESC to dismiss the modal.")
    out("6. Send EOF to finish.")
    out()
    out("Step 1 - Delayed sync")
    out("SYNC STARTED. Wait until SYNC READY appears before pressing ENTER.")
    time.sleep(1.5)
    out("SYNC READY. Press ENTER to continue.")
    if not wait_for_enter():
        out("Objective incomplete: EOF before ENTER.")
        return 1
    out("Sync gate passed.")

    for gate in (tab_gate, arrow_menu, line_gate, escape_gate, eof_gate):
        if not gate():
            out("Objective incomplete.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
