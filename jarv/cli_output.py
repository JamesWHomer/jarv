"""One-shot output protocol. Human diagnostics never enter protocol stdout."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import json
import sys
import threading


class DiagnosticStream:
    def __init__(self, stream):
        self.stream = stream

    def write(self, value):
        return self.stream.write(value)

    def flush(self):
        return self.stream.flush()

    def isatty(self):
        return False


class CliOutput:
    def __init__(self, output_format: str, *, quiet=False, verbose=False):
        self.format = output_format
        self.quiet = quiet
        self.verbose = verbose
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self._lock = threading.Lock()
        self.finished = False
        self.config = {}

    @contextmanager
    def route_diagnostics(self):
        from .display import console

        # Preserve None (dynamic sys.stdout), rather than pinning the resolved
        # stream after this invocation returns to an embedding caller.
        old_file = console._file
        stream = DiagnosticStream(self.stderr)
        # Existing modules all share this Console, including child threads.
        console.file = stream
        try:
            with redirect_stdout(stream):
                yield
        finally:
            console.file = old_file

    def event(self, event_type: str, **data):
        if self.format == "jsonl":
            self._json({"type": event_type, **data})

    def _json(self, value):
        with self._lock:
            # Cancelled workers can finish cleanup after their parent returns.
            # The terminal result must remain the last protocol record.
            if self.finished and value["type"] != "result":
                return
            self.stdout.write(json.dumps(value, ensure_ascii=True) + "\n")
            self.stdout.flush()

    def start_turn(self, query, config):
        self.config = config
        self.event("start", provider=config.get("provider"), model=config.get("model"))
        if self.verbose:
            print(f"Provider: {config.get('provider')}; model: {config.get('model')}", file=self.stderr)

    def show_tool_card(self, card):
        if not self.quiet:
            from .display import console
            console.print(card)

    def show_notice(self, message):
        if not self.quiet:
            from .display import console
            console.print(message)

    def show_error(self, message):
        # The final result carries the error, including in quiet mode.
        pass

    def ask_user(self, question, config):
        from .agent_ui import _dispatch_ask_user
        return _dispatch_ask_user({"question": question}, config)

    def finish(self, *, text="", error=None, status="success", session_id=None,
               turns=0, exit_code=0):
        if self.finished:
            return
        self.finished = True
        if self.verbose:
            print(f"Session: {session_id or '-'}; agent turns: {turns}; status: {status}", file=self.stderr)
        value = dict(type="result", status=status, text=text, error=error,
                     session_id=session_id, turns=turns, exit_code=exit_code)
        if self.format in ("json", "jsonl"):
            self._json(value)
        elif text:
            self.stdout.write(text + ("" if text.endswith("\n") else "\n"))
            self.stdout.flush()
        if error:
            print(error, file=self.stderr)
