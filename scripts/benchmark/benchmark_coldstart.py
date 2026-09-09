"""Fresh-process startup benchmark; no real provider or user data is used.

Run with the checkout's Python: scripts/benchmark/benchmark_coldstart.py.
TTY detection is simulated at 100x30; rendering goes to a pipe, not a terminal
emulator. OS/bytecode caches are not cleared. Every sample includes Python
startup, imports, and CLI dispatch. Menus stop after the real first refresh;
one-shot stops its timer when a loopback server receives the first model POST.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHILD = r'''
import os, sys
mode = sys.argv.pop(1)
if mode == "menu":
    sys.stdin.isatty = lambda: True
    sys.stdout.isatty = lambda: True
    os.environ["JARV_BENCH_STOP_AFTER_PAINT"] = "1"
if os.environ.get("JARV_BENCH_PROFILE"):
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()
from jarv.cli import main
try:
    main()
finally:
    if os.environ.get("JARV_BENCH_PROFILE"):
        profiler.disable()
        profiler.dump_stats(os.environ["JARV_BENCH_PROFILE"])
'''
SEED = '''
from jarv.history import prepare_session_context, save_history
ctx = prepare_session_context(mark_message=True)
save_history([
    {"role": "user", "content": "Say hello", "id": "bench-user"},
    {"role": "assistant", "content": "Hello!", "id": "bench-assistant"},
], ctx.history_file)
'''
CASES = [
    ("one-shot", "request", ["--incognito", "Reply with OK."]),
    ("headsup", "menu", ["--incognito"]),
    *[(name, "menu", ["/" + name]) for name in
      ("settings", "setup", "sessions", "tree", "history", "usage", "help", "about", "config")],
    ("version", "exit", ["--version"]),
    ("CLI help", "exit", ["--help"]),
    ("settings print", "exit", ["/settings"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--case", action="append", choices=[case[0] for case in CASES])
    parser.add_argument("--profile", type=Path, help="Save cProfile data (requires --case one-shot --reps 1)")
    parser.add_argument("--output", type=Path, default=ROOT / "build/benchmarks/coldstart.json")
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")
    if args.profile and (args.case != ["one-shot"] or args.reps != 1):
        parser.error("--profile requires --case one-shot --reps 1")
    selected_cases = [case for case in CASES if not args.case or case[0] in args.case]
    requests = queue.Queue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data": []}')

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            requests.put((time.perf_counter_ns(), self.path))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {"id": "benchmark", "choices": [{"index": 0, "delta": {"content": "OK"}, "finish_reason": None}]}
            done = {"id": "benchmark", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\ndata: " + json.dumps(done) + "\n\ndata: [DONE]\n\n").encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    samples = {name: [] for name, _, _ in CASES}
    try:
        with tempfile.TemporaryDirectory(prefix="jarv-coldstart-") as tmp:
            env = os.environ.copy()
            if args.profile:
                args.profile.parent.mkdir(parents=True, exist_ok=True)
                env["JARV_BENCH_PROFILE"] = str(args.profile.resolve())
            env.update(HOME=tmp, USERPROFILE=tmp, WT_SESSION="jarv-coldstart",
                       PYTHONIOENCODING="utf-8", COLUMNS="100", LINES="30",
                       TERM="xterm-256color", NO_COLOR="1", JARV_BENCH_FIRST_PAINT="1",
                       NO_PROXY="127.0.0.1,localhost")
            config_dir = Path(tmp) / ".jarv"
            config_dir.mkdir()
            config = {"provider": "ollama", "model": "llama3.2", "check_updates": False,
                      "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                      "api_key": "", "api_keys": {}, "read_only_command_display": "fullscreen"}
            (config_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            subprocess.run([sys.executable, "-c", SEED], cwd=ROOT, env=env, check=True, capture_output=True)
            rng = random.Random(42)
            for repetition in range(args.reps):
                cases = list(selected_cases)
                rng.shuffle(cases)
                for name, mode, cli_args in cases:
                    if args.reps == 1:
                        print(f"Measuring {name}", flush=True)
                    started = time.perf_counter_ns()
                    result = subprocess.run([sys.executable, "-c", CHILD, mode, *cli_args],
                                            cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                            capture_output=True, text=True, encoding="utf-8",
                                            errors="replace", timeout=30)
                    ended = time.perf_counter_ns()
                    if result.returncode:
                        raise RuntimeError(f"{name}: exit {result.returncode}\n{result.stderr[-2000:]}")
                    label = None
                    if mode == "menu":
                        marks = [line for line in result.stderr.splitlines() if line.startswith("BENCH_READY ")]
                        labels = [line.split()[1] for line in result.stderr.splitlines() if line.startswith("JARV_FIRST_PAINT ")]
                        if not marks or not labels:
                            raise RuntimeError(f"{name}: no first-paint marker\n{result.stdout[-1000:]}\n{result.stderr[-1000:]}")
                        ended = int(marks[0].split()[1])
                        label = labels[0]
                    elif mode == "request":
                        ended, label = requests.get(timeout=2)
                        if label != "/v1/chat/completions":
                            raise RuntimeError(f"Unexpected one-shot request: {label}")
                    samples[name].append({"ms": (ended - started) / 1e6, "marker": label, "exit_code": result.returncode})
                print(f"Completed round {repetition + 1}/{args.reps}", flush=True)
    finally:
        server.shutdown()
        server.server_close()
    rows = []
    for name, mode, cli_args in selected_cases:
        values = [sample["ms"] for sample in samples[name]]
        rows.append(dict(name=name, endpoint=mode, args=cli_args, median_ms=statistics.median(values),
                         min_ms=min(values), max_ms=max(values), samples=samples[name]))
    payload = dict(timestamp=datetime.now(timezone.utc).isoformat(), python=sys.version,
                   executable=sys.executable, platform=platform.platform(),
                   methodology=__doc__, reps=args.reps, terminal="simulated TTY 100x30, NO_COLOR=1",
                   fixture="isolated home, one session, one exchange; incognito agent modes; Ollama-compatible loopback mock",
                   results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{'case':<18} {'median ms':>10} {'min ms':>10} {'max ms':>10}")
    for row in rows:
        print(f"{row['name']:<18} {row['median_ms']:10.1f} {row['min_ms']:10.1f} {row['max_ms']:10.1f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
