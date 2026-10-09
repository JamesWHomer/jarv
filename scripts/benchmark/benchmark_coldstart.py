"""Fresh-process startup benchmark; no real provider or user data is used.

Run with the checkout's Python: scripts/benchmark/benchmark_coldstart.py.
TTY detection is simulated at 100x30; rendering goes to a pipe, not a terminal
emulator. OS/bytecode caches are not cleared. Every sample includes Python
startup, imports, and CLI dispatch. Menus stop after the real first refresh;
one-shot records the first visible stdout bytes and the arrival of the first
model POST at a loopback server (an upper bound on request-send latency).
Terminal one-shot cases simulate a TTY; the original one-shot case uses a pipe.
With --menu-input, menus also process injected keys through their real event
loop and repaint. Editable menus assert that the text reached their editor.
Keyboard decoding and terminal-emulator presentation are not measured.
All registered slash commands are covered. Update stops immediately before its
release lookup (no network/install); uninstall only prints manager instructions.
Setup's piped-input case measures its expected refusal (exit 130). /btw from the
CLI prints its in-session-only notice; /exit and /quit are in-session controls,
not standalone CLI commands. Configuration/session mutations use the fixture.
--launcher measures a matching-version console executable directly, including
its launcher overhead. This selects only piped command/one-shot cases; menus
and the update boundary require the instrumented Python entry point.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import queue
import random
import re
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
sys.path.insert(0, str(ROOT))
CHILD = r'''
import os, sys
mode = sys.argv.pop(1)
if os.environ.get("JARV_BENCH_SOURCE_ROOT"):
    sys.path.insert(0, os.environ["JARV_BENCH_SOURCE_ROOT"])
# Fail closed if a case unexpectedly attempts a real network connection or
# starts an installer. This guard is installed before importing any Jarv code.
def audit(event, args):
    if event == "subprocess.Popen" and (mode == "update-ready" or "/uninstall" in sys.argv):
        raise RuntimeError("Benchmark child must not launch subprocesses")
    if event == "socket.connect" and args[1][0] not in ("127.0.0.1", "::1"):
        raise RuntimeError("Benchmark child must not contact external services")
sys.addaudithook(audit)
if mode == "update-ready":
    import time
    import jarv.commands as commands
    def stop_before_lookup():
        print("BENCH_READY", time.perf_counter_ns(), file=sys.__stderr__, flush=True)
        os._exit(0)
    commands._fetch_latest_pypi_release = stop_before_lookup
    # Also guard an unexpectedly detected standalone installation.
    import jarv.standalone as standalone
    standalone.fetch_release_manifest = stop_before_lookup
if mode in ("menu", "setup-menu", "request-tty"):
    sys.stdin.isatty = lambda: True
    sys.stdout.isatty = lambda: True
if mode in ("menu", "setup-menu"):
    os.environ["JARV_BENCH_STOP_AFTER_PAINT"] = "1"
if os.environ.get("JARV_BENCH_PROFILE"):
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    original_exit = os._exit
    def profiled_exit(code):
        profiler.disable()
        profiler.dump_stats(os.environ["JARV_BENCH_PROFILE"])
        original_exit(code)
    os._exit = profiled_exit
if mode == "menu" and os.environ.get("JARV_BENCH_MENU_INPUT"):
    from scripts.benchmark.menu_probe import install
    install()
from jarv.cli import main
try:
    main()
finally:
    if os.environ.get("JARV_BENCH_PROFILE"):
        profiler.disable()
        profiler.dump_stats(os.environ["JARV_BENCH_PROFILE"])
'''
SEED = '''
import os, sys
if os.environ.get("JARV_BENCH_SOURCE_ROOT"):
    sys.path.insert(0, os.environ["JARV_BENCH_SOURCE_ROOT"])
from jarv.history import prepare_session_context, save_history, save_redo_stack, redo_file_for
ctx = prepare_session_context(mark_message=True)
history = []
for i in range(int(os.environ.get("JARV_BENCH_HISTORY_TURNS", "1"))):
    history.extend([
        {"role": "user", "content": "Say hello", "id": f"bench-user-{i}"},
        {"role": "assistant", "content": "Hello!", "id": f"bench-assistant-{i}"},
    ])
save_history(history, ctx.history_file)
save_redo_stack([[{"role": "user", "content": "A previously undone prompt", "id": "redo-user"},
                  {"role": "assistant", "content": "A restored answer", "id": "redo-assistant"}]],
                redo_file_for(ctx.history_file))
'''
CASES = [
    ("one-shot", "request", ["--incognito", "Reply with OK."]),
    ("one-shot terminal", "request-tty", ["--incognito", "Reply with OK."]),
    ("one-shot saved", "request-tty", ["--new", "Reply with OK."]),
    ("one-shot resumed", "request-tty", ["Reply with OK."]),
    ("headsup", "menu", ["--incognito"]),
    ("headsup saved", "menu", []),
    *[(name, "menu", ["/" + name]) for name in
      ("settings", "setup", "sessions", "session", "tree", "history", "usage", "help", "about", "config")],
    # The generic key probe is for the setup landing screen, not its submenus.
    *[("setup " + step, "setup-menu", ["/setup", step]) for step in ("provider", "key", "model", "base_url")],
    ("version", "exit", ["--version"]),
    ("CLI help", "exit", ["--help"]),
    ("help alias", "exit", ["help"]),
    ("settings print", "exit", ["/settings"]),
    *[(name + " print", "exit", ["/" + name]) for name in
      ("help", "about", "config", "history", "usage", "sessions", "session", "tree")],
    *[("usage " + period, "exit", ["/usage", period]) for period in ("session", "day", "week", "month", "all")],
    ("new", "exit", ["/new"]),
    ("archive", "exit", ["/archive"]),
    ("undo", "exit", ["/undo"]),
    ("redo", "exit", ["/redo"]),
    ("resume", "exit", ["/resume"]),
    ("set", "exit", ["/set", "command_timeout", "30"]),
    ("unset", "exit", ["/unset", "command_timeout"]),
    ("btw CLI notice", "exit", ["/btw", "Reply with OK."]),
    ("restart CLI notice", "exit", ["/restart"]),
    ("setup piped refusal", "exit", ["/setup"]),
    ("update pre-network", "update-ready", ["/update"]),
    ("uninstall instructions", "exit", ["/uninstall"]),
]


def run_child(command, *, env):
    """Drain both pipes, timestamping visible output without unbuffering the CLI."""
    chunks = {"stdout": [], "stderr": []}
    first_print = []
    ansi = re.compile(rb"\x1b\[[0-?]*[ -/]*[@-~]")

    def drain(stream, name):
        with stream:
            while chunk := stream.read1(4096):
                received = time.perf_counter_ns()
                chunks[name].append(chunk)
                if name == "stdout" and not first_print:
                    visible = ansi.sub(b"", b"".join(chunks[name])).strip()
                    if visible:
                        first_print.append(received)

    started = time.perf_counter_ns()
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    readers = [threading.Thread(target=drain, args=(getattr(process, name), name))
               for name in chunks]
    for reader in readers:
        reader.start()
    try:
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for reader in readers:
            reader.join()
    ended = time.perf_counter_ns()
    result = subprocess.CompletedProcess(
        command, process.returncode,
        **{name: b"".join(parts).decode("utf-8", errors="replace")
           for name, parts in chunks.items()},
    )
    return result, started, ended, first_print[0] if first_print else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--case", action="append", choices=[case[0] for case in CASES])
    parser.add_argument("--profile", type=Path, help="Save cProfile data (requires one --case and --reps 1)")
    parser.add_argument("--menu-input", action="store_true", help="Measure real key handling and its repaint too")
    parser.add_argument("--history-turns", type=int, default=1)
    parser.add_argument("--launcher", type=Path, help="Console executable for piped cases (must match checkout version)")
    parser.add_argument("--source-root", type=Path, default=ROOT,
                        help="Source snapshot to benchmark, keeping the same working directory")
    parser.add_argument("--baseline-python", type=Path,
                        help="Interleave a reference interpreter/source with each round")
    parser.add_argument("--baseline-source-root", type=Path,
                        help="Reference source snapshot for --baseline-python")
    parser.add_argument("--single-core", action="store_true", help="Constrain this process and its children to one logical CPU")
    parser.add_argument("--output", type=Path, default=ROOT / "build/benchmarks/coldstart.json")
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    if not (args.source_root / "jarv" / "cli.py").is_file():
        parser.error("--source-root must contain jarv/cli.py")
    if args.launcher and args.source_root != ROOT:
        parser.error("--source-root cannot override an installed console launcher")
    if bool(args.baseline_python) != bool(args.baseline_source_root):
        parser.error("--baseline-python and --baseline-source-root must be supplied together")
    baseline = None
    if args.baseline_python:
        if args.launcher or args.profile:
            parser.error("Interleaved comparisons cannot use --launcher or --profile")
        args.baseline_python = args.baseline_python.resolve()
        args.baseline_source_root = args.baseline_source_root.resolve()
        if not (args.baseline_source_root / "jarv" / "cli.py").is_file():
            parser.error("--baseline-source-root must contain jarv/cli.py")
        version = subprocess.run([str(args.baseline_python), "-c", "import sys; print(sys.version)"],
                                 capture_output=True, text=True, check=True)
        baseline = dict(python=version.stdout.strip(), executable=str(args.baseline_python),
                        source_root=str(args.baseline_source_root))
    if args.reps < 1:
        parser.error("--reps must be positive")
    if args.history_turns < 1:
        parser.error("--history-turns must be positive")
    cpu = None
    if args.single_core:
        from benchmark_support import pin_single_cpu
        cpu = pin_single_cpu()
    if args.profile and (not args.case or len(args.case) != 1 or args.reps != 1):
        parser.error("--profile requires one --case and --reps 1")
    selected_cases = [case for case in CASES if not args.case or case[0] in args.case]
    from jarv import __version__
    from jarv.command_registry import COMMANDS

    if args.launcher:
        args.launcher = args.launcher.resolve()
        if args.menu_input or args.profile:
            parser.error("--launcher cannot use --menu-input or --profile")
        unsupported = [name for name, mode, _ in selected_cases if mode not in ("exit", "request")]
        if args.case and unsupported:
            parser.error(f"--launcher cannot instrument these cases: {unsupported}")
        selected_cases = [case for case in selected_cases if case[1] in ("exit", "request")]
        version = subprocess.run([str(args.launcher), "--version"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        if version.returncode or version.stdout.strip() != f"jarv {__version__}":
            parser.error(f"Console launcher does not match jarv {__version__}: {version.stdout.strip()}")
    covered = {cli_args[0][1:] for _, _, cli_args in CASES if cli_args and cli_args[0].startswith("/")}
    missing = set(COMMANDS) - covered
    if missing:
        parser.error(f"Missing benchmark cases for registered commands: {sorted(missing)}")
    if baseline:
        selected_cases += [("baseline: " + name, mode, cli_args)
                           for name, mode, cli_args in list(selected_cases)]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
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
    samples = {name: [] for name, _, _ in selected_cases}
    try:
        with tempfile.TemporaryDirectory(prefix="jarv-coldstart-") as tmp:
            env = {key: value for key, value in os.environ.items() if not key.startswith("JARV_BENCH_")}
            env["JARV_BENCH_HISTORY_TURNS"] = str(args.history_turns)
            if args.source_root != ROOT:
                env["JARV_BENCH_SOURCE_ROOT"] = str(args.source_root)
            if args.menu_input:
                env["JARV_BENCH_MENU_INPUT"] = "1"
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
                      "api_key": "", "api_keys": {}, "read_only_command_display": "fullscreen",
                      "command_timeout": 45}
            (config_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            subprocess.run([sys.executable, "-c", SEED], cwd=ROOT, env=env, check=True, capture_output=True)
            seed_files = {path: path.read_bytes() for path in config_dir.rglob("*") if path.is_file()}
            rng = random.Random(42)
            for repetition in range(args.reps):
                cases = list(selected_cases)
                rng.shuffle(cases)
                for name, mode, cli_args in cases:
                    case_name = name.removeprefix("baseline: ")
                    sample_env = env
                    interpreter = sys.executable
                    if name.startswith("baseline: "):
                        sample_env = {**env, "JARV_BENCH_SOURCE_ROOT": str(args.baseline_source_root)}
                        interpreter = str(args.baseline_python)
                    # A saved one-shot changes the active session. Restore the
                    # seeded fixture before every sample, outside the timer.
                    for path in config_dir.rglob("*"):
                        if path.is_file() and path not in seed_files:
                            path.unlink()
                    for path, content in seed_files.items():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(content)
                    if args.reps == 1:
                        print(f"Measuring {name}", flush=True)
                    command = ([str(args.launcher), *cli_args] if args.launcher else
                               [interpreter, "-c", CHILD, mode, *cli_args])
                    result, started, ended, first_print = run_child(
                        command, env=sample_env)
                    expected_status = {
                        "setup piped refusal": 130, "restart CLI notice": 2,
                    }.get(case_name, 0)
                    if result.returncode != expected_status:
                        raise RuntimeError(f"{name}: exit {result.returncode}\n{result.stderr[-2000:]}")
                    expected_text = {"new": "New session starts", "archive": "Session archived",
                                     "undo": "Unsent", "redo": "Restored"}.get(case_name)
                    if expected_text and expected_text not in result.stdout:
                        raise RuntimeError(f"{name}: expected a real mutation, got\n{result.stdout}")
                    completed = ended
                    label = None
                    if mode == "update-ready":
                        marks = [line for line in result.stderr.splitlines() if line.startswith("BENCH_READY ")]
                        if len(marks) != 1:
                            raise RuntimeError(f"{name}: no release-lookup marker\n{result.stdout}\n{result.stderr}")
                        ended = int(marks[0].split()[1])
                        label = "before external release lookup"
                    elif mode in ("menu", "setup-menu"):
                        marks = [line for line in result.stderr.splitlines() if line.startswith("BENCH_READY ")]
                        labels = [line.split()[1] for line in result.stderr.splitlines() if line.startswith("JARV_FIRST_PAINT ")]
                        if not marks or not labels:
                            raise RuntimeError(f"{name}: no first-paint marker\n{result.stdout[-1000:]}\n{result.stderr[-1000:]}")
                        ended = int(marks[0].split()[1])
                        label = labels[0]
                    elif mode.startswith("request"):
                        ended, label = requests.get(timeout=2)
                        if label != "/v1/chat/completions":
                            raise RuntimeError(f"Unexpected one-shot request: {label}")
                    samples[name].append({"ms": (ended - started) / 1e6, "marker": label,
                                          "completion_ms": (completed - started) / 1e6,
                                          "first_print_ms": (first_print - started) / 1e6 if first_print else None,
                                          "exit_code": result.returncode})
                    if mode == "menu" and args.menu_input:
                        sample = samples[name][-1]
                        for marker, field in (("BENCH_FIRST_KEY", "first_key_ms"),
                                              ("BENCH_INPUT_READY", "input_ready_ms"),
                                              ("BENCH_INPUT_PAINT", "input_paint_ms")):
                            marks = [line for line in result.stderr.splitlines() if line.startswith(marker + " ")]
                            if len(marks) != 1:
                                raise RuntimeError(f"{name}: missing/duplicate {marker}")
                            sample[field] = (int(marks[0].split()[1]) - started) / 1e6
                        sample["modules"] = next(line.split(" ", 1)[1].split(",") for line in
                                                  result.stderr.splitlines() if line.startswith("BENCH_MODULES "))
                print(f"Completed round {repetition + 1}/{args.reps}", flush=True)
    finally:
        server.shutdown()
        server.server_close()
    rows = []
    for name, mode, cli_args in selected_cases:
        values = [sample["ms"] for sample in samples[name]]
        rows.append(dict(name=name, endpoint=mode, args=cli_args, median_ms=statistics.median(values),
                         min_ms=min(values), max_ms=max(values),
                         p95_ms=sorted(values)[max(0, math.ceil(.95 * len(values)) - 1)],
                         samples=samples[name]))
        prints = [sample["first_print_ms"] for sample in samples[name] if sample["first_print_ms"] is not None]
        rows[-1]["first_print_median_ms"] = statistics.median(prints) if prints else None
        rows[-1]["completion_median_ms"] = statistics.median(sample["completion_ms"] for sample in samples[name])
        for field in ("first_key_ms", "input_ready_ms", "input_paint_ms"):
            values = [sample[field] for sample in samples[name] if field in sample]
            if values:
                rows[-1][field.replace("_ms", "_median_ms")] = statistics.median(values)
    def source_digest(root):
        digest = hashlib.sha256()
        for path in sorted((root / "jarv").rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json"}:
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        return digest.hexdigest()

    if baseline:
        baseline["source_sha256"] = source_digest(args.baseline_source_root)
        baseline["results"] = [{**row, "name": row["name"].removeprefix("baseline: ")}
                               for row in rows if row["name"].startswith("baseline: ")]
        rows = [row for row in rows if not row["name"].startswith("baseline: ")]
    payload = dict(timestamp=datetime.now(timezone.utc).isoformat(), python=sys.version,
                   jarv_version=__version__, revision=revision.stdout.strip() if revision.returncode == 0 else None,
                   source_root=str(args.source_root), source_sha256=source_digest(args.source_root),
                   working_directory=str(ROOT), registered_commands=sorted(COMMANDS),
                   processor=platform.processor(), logical_cpus=os.cpu_count(),
                   launcher=str(args.launcher) if args.launcher else None,
                   executable=sys.executable, platform=platform.platform(),
                   methodology=__doc__, reps=args.reps, menu_input=args.menu_input,
                   history_turns=args.history_turns, single_cpu=cpu,
                   terminal="simulated TTY 100x30, NO_COLOR=1",
                   fixture="isolated home, restored seeded session per sample; incognito, fresh saved and resumed one-shot modes; Ollama-compatible loopback mock",
                   results=rows, interleaved_baseline=baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    fields = ["name", "endpoint", "args", "median_ms", "p95_ms", "min_ms", "max_ms",
              "first_print_median_ms", "completion_median_ms"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "args": " ".join(row["args"])})
    print(f"{'case':<18} {'median ms':>10} {'min ms':>10} {'max ms':>10} {'first print':>12}")
    for row in rows:
        first = row['first_print_median_ms']
        print(f"{row['name']:<18} {row['median_ms']:10.1f} {row['min_ms']:10.1f} {row['max_ms']:10.1f} {f'{first:.1f}' if first is not None else '-':>12}")
        if "input_ready_median_ms" in row:
            print(f"  input accepted: {row['input_ready_median_ms']:.1f} ms; repainted: {row['input_paint_median_ms']:.1f} ms")
    print(f"Saved {args.output}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
