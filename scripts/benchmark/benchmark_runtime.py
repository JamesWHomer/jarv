"""Offline runtime benchmark with deterministic fixtures and output fingerprints.

Run separately from tests/coldstart benchmarks. Timings exclude imports, fixture
creation and hashing. Each case is warmed once, then measured in shuffled rounds.
Allocation peaks are measured in a separate tracemalloc pass (not RSS). No API,
user home, or user session is accessed. Disk cases retain real locking/fsync.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import re
import statistics
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def fingerprint(value):
    from rich.text import Text
    if isinstance(value, Text):
        return [value.plain, str(value.style), [(s.start, s.end, str(s.style)) for s in value.spans]]
    if isinstance(value, (tuple, list)):
        return [fingerprint(item) for item in value]
    if isinstance(value, dict):
        return {str(key): fingerprint(item) for key, item in value.items()}
    return value


def digest(value):
    serialized = json.dumps(fingerprint(value), sort_keys=True, ensure_ascii=True)
    # Retained result handles are random; preserve their relationships in hashes.
    handles = {}
    serialized = re.sub(r"cmd_[0-9a-f]{12}", lambda m: handles.setdefault(m[0], f"cmd_{len(handles):012x}"), serialized)
    return hashlib.sha256(serialized.encode()).hexdigest()


def make_cases(home):
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.text import Text
    from jarv import anthropic_http, gemini_http, models_dev, openai_http
    from jarv.agent import build_agent_tools
    from jarv.agent_ui import StreamingMarkdownPreview
    from jarv.artifacts import ArtifactStore
    from jarv.config import DEFAULT_CONFIG, load_config
    from jarv.context_budget import build_input, trim_turn_input
    from jarv.edit_tool import build_edit_diff
    from jarv.headsup import HeadsupApp, TranscriptEntry
    from jarv.history import load_history, save_history
    from jarv.http_transport import iter_sse_json
    from jarv.project_context import build_project_context
    from jarv.provider import _to_chat_messages, _to_chat_tools
    from jarv.read_tool import dispatch_read_tool
    from jarv.retained_outputs import RetainedOutputStore
    from jarv.session_render import _history_visual_lines, _read_result_summary
    from jarv.session_browser import SessionBrowserScreen
    from jarv.session_tree import build_tree
    from jarv.text_editor import render_visual_line_window
    from jarv.usage import load_global_usage_records, aggregate_usage_records
    from jarv.shell import execute_command
    from jarv.web import web_content_from_bytes
    from jarv.intro_animation import render_intro

    config = dict(DEFAULT_CONFIG, provider="ollama", model="bench-model", check_updates=False)
    (home / ".jarv").mkdir()
    (home / ".jarv/config.json").write_text(json.dumps(config), encoding="utf-8")
    tools = build_agent_tools(config)
    instructions = "Preserve all project instructions.\n" * 100
    history = []
    for i in range(1000):
        history.extend([
            {"role": "user", "content": f"Inspect item {i}", "frame_id": f"frame-{i}"},
            {"type": "function_call", "id": f"fc_{i}", "call_id": f"call-{i}", "name": "read", "arguments": '{"input":"sample.txt"}'},
            {"type": "function_call_output", "call_id": f"call-{i}", "output": "line of tool output\n" * 40},
            {"role": "assistant", "content": f"Result {i}: **complete**.\n\n- First item\n- Second item"},
        ])
    history_path = home / ".jarv/sessions/history-bench.json"
    save_history(history, history_path)
    unicode_history = [{"role": "user", "content": "界🙂" * 2000} for _ in range(100)]
    unicode_path = home / ".jarv/sessions/history-unicode.json"
    save_history(unicode_history, unicode_path)
    read_path = home / "sample.txt"
    read_path.write_text("abcdef 界 é\n" * 100000, encoding="utf-8")
    usage_path = home / ".jarv/usage.json"
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    records = [{"created_at": (now - timedelta(hours=i)).isoformat(), "provider": "ollama",
                "model": "bench-model", "source": "root", "input_tokens": 100, "output_tokens": 20}
               for i in range(20000)]
    usage_path.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    cases = {}
    for label, value in (("short", "hello world"), ("long ASCII", "line of draft text\n" * 10000),
                         ("long Unicode", "界 é draft text\n" * 10000)):
        state = {"buffer": value, "cursor": len(value)}
        cases[f"editor {label}"] = lambda state=state: render_visual_line_window(state, 92, max_lines=6)

    render_console = Console(file=io.StringIO(), width=100, height=30, color_system=None, force_terminal=True)
    app = HeadsupApp(config, None, args=SimpleNamespace(incognito=True), agent_loader=None,
                     handle_slash=lambda *a: None, maybe_command=lambda *a: None, render_console=render_console)
    app._idle_anim_stop.set()
    app._cwd_label = lambda: "benchmark"
    app.entries = [TranscriptEntry("assistant", Markdown(f"Result {i}: **complete**.\n\n- First\n- Second")) for i in range(1000)]

    def frame(cold=False):
        if cold:
            for entry in app.entries:
                entry.invalidate()
        return [[(s.text, str(s.style), str(s.control)) for s in line]
                for line in render_console.render_lines(app.render(), pad=False)]

    cases["transcript first frame 1000"] = lambda: frame(True)
    cases["transcript warm frame 1000"] = frame
    cases["history render 100 turns"] = lambda: _history_visual_lines(history[:400], 96)
    cases["session tree 1000 turns"] = lambda: [(n.frame_id, n.depth, n.prompt_text, n.response_preview) for n in build_tree(history, []).nodes]
    browser = SessionBrowserScreen.__new__(SessionBrowserScreen)
    browser.background = False
    browser.sessions = {"bench": {"history_file": str(history_path), "label": "benchmark"}}
    browser.search_text_cache = {}
    browser.search_folded_cache = {}
    cases["session search cold 1000 turns"] = lambda: browser._build_search_text("bench")
    browser._search_text("bench")
    cases["session search warm 1000 turns"] = lambda: browser._search_text("bench")
    cases["intro animation frame"] = lambda: render_intro(96, 23, 1.25)

    class PreviewSink:
        def update(self, renderable, *, refresh=False):
            self.text = renderable._text

    def stream():
        sink = PreviewSink()
        ticks = iter(i * 0.01 for i in range(5001))
        preview = StreamingMarkdownPreview(sink, max_lines=28, clock=lambda: next(ticks))
        for i in range(5000):
            preview.append(f"delta {i}: words **bold**\n")
        preview.flush()
        return preview.text, sink.text

    cases["stream 5000 deltas"] = stream
    sse_lines = [line for i in range(10000) for line in ("data: " + json.dumps({"delta": str(i)}), "")]
    cases["SSE parse 10000 events"] = lambda: list(iter_sse_json("benchmark", SimpleNamespace(iter_lines=lambda: iter(sse_lines))))
    for count in (10, 1000):
        items = history[:count * 4]
        cases[f"context build {count} turns"] = lambda items=items: build_input(items, model="bench-model", config=config, instructions=instructions, tools=tools)
    # Oversized single turn exercises retained-output budgeting without changing fixtures.
    oversized = [history[0], *[dict(history[2], call_id=f"call-{i}", output="x" * 20000) for i in range(200)]]
    cases["context retained 200 results"] = lambda: trim_turn_input(oversized, model="bench-model", config=config, instructions=instructions, tools=tools, retained_store=RetainedOutputStore())
    cases["Anthropic convert 1000 turns"] = lambda: anthropic_http.to_messages(history)
    cases["Gemini convert 1000 turns"] = lambda: gemini_http.to_contents(history)
    cases["Chat payload 1000 turns"] = lambda: openai_http.build_chat_payload("bench-model", _to_chat_messages(instructions, history), _to_chat_tools(tools))
    cases["Responses payload 1000 turns"] = lambda: openai_http.build_responses_payload("bench-model", instructions, tools, history)
    cases["Anthropic payload 1000 turns"] = lambda: anthropic_http.build_payload(config, "bench-model", instructions, tools, history)
    cases["Gemini payload 1000 turns"] = lambda: gemini_http.build_payload(config, "bench-model", instructions, tools, history)
    cases["history load 1000 turns"] = lambda: list(load_history(history_path))
    cases["history save 1000 turns"] = lambda: save_history(history, history_path)
    cases["history load Unicode 400k chars"] = lambda: list(load_history(unicode_path))
    cases["history save Unicode 400k chars"] = lambda: save_history(unicode_history, unicode_path)
    cases["usage load all 20000"] = lambda: load_global_usage_records(usage_path, warn=False)
    cases["usage load day 20000"] = lambda: load_global_usage_records(usage_path, since=timedelta(days=1), now=now, warn=False)
    cases["usage aggregate 20000"] = lambda: aggregate_usage_records(records)
    cases["read file bounded 1MB"] = lambda: str(dispatch_read_tool({"input": "sample.txt", "offset": 800000, "size": 2000}, visible_labels=set(), artifact_store=ArtifactStore(), retained_store=RetainedOutputStore(), config=config, cwd=home)).replace(str(home), "<HOME>")
    before = "line of text\n" * 10000
    cases["edit diff 10000 lines"] = lambda: build_edit_diff(before, before + "extra line\n", "sample.txt")
    cases["config load"] = lambda: dict(load_config())
    cases["model catalog cold"] = lambda: (models_dev.clear_cache(), len(models_dev.catalog()))
    cases["model catalog warm"] = lambda: len(models_dev.catalog())
    lookup_catalog = models_dev.catalog()

    def exact_lookup():
        models_dev._INDEXES.clear()
        return models_dev._find(lookup_catalog, "openrouter", "openai", "openai/gpt-5.5")

    cases["model exact lookup cold index"] = exact_lookup
    read_result = (
        "[READ RESULT]\nReturned size: 200000\nTotal size: 400000\nEOF: false\n\n"
        + "read body\n" * 20000
    )
    cases["read card summary 200k chars"] = lambda: _read_result_summary(read_result)
    label_store = ArtifactStore()
    label_store.reserve_labels({f"saved-{i}" for i in range(10000)})
    batches = [{f"new-{i}"} for i in range(100)]

    def reserve_labels():
        try:
            for batch in batches:
                label_store.reserve_labels(batch)
            return len(label_store._reserved_labels)
        finally:
            for batch in batches:
                label_store._reserved_labels.difference_update(batch)

    cases["reserve 100 batches after 10000 labels"] = reserve_labels
    cases["project context non-git"] = lambda: build_project_context(config, cwd=home)
    html = ("<html><title>Sample</title><body>" + "<p>Paragraph with <b>bold</b> text.</p>" * 2000 + "</body></html>").encode()
    cases["web HTML 2000 paragraphs"] = lambda: web_content_from_bytes("https://example.test", "https://example.test", "text/html", html).text

    def shell():
        result = execute_command("echo benchmark", timeout=5)
        assert result.exit_code == 0, result
        return result.stdout.strip(), result.stderr.strip()

    cases["shell launch and capture"] = shell
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--case", action="append")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--single-core", action="store_true", help="Constrain this process to one logical CPU")
    parser.add_argument("--output", type=Path, default=ROOT / "build/benchmarks/runtime.json")
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")
    cpu = None
    if args.single_core:
        from benchmark_support import pin_single_cpu
        cpu = pin_single_cpu()
    with tempfile.TemporaryDirectory(prefix="jarv-runtime-") as tmp:
        # Set before importing Jarv: its path constants must never point at user data.
        os.environ.update(HOME=tmp, USERPROFILE=tmp, NO_COLOR="1", TERM="xterm-256color", WT_SESSION="jarv-benchmark")
        cases = make_cases(Path(tmp))
        if args.case:
            unknown = set(args.case) - cases.keys()
            if unknown:
                parser.error(f"Unknown cases: {sorted(unknown)}")
            cases = {key: fn for key, fn in cases.items() if key in args.case}
        hashes = {}
        for name, fn in cases.items():
            hashes[name] = digest(fn())
        samples = {name: [] for name in cases}
        rng = random.Random(42)
        for repetition in range(args.reps):
            names = list(cases)
            rng.shuffle(names)
            for name in names:
                gc.collect()
                wall, cpu = time.perf_counter_ns(), time.process_time_ns()
                result = cases[name]()
                cpu_ms = (time.process_time_ns() - cpu) / 1e6
                ms = (time.perf_counter_ns() - wall) / 1e6
                if digest(result) != hashes[name]:
                    raise AssertionError(f"Non-deterministic result: {name}")
                samples[name].append({"wall_ms": ms, "cpu_ms": cpu_ms})
                del result
            print(f"Completed runtime round {repetition + 1}/{args.reps}", flush=True)
        rows = []
        for name, fn in cases.items():
            # Other cases deliberately clear shared catalogs/caches. Warm each
            # workload immediately before its separate allocation measurement.
            fn()
            gc.collect()
            tracemalloc.start()
            result = fn()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append({"name": name, "median_ms": statistics.median(s["wall_ms"] for s in samples[name]),
                         "median_cpu_ms": statistics.median(s["cpu_ms"] for s in samples[name]),
                         "peak_python_bytes": peak, "sha256": hashes[name], "samples": samples[name]})
            del result
        if args.profile:
            import cProfile
            args.profile.parent.mkdir(parents=True, exist_ok=True)
            profiler = cProfile.Profile()
            profiler.enable()
            for fn in cases.values():
                fn()
            profiler.disable()
            profiler.dump_stats(str(args.profile))
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "python": sys.version,
               "platform": platform.platform(), "methodology": __doc__, "reps": args.reps,
               "single_cpu": cpu, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['name']:<34} {row['median_ms']:9.2f} ms  {row['peak_python_bytes']/1024:10.1f} KiB peak")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
