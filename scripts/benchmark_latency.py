"""Benchmark Jarv command latency.

This script is intentionally non-mutating by default. Mutating command cases use
an isolated temporary home directory when explicitly enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


BASE_CASES = (
    ("version", ["--version"], 7),
    ("help", ["/help"], 7),
    ("help alias", ["help"], 7),
    ("about", ["/about"], 5),
    ("config", ["/config"], 7),
    ("history", ["/history"], 5),
    ("usage", ["/usage"], 5),
    ("usage day", ["/usage", "day"], 5),
    ("sessions", ["/sessions"], 5),
    ("settings print", ["/settings"], 5),
)

MUTATING_CASES = (
    ("new isolated", ["/new"], 5),
    ("set isolated", ["/set", "check_updates", "false"], 5),
    ("unset isolated", ["/unset", "check_updates"], 5),
)


def _default_executable(use_installed: bool) -> list[str]:
    if use_installed:
        jarv = shutil.which("jarv")
        if not jarv:
            raise SystemExit("jarv was not found on PATH")
        return [jarv]
    return [sys.executable, "-m", "jarv.cli"]


def _isolated_home() -> Path:
    home = Path(tempfile.mkdtemp(prefix="jarv-bench-home-"))
    config_dir = home / ".jarv"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "provider": "ollama",
        "api_key": "",
        "api_keys": {},
        "base_url": "",
        "model": "llama3.2",
        "service_tiers": {},
        "reasoning_effort": "",
        "max_history": 40,
        "context_budget_ratio": 0.75,
        "context_compaction_threshold": 0.85,
        "context_output_reserve_ratio": 0.15,
        "context_window_fallback": 128000,
        "max_stdin_chars": 200000,
        "max_tool_output_chars": 20000,
        "disabled_tools": [],
        "command_timeout": 60,
        "web_timeout": 15,
        "command_safety": "risky",
        "audit": False,
        "auditor_auto_approve": True,
        "auditor_model": "",
        "system_prompt": "You are Jarv, a helpful CLI assistant. Be concise.",
        "max_subagent_depth": 4,
        "subagent_thread_pool_max_workers": 8,
        "check_updates": False,
        "read_only_command_display": "print",
        "tool_call_display": "print",
        "print_usage_after_agent": False,
    }
    (config_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return home


def _env(home: Path | None = None, first_paint: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    if first_paint:
        env["JARV_BENCH_FIRST_PAINT"] = "1"
    if home is not None:
        env["USERPROFILE"] = str(home)
        env["HOME"] = str(home)
    return env


def _run_case(
    base_cmd: list[str],
    name: str,
    args: list[str],
    reps: int,
    *,
    home: Path | None = None,
    timeout: float = 30,
    first_paint: bool = False,
) -> dict:
    times: list[float] = []
    codes: list[str] = []
    first_paint_ms: list[float] = []
    sample = ""
    for index in range(reps):
        started_ns = time.time_ns()
        started = time.perf_counter()
        try:
            result = subprocess.run(
                [*base_cmd, *args],
                cwd=REPO_ROOT,
                env=_env(home, first_paint=first_paint),
                input="",
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - started
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - started
            times.append(elapsed)
            codes.append("timeout")
            sample = sample or f"timeout after {exc.timeout}s"
            break

        times.append(elapsed)
        codes.append(str(result.returncode))
        if index == 0:
            sample = (result.stdout + "\n" + result.stderr).strip()[:500]
        for line in result.stderr.splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[0] == "JARV_FIRST_PAINT":
                try:
                    first_paint_ms.append((int(parts[2]) - started_ns) / 1_000_000)
                except ValueError:
                    pass

    row = {
        "name": name,
        "reps": len(times),
        "codes": sorted(set(codes)),
        "min_ms": min(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
        "mean_ms": statistics.mean(times) * 1000,
        "max_ms": max(times) * 1000,
        "sample": sample,
    }
    if first_paint_ms:
        row["first_paint_median_ms"] = statistics.median(first_paint_ms)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Jarv command latency")
    parser.add_argument("--installed", action="store_true", help="Benchmark jarv from PATH instead of this checkout")
    parser.add_argument("--include-mutating", action="store_true", help="Benchmark mutating commands in an isolated temp home")
    parser.add_argument("--prompt", help="Optional one-shot prompt to benchmark once with --incognito")
    parser.add_argument("--first-paint", action="store_true", help="Enable first-paint stderr instrumentation")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text table")
    args = parser.parse_args()

    base_cmd = _default_executable(args.installed)
    isolated_home = _isolated_home() if args.include_mutating else None
    try:
        rows = [
            _run_case(base_cmd, name, case_args, reps, first_paint=args.first_paint)
            for name, case_args, reps in BASE_CASES
        ]
        if args.include_mutating and isolated_home is not None:
            rows.extend(
                _run_case(base_cmd, name, case_args, reps, home=isolated_home)
                for name, case_args, reps in MUTATING_CASES
            )
        if args.prompt:
            rows.append(
                _run_case(
                    base_cmd,
                    "one-shot prompt",
                    ["--incognito", args.prompt],
                    1,
                    timeout=120,
                )
            )
    finally:
        if isolated_home is not None:
            shutil.rmtree(isolated_home, ignore_errors=True)

    payload = {"command": base_cmd, "results": rows}
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Command: {' '.join(base_cmd)}")
    print(f"{'case':<18} {'median':>10} {'min':>10} {'max':>10}  codes")
    for row in rows:
        print(
            f"{row['name']:<18} {row['median_ms']:>9.1f}ms "
            f"{row['min_ms']:>9.1f}ms {row['max_ms']:>9.1f}ms  "
            f"{','.join(row['codes'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
