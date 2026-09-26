# Jarv performance audit — 26 September 2026

Four measured optimization passes reduced the largest local stalls while preserving agent inputs and rendered output. The final matrix covers **25 startup scenarios, 36 runtime workloads, and 5 startup scenarios with a 1,000-turn saved session**. All 36 runtime fingerprints match the original implementation; the full regression suite passes.

Baseline: `a6e23faf4da6eac24aa31b920ae91a1179a839be`. Baseline source was exported from that commit and run with the same expanded benchmark scripts as the final working tree. CPython 3.14.6 on Windows 11; final paired results below constrain the benchmark, loopback server, and child processes to logical CPU 0. This checks limited CPU concurrency, not a physical low-end processor.

**Changes retained**

- Count context tokens incrementally during compaction and retained-result budgeting. Summary text, ordering, budgets, retained content, and the stopping boundary are unchanged.
- Style only visible editor rows, with an exact printable-ASCII wrapping path. Unicode, controls, selection, and cursor placement keep their existing rules.
- Render only the transcript entries needed for the current viewport. Saved Markdown parses on first display, and remains available for scrolling and resizing.
- Avoid the heading regex for text without `#`, and avoid an encoding round trip for already-ASCII strings. Text content remains identical.
- Load CLI configuration and command metadata when used, so `--help` and `--version` avoid those imports.

Changes are confined to six production modules. No new dependencies, persistent caches, background services, prompt/schema changes, or timing-policy changes were added. A trial replacing streamed JSON writes with whole-document serialization was rejected: Unicode-heavy writes allocated more memory without a reliable speed improvement. The original transaction, journal, locking and fsync code is retained.

**Runtime measurements**

Medians of seven shuffled rounds after warmup; imports, fixture setup, hashing, and garbage collection are outside the timer. A separate warmed tracemalloc pass records peak additional Python allocations, including the result. These figures are not whole-process RSS. CPU samples are in the raw JSON; Windows process CPU timers are too coarse for precise sub-millisecond comparisons.

| Workload | Before ms | After ms | Before peak KiB | After peak KiB |
| --- | ---: | ---: | ---: | ---: |
| editor short | 0.03 | 0.02 | 2.1 | 2.0 |
| editor long ASCII | 61.34 | 4.46 | 5821.8 | 1992.8 |
| editor long Unicode | 55.50 | 11.42 | 6104.8 | 2275.9 |
| transcript first frame 1000 | 119.98 | 1.64 | 2389.9 | 78.6 |
| transcript warm frame 1000 | 0.96 | 0.89 | 65.3 | 65.3 |
| history render 100 turns | 49.14 | 47.93 | 745.9 | 746.3 |
| session tree 1000 turns | 2.57 | 2.59 | 659.9 | 659.9 |
| session search cold 1000 turns | 17.31 | 14.62 | 6553.2 | 4926.4 |
| session search warm 1000 turns | 0.00 | 0.00 | 0.0 | 0.0 |
| intro animation frame | 0.57 | 0.58 | 97.6 | 97.6 |
| stream 5000 deltas | 148.06 | 5.52 | 528.1 | 528.0 |
| SSE parse 10000 events | 10.53 | 10.60 | 3395.1 | 3395.1 |
| context build 10 turns | 0.59 | 0.55 | 18.8 | 18.8 |
| context build 1000 turns | 2446.47 | 14.73 | 1164.7 | 1081.1 |
| context retained 200 results | 10.23 | 0.83 | 158.0 | 163.1 |
| Anthropic convert 1000 turns | 2.88 | 2.94 | 2026.6 | 2026.6 |
| Gemini convert 1000 turns | 4.84 | 4.81 | 2796.9 | 2796.9 |
| Chat payload 1000 turns | 7.73 | 5.02 | 4527.4 | 2415.8 |
| Responses payload 1000 turns | 6.15 | 3.48 | 3600.0 | 1522.1 |
| Anthropic payload 1000 turns | 11.97 | 8.57 | 6483.8 | 4038.3 |
| Gemini payload 1000 turns | 15.36 | 11.74 | 7821.6 | 5476.3 |
| history load 1000 turns | 22.19 | 18.87 | 6552.2 | 4925.3 |
| history save 1000 turns | 40.62 | 44.36 | 7722.3 | 5661.4 |
| history load Unicode 400k chars | 10.47 | 9.97 | 7299.3 | 7299.3 |
| history save Unicode 400k chars | 24.08 | 25.49 | 8918.9 | 8905.4 |
| usage load all 20000 | 40.13 | 40.52 | 23075.6 | 23075.6 |
| usage load day 20000 | 44.89 | 45.28 | 23075.6 | 23075.6 |
| usage aggregate 20000 | 75.62 | 73.91 | 4.0 | 4.0 |
| read file bounded 1MB | 2.02 | 2.01 | 766.8 | 766.8 |
| edit diff 10000 lines | 2.68 | 2.65 | 1718.1 | 1718.1 |
| config load | 0.85 | 0.81 | 266.4 | 266.4 |
| model catalog cold | 2.00 | 2.20 | 1581.4 | 1581.4 |
| model catalog warm | 0.06 | 0.06 | 1.3 | 1.3 |
| project context non-git | 18.52 | 18.28 | 286.1 | 286.1 |
| web HTML 2000 paragraphs | 16.15 | 15.91 | 825.7 | 825.7 |
| shell launch and capture | 263.54 | 264.15 | 30.9 | 31.0 |

The largest repeatable improvements are context preparation (~166×), first transcript frame (~73×), long ASCII editor redraw (~14×), and streaming preview assembly (~27×). The stream benchmark assembles/coalesces 5,000 deltas; it does not measure terminal painting for each token. Small timing differences should be treated as noise. Disk-write samples overlap substantially (ordinary history: 38–48 ms before, 35–49 ms after); no write-latency improvement is claimed. Their allocation peak fell from 7.5 MiB to 5.5 MiB. Retained budgeting uses about 5 KiB more temporary memory to eliminate repeated scans.

**Fresh-process startup**

Medians of seven rounds. Timers include Python startup and imports. One-shot timing ends when the loopback server has read the first complete request; menu timing ends at the real first refresh. Exit commands measure completion. Each sample restores the same isolated seed data before the timer starts. Menus also receive keys through their real handlers and repaint. TTY detection is simulated at 100×30; output goes through pipes.

| Scenario | Before ms | After ms | After first visible output ms |
| --- | ---: | ---: | ---: |
| one-shot | 330.9 | 329.4 | 330.0 |
| one-shot terminal | 333.4 | 328.6 | 107.0 |
| one-shot saved | 355.9 | 346.1 | 109.2 |
| one-shot resumed | 345.7 | 338.3 | 106.1 |
| headsup | 136.2 | 133.1 | 131.7 |
| headsup saved | 196.5 | 193.5 | 192.1 |
| settings | 139.1 | 137.4 | 135.5 |
| setup | 136.6 | 135.8 | 134.6 |
| sessions | 121.4 | 119.8 | 118.7 |
| tree | 134.1 | 130.6 | 129.3 |
| history | 174.1 | 172.0 | 170.7 |
| usage | 144.2 | 145.5 | 143.7 |
| help | 120.7 | 120.1 | 118.0 |
| about | 181.3 | 179.7 | 178.0 |
| config | 129.0 | 128.0 | 126.1 |
| version | 74.9 | 58.6 | 53.9 |
| CLI help | 74.2 | 59.7 | 54.8 |
| settings print | 148.9 | 147.2 | 132.5 |
| help print | 127.9 | 125.9 | 114.5 |
| about print | 202.1 | 201.4 | 176.7 |
| config print | 138.0 | 137.8 | 126.0 |
| history print | 182.6 | 182.2 | 170.4 |
| usage print | 152.6 | 146.8 | 134.8 |
| sessions print | 128.2 | 126.8 | 117.2 |
| tree print | 123.3 | 121.3 | 111.7 |

Lightweight CLI startup improves by about 20%; most already-small menus and one-shot request timings change little. Terminal one-shot first visible output remains the waiting indicator, not a model token.

**Large saved session startup**

Five rounds per scenario, 1,000 short user/assistant exchanges restored before each sample, one CPU.

| Scenario | Before ms | After ms | After input repainted ms |
| --- | ---: | ---: | ---: |
| one-shot resumed | 371.1 | 358.2 | — |
| headsup saved | 382.7 | 215.3 | 218.3 |
| sessions | 133.1 | 128.0 | 146.9 |
| tree | 151.2 | 146.7 | 150.5 |
| history | 364.0 | 360.0 | 362.7 |

Opening the saved heads-up session is 44% faster (383 → 215 ms). The full-history viewer still renders its complete document, and its timing is essentially unchanged.

**Validation and limits**

- Final full suite: **1,647 passed, 1 skipped, 144 subtests passed**. Baseline: 1,636 passed, 1 skipped, 144 subtests passed.
- All 36 runtime fingerprints match, including complete context output and Chat Completions, Responses, Anthropic and Gemini payloads. Random retained-result handles are normalized in fingerprints while preserving their relationships.
- Added differential tests compare the original compaction algorithm across randomized histories and budgets, including signed reasoning, multimodal results and existing summaries. Rendering checks cover Unicode wrapping, style spans, scrolling, resizing, appended entries, lazy Markdown, and work bounded by visible rows.
- Existing CLI, provider streaming, cancellation, storage conflict/recovery, tool and session tests pass. One full run had an intermittent socket-close assertion failure in the unchanged HTTP cancellation suite; all 22 cancellation tests passed in isolation and the subsequent full suite passed. No cancellation code or assertions were changed.
- Byte-compilation and `git diff --check` pass.
- No live model services, paid API requests, or user data are involved. Provider inference, WAN/TLS latency, actual terminal-emulator painting/keyboard decoding, packaged binaries, reboot-cold disk caches, other operating systems and physical slow machines are outside these measurements. This is a broad benchmark matrix, not a claim of every possible workload or 100% source-line coverage. Literal zero latency cannot be guaranteed.
- Remaining measured costs include full-history rendering, usage aggregation, shell startup and HTTP initialization. The loop stops here because the remaining small-code opportunities have been validated, while further changes would require broader designs or tradeoffs against the requested behaviour and simplicity constraints.

**Reproduce**

From the repository root in PowerShell:

```powershell
.\.venv\dev\Scripts\python.exe scripts/benchmark/benchmark_runtime.py --reps 7 --single-core --output build/benchmarks/runtime.json
.\.venv\dev\Scripts\python.exe scripts/benchmark/benchmark_coldstart.py --reps 7 --menu-input --single-core --output build/benchmarks/coldstart.json
.\.venv\dev\Scripts\python.exe scripts/benchmark/benchmark_coldstart.py --reps 5 --menu-input --single-core --history-turns 1000 --case "headsup saved" --case history --case sessions --case tree --case "one-shot resumed" --output build/benchmarks/large-coldstart.json
$env:TERM = "xterm-256color"
$env:WT_SESSION = "jarv-tests"
Remove-Item Env:NO_COLOR -ErrorAction SilentlyContinue
.\.venv\dev\Scripts\python.exe -m pytest -q
```

Omit `--single-core` for normal CPU scheduling. Use repeated `--case` options to select workloads; both scripts support optional cProfile output. Run timed benchmarks separately from tests and profilers. For baseline comparisons, export the recorded commit and copy the current benchmark scripts into that checkout before running it with the same Python environment.

**Raw local artifacts**

- Runtime: [before](../../build/benchmarks/goal-baseline-runtime-single.json), [after](../../build/benchmarks/goal-final-runtime-single.json).
- Startup: [before](../../build/benchmarks/goal-baseline-coldstart-single.json), [after](../../build/benchmarks/goal-final-coldstart-single.json).
- Large saved sessions: [before](../../build/benchmarks/goal-baseline-large-coldstart.json), [after](../../build/benchmarks/goal-final-large-coldstart.json).
- [Final test results](../../build/benchmarks/goal-tests.xml).

Generated JSON/XML and the exported baseline source live under the ignored `build/benchmarks` directory. This report and the reusable benchmark scripts are part of the source changes.
