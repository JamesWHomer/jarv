# One-shot startup performance

Measured 10 September 2026 on Windows 11 (build 26200), CPython 3.14.6, using
the checkout's `.venv/dev` environment. The baseline includes the working-tree
changes present before the startup optimization task; it is not pristine HEAD.

Each result is the median of 15 fresh Python processes. The three cases are
shuffled with a fixed seed each round. No profiler ran during these measurements.

| Mode | First visible output before | After | Reduction | Request arrival before | After | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Incognito, piped stdout | 393.3 ms | 258.9 ms | 34.2% | 392.4 ms | 258.0 ms | 34.2% |
| Incognito, terminal | 399.7 ms | 109.8 ms | 72.5% | 398.0 ms | 255.3 ms | 35.9% |
| Fresh saved session, terminal | 420.7 ms | 108.9 ms | 74.1% | 418.9 ms | 273.2 ms | 34.8% |

Request-arrival ranges (minimum–maximum) were 373.1–441.3 → 243.9–272.7 ms
for piped output, 377.6–415.2 → 242.4–281.2 ms for incognito terminal output,
and 393.1–466.2 → 259.6–289.6 ms for saved terminal output.

## Method

- Timing starts immediately before launching Python and includes interpreter
  startup, imports, CLI dispatch, client initialization, and request preparation.
- The parent drains stdout and stderr concurrently and timestamps the first
  non-whitespace stdout content after removing terminal escape sequences.
  The child's normal buffering is preserved.
- Request timing ends when an Ollama-compatible server on `127.0.0.1` has read
  the first complete `/v1/chat/completions` POST body. This is an upper bound on
  time until the request is sent, including local socket/receiver overhead.
- The mock immediately streams `OK`. No real provider, API key, or inference
  latency is involved. Piped stdout first output is the completed reply;
  terminal first output is the waiting indicator after the optimization.
- Terminal detection is simulated at 100×30 with `NO_COLOR=1`, with output
  captured through pipes. These are not terminal-emulator paint measurements.
- Each benchmark run uses an isolated temporary home and seeded session.
  Saved cases use `--new`; incognito cases use `--incognito`. Project context
  and Git collection remain enabled in the actual checkout. Update checks are
  disabled to avoid unrelated background network work.
- OS filesystem and bytecode caches are not cleared. These measure fresh-process
  startup, not the first launch after reboot, a packaged executable, or WAN TLS.

## Changes

1. Extract the small waiting indicator so CLI startup can render it before
   importing the agent. Paint synchronously instead of waiting for Rich's
   250 ms refresh interval. Hand the live indicator and elapsed timer to the
   agent, and restore the terminal on import failure or cancellation.
2. Import Markdown only when rendering content. Import HTTPX in the catalog
   and web tools only when making requests, allowing HTTP setup to overlap
   with instruction collection.
3. Collect system/project instructions in one worker while the main thread
   initializes the HTTP client. Wait for the complete instructions before
   constructing/sending the model request. Reuse supplied clients as before.
4. Load updater code only when a pending update/uninstall result exists, and
   defer standalone update discovery until an update check actually needs it.

## Reproduce

From the repository root in PowerShell, after creating the
[development environment](../../README.md#development):

```powershell
.\.venv\dev\Scripts\python.exe scripts/benchmark/benchmark_coldstart.py --case one-shot --case "one-shot terminal" --case "one-shot saved" --reps 15 --output build/benchmarks/oneshot-after.json
```

Raw local samples: `build/benchmarks/oneshot-before.json` and
`build/benchmarks/oneshot-after.json` (paths relative to the repository root).
These ignored captures are not included in a clone; rerunning the command above
generates a new after sample rather than recreating the historical measurements.

## Validation

Full suite: **1,340 passed, 1 skipped, 117 subtests passed**. Tests ran with
`TERM=xterm-256color`, `WT_SESSION=jarv-tests`, and `NO_COLOR` unset, matching
the existing terminal-rendering tests' assumptions.

New regression tests enforce deferred heavy imports, synchronous first paint,
quiet piped output, overlapping initialization, complete instruction delivery,
client reuse, cleanup after context failure, and cancellation during CLI imports.
The existing CLI, provider, streaming, project-context, and tool tests pass.
