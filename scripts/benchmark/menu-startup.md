# Menu startup performance

Measured 10 September 2026 on Windows 11, CPython 3.14.6, using the
checkout's `.venv/dev` environment. The baseline is the working tree at the
start of this task, including the earlier one-shot startup improvements.

Each value is the median of **9 fresh Python processes**, with all 11 menus
shuffled using the same fixed seed each round. Profiling was disabled for
the before/after runs.

| Menu | Open before | Open after | Reduction | First key handled after | Input repainted after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heads-up, incognito | 301.4 ms | 134.1 ms | 55.5% | 134.1 ms | 136.6 ms |
| Heads-up, saved session | 306.4 ms | 194.7 ms | 36.5% | 194.8 ms | 196.8 ms |
| Settings | 150.5 ms | 139.5 ms | 7.3% | 139.5 ms | 146.1 ms |
| Setup | 145.0 ms | 138.5 ms | 4.4% | 138.9 ms | 144.9 ms |
| Sessions | 189.6 ms | 127.0 ms | 33.0% | 127.4 ms | 131.3 ms |
| Tree | 201.4 ms | 140.4 ms | 30.3% | 140.4 ms | 142.4 ms |
| History | 203.6 ms | 189.1 ms | 7.1% | 189.1 ms | 191.0 ms |
| Usage | 137.8 ms | 133.9 ms | 2.9% | 134.0 ms | 135.9 ms |
| Help | 132.7 ms | 124.2 ms | 6.5% | 124.2 ms | 126.5 ms |
| About | 195.8 ms | 193.2 ms | 1.3% | 193.3 ms | 195.6 ms |
| Config | 142.6 ms | 127.9 ms | 10.3% | 127.9 ms | 130.2 ms |

The largest improvements are in heads-up, sessions, and tree. Small changes
in other menus overlap normal run-to-run variation; About is essentially
unchanged. Saved heads-up and History still load Markdown to display existing
assistant responses, and About needs it for its reference content.

## What changed

- Heads-up opens without importing the agent or constructing the HTTP client.
  The first submitted prompt performs this work on the existing turn worker;
  subsequent prompts reuse the client. This moves initialization to first
  use, so the first prompt still pays that cost while the menu stays editable.
- Shared status labels live in a small UI module. Drawing the heads-up menu
  no longer imports the tool orchestrator through the agent UI module.
- Session rendering loads Markdown and web-tool labels only when needed.
  Tree logic imports plain-content conversion directly, avoiding rendering
  dependencies entirely.
- Settings and setup read provider metadata from the lightweight catalog.
  Cache-key construction imports only the backend needed for an otherwise
  unspecified endpoint. Update installation metadata is deferred until used.
- Escape is bound before the first worker starts and preserved when control
  passes to the agent. Cancelling during client initialization closes the
  unused client and prevents sending the prompt. Initialization errors appear
  in the menu and allow retry.

## Measurement boundaries

- Timing starts immediately before launching Python and includes interpreter
  startup, imports, configuration/session loading, dispatch, and rendering.
- “Open” ends after the real initial `Live.refresh()`. “First key” ends after
  the real input-loop handler returns. The final column ends after the real
  repaint following the probe's complete input sequence.
- Heads-up types `x`; session search receives Ctrl+F then `x`. Assertions
  verify the resulting editor/search text. Settings/setup open the provider
  picker with Enter then move its selection with Up, without saving anything.
  Read-only screens receive Down. These are different interactions, so their
  repaint columns should not be compared as equal workloads.
- Terminal detection is simulated at 100×30 with `NO_COLOR=1`. Rendering goes
  through pipes; injected key tokens bypass OS keyboard decoding. This measures
  application readiness, not physical keyboard-to-pixel or ConPTY latency.
- Each run uses an isolated temporary home, an Ollama configuration, and a
  seeded one-exchange session. Heads-up is tested both incognito and with the
  saved exchange. Update checks are disabled. No real provider or user data
  is used; menus make no model requests.
- OS and bytecode caches are not cleared. These are fresh-process starts,
  not first launch after reboot or packaged executable measurements. Large
  histories, remote profiles, terminal sizes, and machines can differ.

## Reproduce

Run from the repository root in PowerShell, after creating the
[development environment](../../README.md#development):

```powershell
.\.venv\dev\Scripts\python.exe scripts/benchmark/benchmark_coldstart.py --menu-input --reps 9 --case headsup --case "headsup saved" --case settings --case setup --case sessions --case tree --case history --case usage --case help --case about --case config --output build/benchmarks/menu-after.json
```

Use `--profile build/benchmarks/menu.prof --case headsup --reps 1 --menu-input`
for a diagnostic profile. Do not compare profiled timings with the table.

Raw local samples, ranges, timestamps, and imported-module inventories:
`build/benchmarks/menu-before.json` and `build/benchmarks/menu-after.json`
(paths relative to the repository root). These ignored captures are not included
in a clone; rerunning the command above generates a new after sample. The input probe is benchmark-only and is never
imported by the installed CLI.

## Validation

Full suite: **1,352 passed, 1 skipped, 117 subtests passed**, with
`TERM=xterm-256color` and `NO_COLOR` unset to match terminal-test assumptions.
The new startup tests exercise actual CLI menu input, prohibit unnecessary
agent/network imports, and check slow client setup, typing, cancellation,
connection reuse, failure/retry, and configuration reload before first use.
