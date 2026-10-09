# Performance audit — 10 October 2026

Reviewed the Python application, its tests, packaging and supporting scripts for
repeated copying, unbounded rendering work, eager indexing and redundant scans.
Retained eight small production changes with output-equivalence checks:

- Buffer Chat Completions tool-argument fragments and join at completion.
- Copy strict-tool schema subtrees once per normalization level that owns them.
- Build fuzzy model indexes only when exact lookup misses.
- Parse read-card summaries without sanitizing or splitting the content body.
- Check new artifact labels against existing stores without copying those stores.
- Copy only visible rows from cached transcript entries, retaining scroll anchors.
- Avoid rendering an entire settings draft before rendering its visible window.
- Calculate each intro wordmark column's color once across all its rows.

Existing storage transactions, file/process boundaries, cancellation policies,
cache invalidation, provider payload semantics and rendering rules are retained.
Further caching and concurrency changes were excluded where invalidation or
ordering could change behavior. No dependencies were added.

## Measurements

CPython 3.14.6 on Windows. The broad runtime comparison used the same expanded
benchmark script against `a3c07c2` and an isolated copy with this patch applied.
Seven shuffled rounds, one logical CPU, imports/fixture setup/hashing outside
timings; allocation peaks were measured separately with tracemalloc.
All **39 output fingerprints match**.

| Runtime workload | Before ms | After ms | Before peak KiB | After peak KiB |
| --- | ---: | ---: | ---: | ---: |
| Exact model lookup, cold index and loaded catalog | 2.52 | 0.11 | 254.4 | 65.7 |
| Read-card summary, 200k-character body | 0.47 | 0.02 | 1341.2 | 1.2 |
| Reserve 100 batches after 10k labels | 13.13 | 0.04 | 512.5 | 0.4 |
| Intro animation frame | 0.59 | 0.53 | 97.6 | 95.0 |

Targeted comparisons used the unchanged pre-audit functions from `56e0660`
(the affected functions are identical at `a3c07c2`). Provider measurements used
nine rounds in separate processes pinned to one logical CPU. UI measurements
used seven alternating before/after rounds in the same interpreter.

| Targeted workload | Before ms | After ms |
| --- | ---: | ---: |
| Streamed tool arguments, 32 KiB | 0.589 | 0.369 |
| Streamed tool arguments, 1 MiB | 201.832 | 11.559 |
| Jarv's default strict tool schemas | 0.094 | 0.066 |
| Nested strict tool schema, depth 7 | 2.967 | 0.767 |
| Cached transcript viewport, 10k rows | 0.081 | 0.0014 |
| Settings draft, 10k lines | 28.458 | 7.326 |

All five targeted provider output hashes match. Large cached transcript copying
now scales with visible rows; it does not eliminate initial Markdown rendering.
A small 12-row viewport adds approximately 0.1 microsecond of bookkeeping.
These are local CPU microbenchmarks, not end-to-end model-response latency.
Small differences in unchanged workloads are noise; no general speedup is
claimed for disk I/O, shell startup, network requests or inference.

## Validation and reproduction

Regression tests cover fragmented/interleaved calls, schema copying and ordering,
lookup precedence, malformed/Unicode read headers, atomic label conflicts,
scrolling/anchors, bounded draft rendering and exact animation characters/colors.
Independent comparisons also matched 3,000 randomized transcript windows and
500 generated schema conversions against the original implementation.

The isolated full suite passed: **3,322 tests, 6 skipped, 179 subtests passed**.
Byte compilation, dependency compatibility (`pip check`) and diff whitespace
checks passed.

The test and runtime snapshots exclude unrelated edits being made concurrently
in the shared checkout. Existing editor/tree work was preserved.

```powershell
.\.venv\dev\Scripts\python.exe -m pytest -q
.\.venv\dev\Scripts\python.exe scripts/benchmark/benchmark_runtime.py --reps 7 --single-core --output build/benchmarks/runtime.json
```

For before/after comparisons, export the baseline and copy the current benchmark
script into it before running with the same interpreter. The benchmark uses a
temporary home and offline fixtures. The added model-index, read-card and label
cases are reusable alongside the existing runtime matrix.

Local, ignored captures are under `build/benchmarks/oct10/`: `runtime-before.json`,
`runtime-after.json`, `provider-before.json`, `provider-after.json`,
`ui_compare.json`, the targeted measurement drivers, and `tests.xml`.
