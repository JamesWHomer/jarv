# Development

The test workflow installs the checkout in editable mode with `pytest` and runs on Python 3.12 on Windows and Linux. The benchmark reports use a development environment at `.venv/dev`. To create it from the repository root on Windows:

```powershell
python -m venv .venv/dev
.\.venv\dev\Scripts\python.exe -m pip install -e . pytest
$env:TERM = "xterm-256color"
$env:WT_SESSION = "jarv-tests"
Remove-Item Env:NO_COLOR -ErrorAction SilentlyContinue
.\.venv\dev\Scripts\python.exe -m pytest -q
```

On macOS/Linux, use `.venv/dev/bin/python` instead and run tests with `TERM=xterm-256color` and `NO_COLOR` unset. See the [demo recording guide](https://github.com/JamesWHomer/jarv/blob/main/demos/README.md) and dated [performance audit](https://github.com/JamesWHomer/jarv/blob/main/scripts/benchmark/performance-audit.md). Benchmark timings and test counts in those reports describe their recorded revisions, not the current checkout; raw captures under `build/benchmarks/` are local, ignored artifacts.

## Runtime dependencies

| Package | Role |
| --- | --- |
| [httpx](https://pypi.org/project/httpx/) | Direct provider API transports |
| [packaging](https://pypi.org/project/packaging/) | Version parsing and update comparisons |
| [pypdf](https://pypi.org/project/pypdf/) | Lazy-loaded embedded-text extraction for PDF reads |
| [rich](https://pypi.org/project/rich/) | Terminal styling, live rendering, markdown |

Model metadata comes from [models.dev](https://models.dev) (MIT), vendored as a snapshot — no runtime package dependency.

## Documentation

See [Working on the documentation](documentation.md) to edit Markdown, preview the site, and validate changes before publishing.
