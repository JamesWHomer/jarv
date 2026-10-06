# Development

The test workflow runs the suite on Linux with Python 3.10, 3.12, and 3.14; on Windows with Python 3.12 and 3.14; and on macOS with Python 3.12. The benchmark reports use a development environment at `.venv/dev`. To create it from the repository root on Windows:

```powershell
python -m venv .venv/dev
.\.venv\dev\Scripts\python.exe -m pip install -e . pytest
.\.venv\dev\Scripts\python.exe -m pytest -q
```

On macOS/Linux, use `.venv/dev/bin/python` instead. Tests establish their own terminal settings and temporary user directories. See the [demo recording guide](https://github.com/JamesWHomer/jarv/blob/main/demos/README.md) and dated [performance audit](https://github.com/JamesWHomer/jarv/blob/main/scripts/benchmark/performance-audit.md). Benchmark timings and test counts in those reports describe their recorded revisions, not the current checkout; raw captures under `build/benchmarks/` are local, ignored artifacts.

## Distribution checks

CI also builds the wheel and source distribution, installs each into a separate temporary environment, and runs checks from outside the checkout on Windows and Linux. The checks verify the installed command, package version, bundled model catalog, and a complete streamed response from a local test provider. No provider credentials are needed; application network access is restricted to loopback. Installing dependencies can access the package index.

To run the same checks locally:

```powershell
.\.venv\dev\Scripts\python.exe -m pip install build
.\.venv\dev\Scripts\python.exe -m build --outdir build/test-distributions
.\.venv\dev\Scripts\python.exe scripts/smoke_distributions.py --dist-dir build/test-distributions
```

Use an output directory containing exactly one wheel and one source distribution for the version being tested. Release publication is gated on these checks against the exact artifacts that will be uploaded to PyPI.

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
