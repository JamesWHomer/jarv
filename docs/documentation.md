# Working on the documentation

Jarv's documentation is written in Markdown under `docs/` and built with
[Zensical](https://zensical.org/). Hosting is configured for Read the Docs;
the project must be connected there before a hosted address is available.
The [documentation source](https://github.com/JamesWHomer/jarv/tree/main/docs)
can always be read on GitHub.

## Preview locally

Use Python 3.10 or newer; CI uses Python 3.12. From the repository root, create
a separate documentation environment and install the pinned build dependency.

### Windows PowerShell

```powershell
python -m venv .venv/docs
.\.venv\docs\Scripts\python.exe -m pip install -r requirements-docs.txt
.\.venv\docs\Scripts\zensical.exe serve
```

### macOS and Linux

```bash
python3 -m venv .venv/docs
.venv/docs/bin/python -m pip install -r requirements-docs.txt
.venv/docs/bin/zensical serve
```

Open the address printed by Zensical, normally <http://127.0.0.1:8000/>, and
leave the command running while editing. Pages reload when files change.
Press Ctrl+C to stop the preview.

If you already use [uv](https://docs.astral.sh/uv/), you can preview without
creating or activating a virtual environment yourself:

```bash
uv run --no-project --with-requirements requirements-docs.txt zensical serve
```

## Edit and validate

1. Edit the relevant Markdown page in `docs/`.
2. Add new pages to `project.nav` in `zensical.toml`.
3. Link to other pages with relative Markdown paths, such as
   `[Usage](usage.md)`. Zensical turns these into site URLs.
4. Keep the README focused on installation and first use; put detailed guides
   and reference material in `docs/`.
5. Run a clean, strict build before opening a pull request.

With uv:

```bash
uv run --no-project --with-requirements requirements-docs.txt python scripts/build_docs.py
```

With the virtual environment above, run its Python executable with
`scripts/build_docs.py` (for example,
`.venv/docs/bin/python scripts/build_docs.py` on macOS/Linux).
The build checks internal links and heading anchors and fails on warnings.
Generated HTML goes into `site/`; that directory and Zensical's `.cache/`
are ignored by Git. Commit the Markdown, configuration, and dependency file.

## Read the Docs

The repository's `.readthedocs.yaml` installs `requirements-docs.txt`, runs a
clean, strict Zensical build, and copies the generated site into Read the Docs'
HTML output directory. Both Read the Docs and GitHub Actions use Python 3.12.

### Connect the repository

1. Sign in to [Read the Docs Community](https://app.readthedocs.org/).
2. Add `JamesWHomer/jarv` using its GitHub integration, granting the integration
   access to this repository. Confirm the available project slug in the form.
3. Use `main` as the default branch and `.readthedocs.yaml` as the configuration file.
4. Complete the import and check the first `latest` build.
5. Enable pull-request builds in the project's settings if preview builds are wanted.
6. Replace the GitHub documentation links in `README.md` and `pyproject.toml`
   with the assigned public documentation URL after that build succeeds.

See the [Read the Docs import guide](https://docs.readthedocs.com/platform/stable/intro/add-project.html).
The GitHub integration triggers builds when changes are pushed. Release versions
can be activated in Read the Docs once their tags include `.readthedocs.yaml`.

### Site addresses and validation

`scripts/build_docs.py` uses Read the Docs' `READTHEDOCS_CANONICAL_URL` in a
temporary Zensical configuration. This gives each build its assigned domain,
language, and version path, including pull-request previews. Local builds use
`http://localhost:8000/`. The tracked `zensical.toml` is not rewritten by builds.

The [Documentation workflow](https://github.com/JamesWHomer/jarv/actions/workflows/docs.yml)
only validates builds on GitHub. It has no deployment job or Pages permissions.
GitHub Pages should remain disabled for this repository.

To update Zensical, change its pinned version in `requirements-docs.txt`, reinstall
the documentation dependencies, and check a clean build and local preview.
See [Deploying Zensical on Read the Docs](https://docs.readthedocs.com/platform/latest/intro/zensical.html)
for the upstream build instructions.
