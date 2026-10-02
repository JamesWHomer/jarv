# Working on the documentation

Jarv's documentation is written in Markdown under `docs/` and built with
[Zensical](https://zensical.org/). The published site is
[jameshomer.dev/jarv](https://jameshomer.dev/jarv/).

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
4. Keep overlapping examples in the repository README consistent with the guides.
5. Run a clean, strict build before opening a pull request.

With uv:

```bash
uv run --no-project --with-requirements requirements-docs.txt zensical build --clean --strict
```

With the virtual environment above, replace `serve` with `build --clean --strict`.
The build checks internal links and heading anchors and fails on warnings.
Generated HTML goes into `site/`; that directory and Zensical's `.cache/`
are ignored by Git. Commit the Markdown, configuration, and dependency file.

## GitHub Pages

The [Documentation workflow](https://github.com/JamesWHomer/jarv/actions/workflows/docs.yml)
builds documentation changes in pull requests. Pushes to `main` that change
`docs/`, `zensical.toml`, `requirements-docs.txt`, or the workflow also publish
the resulting site. A manual **Run workflow** on `main` can redeploy it.
Pull requests and manual runs on other branches only build the site.

In the repository's **Settings → Pages → Build and deployment**, the source
must be **GitHub Actions**. The deployment uses the `github-pages` environment
and GitHub's built-in token with `pages: write` and `id-token: write` permissions.
No personal access token or `gh-pages` branch is needed.

The project inherits `jameshomer.dev` from the account's existing GitHub Pages
site, with HTTPS enforced. It does not need its own `CNAME` file. See GitHub's
[custom domain documentation](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages)
for how project sites inherit that domain.

The site URL in `zensical.toml` includes the `/jarv/` repository path. If the
repository moves or its domain changes, update that URL and the documentation
links in `README.md` and `pyproject.toml` together.

To update Zensical, change its pinned version in `requirements-docs.txt`, reinstall
the documentation dependencies, and check a clean build and local preview.
See [Zensical's publishing guide](https://zensical.org/docs/publish-your-site/)
for the upstream deployment instructions.
