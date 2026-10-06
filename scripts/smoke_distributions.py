"""Check wheel and sdist installs outside the checkout, without a real provider."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import threading
import venv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


CHECK_PACKAGE = """
import importlib.metadata
import json
import sys
from pathlib import Path
import jarv
from jarv import models_dev

assert Path(jarv.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), jarv.__file__
version = importlib.metadata.version('jarv')
assert jarv.__version__ == version
catalog = json.loads(models_dev.SNAPSHOT_PATH.read_text(encoding='utf-8'))
model_id = next(iter(catalog['providers']['openai']['models']))
facts = models_dev.lookup('openai', model_id)
assert facts is not None and facts.id == model_id
print(version)
"""

# Installed after pip finishes, so distribution checks cannot contact providers
# even if a regression ignores the isolated configuration or update opt-out.
NETWORK_GUARD = """
import sys

def _loopback_only(event, args):
    if event == 'socket.connect':
        address = args[1]
        if not isinstance(address, tuple) or address[0] not in ('127.0.0.1', '::1'):
            raise RuntimeError('Distribution smoke test attempted external network access')
    if event == 'socket.getaddrinfo' and args[0] not in ('127.0.0.1', '::1', 'localhost'):
        raise RuntimeError('Distribution smoke test attempted external name resolution')

sys.addaudithook(_loopback_only)
"""


def run(command, *, cwd, env, timeout=30):
    result = subprocess.run(
        [str(part) for part in command], cwd=cwd, env=env,
        stdin=subprocess.DEVNULL, capture_output=True, encoding="utf-8",
        errors="replace", timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def smoke_distribution(artifact: Path) -> None:
    print(f"Checking {artifact.name}", flush=True)
    with tempfile.TemporaryDirectory(prefix="jarv-distribution-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        work = root / "work"
        home = root / "home"
        work.mkdir()
        (home / ".jarv").mkdir(parents=True)
        venv.EnvBuilder(with_pip=True).create(environment)
        binary_dir = environment / ("Scripts" if os.name == "nt" else "bin")
        python = binary_dir / ("python.exe" if os.name == "nt" else "python")
        launcher = binary_dir / ("jarv.exe" if os.name == "nt" else "jarv")
        env = {key: value for key, value in os.environ.items()
               if key.upper() not in {"PYTHONPATH", "PYTHONHOME"}}
        env.update(HOME=str(home), USERPROFILE=str(home),
                   PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8",
                   NO_COLOR="1", NO_PROXY="127.0.0.1,localhost", TERM="dumb")
        # Avoid pip reusing a previously built wheel for the sdist check.
        run([python, "-I", "-m", "pip", "install", "--disable-pip-version-check",
             "--no-cache-dir", artifact], cwd=work, env=env, timeout=180)
        site_packages = Path(run(
            [python, "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            cwd=work, env=env,
        ))
        (site_packages / "sitecustomize.py").write_text(NETWORK_GUARD, encoding="utf-8")
        version = run([python, "-I", "-c", CHECK_PACKAGE], cwd=work, env=env)
        actual = run([launcher, "--version"], cwd=work, env=env)
        assert actual == f"jarv {version}", actual

        requests = []

        class Provider(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, payload))
                chunks = [
                    {"id": "smoke", "choices": [{"index": 0, "delta": {"content": "Package OK"},
                                                 "finish_reason": None}]},
                    {"id": "smoke", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
                body = ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                        + "data: [DONE]\n\n").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            config = {
                "provider": "ollama", "model": "package-smoke",
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "api_key": "", "api_keys": {}, "check_updates": False,
                "project_context": False, "audit": False,
            }
            (home / ".jarv" / "config.json").write_text(json.dumps(config), encoding="utf-8")
            result = json.loads(run(
                [launcher, "--incognito", "--non-interactive", "--no-tools", "--no-update-check",
                 "--output-format", "json", "--run-timeout", "15", "Reply with Package OK"],
                cwd=work, env=env,
            ))
            assert result["status"] == "success" and result["text"] == "Package OK", result
            assert result["error"] is None and result["exit_code"] == 0, result
            assert len(requests) == 1, requests
            path, payload = requests[0]
            assert path == "/v1/chat/completions", path
            assert payload["model"] == "package-smoke" and payload["stream"] is True, payload
            assert any(message.get("content") == "Reply with Package OK"
                       for message in payload["messages"]), payload
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
            assert not worker.is_alive(), "Local provider did not stop"
        print(f"Passed {artifact.name}: entry point, bundled catalog, local provider", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    artifacts = []
    for pattern in ("*.whl", "*.tar.gz"):
        matches = list(args.dist_dir.glob(pattern))
        if len(matches) != 1:
            parser.error(f"Expected exactly one {pattern} in {args.dist_dir}; found {len(matches)}")
        artifacts.append(matches[0].resolve())
    for artifact in artifacts:
        smoke_distribution(artifact)


if __name__ == "__main__":
    main()
