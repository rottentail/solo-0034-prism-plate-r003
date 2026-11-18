"""Bounded HTTP client and isolated prism-plate process fixture.

Every journey builds the real binary with the real Go toolchain, starts it
against a fresh temporary data root, drives it only through public HTTP, and
stops it with SIGTERM.  No server package is imported or stubbed.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def go_toolchain() -> str:
    """Locate a Go toolchain without assuming a particular install layout."""
    explicit = os.environ.get("PRISM_GO")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(Path("/tmp/toolchain/go/bin"))
    candidates.append(Path("/usr/local/go/bin"))
    for candidate in candidates:
        go = candidate / "go"
        if go.exists():
            return str(go)
    found = subprocess.run(["bash", "-lc", "command -v go"], capture_output=True, text=True)
    if found.returncode == 0 and found.stdout.strip():
        return found.stdout.strip()
    raise RuntimeError("Go toolchain not found; set PRISM_GO to the go binary directory")


class BuildError(RuntimeError):
    pass


class Server:
    """A real prismplate process on a fresh, temporary persistence root."""

    def __init__(self, binary: str, root: Path, listen_port: int = 0, start: bool = True):
        self.binary = binary
        self.root = root
        self.listen_port = listen_port
        self.url = ""
        self.proc = None
        self.log_path = root / "service.log"
        self._log = None
        if start:
            self.start()

    # -- process lifecycle -------------------------------------------------
    def _free_port(self) -> int:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError("server already running")
        port = self.listen_port or self._free_port()
        self.listen_port = port
        self.url = f"http://127.0.0.1:{port}"
        self.root.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "a")
        self.proc = subprocess.Popen(
            [self.binary, "-listen", f"127.0.0.1:{port}", "-data", str(self.root)],
            stdout=self._log,
            stderr=self._log,
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise BuildError(f"service exited during startup; log:\n{self.log_tail()}")
            try:
                urllib.request.urlopen(self.url + "/healthz", timeout=0.5).close()
                return
            except OSError:
                time.sleep(0.02)
        self.stop()
        raise BuildError(f"service did not become ready; log:\n{self.log_tail()}")

    def stop(self) -> int:
        if self.proc is None:
            return 0
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
                raise RuntimeError("service ignored SIGTERM")
        code = self.proc.returncode
        self.proc = None
        if self._log is not None:
            self._log.close()
            self._log = None
        return code

    def restart(self) -> None:
        self.stop()
        self.start()

    def log_tail(self, limit: int = 2000) -> str:
        try:
            data = self.log_path.read_text(errors="replace")
        except FileNotFoundError:
            return ""
        return data[-limit:]

    # -- HTTP --------------------------------------------------------------
    def call(self, method: str, path: str, data=None, expect: int = 200,
             raw: bool = False, headers: dict | None = None, timeout: float = 45.0):
        if isinstance(data, bytes):
            body = data
            default_headers = {"Content-Type": "application/json"}
        elif data is None:
            body = None
            default_headers = {}
        else:
            body = json.dumps(data).encode()
            default_headers = {"Content-Type": "application/json"}
        merged = dict(default_headers)
        if headers:
            merged.update(headers)
        request = urllib.request.Request(self.url + path, data=body, method=method, headers=merged)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                code = response.status
                payload = response.read()
                response_headers = {k.lower(): v for k, v in response.headers.items()}
        except urllib.error.HTTPError as error:
            code = error.code
            payload = error.read()
            response_headers = {k.lower(): v for k, v in error.headers.items()}
        if code != expect:
            raise AssertionError(
                f"{method} {path}: expected {expect}, observed {code}: {payload[:400]!r}")
        if raw:
            return payload, response_headers
        return json.loads(payload) if payload else None

    # -- convenience -------------------------------------------------------
    def create_plate(self, plate_id: str, title: str, document: dict) -> dict:
        return self.call("POST", "/v1/plates",
                         {"id": plate_id, "title": title, "document": document}, expect=201)

    def revise(self, plate_id: str, expected: int, document: dict) -> dict:
        return self.call("PUT", f"/v1/plates/{plate_id}",
                         {"expected_revision": expected, "document": document}, expect=200)

    def render_sync(self, plate_id: str, revision: int, fmt: str = "png") -> dict:
        return self.call(
            "POST",
            f"/v1/plates/{plate_id}/render?revision={revision}&format={fmt}",
            expect=201,
        )

    def submit(self, plate_id: str, revision: int, fmt: str = "png") -> dict:
        return self.call(
            "POST",
            f"/v1/plates/{plate_id}/exposures?revision={revision}&format={fmt}",
            expect=202,
        )

    def job(self, job_id: str) -> dict:
        return self.call("GET", f"/v1/exposures/{job_id}")

    def cancel(self, job_id: str) -> dict:
        return self.call("DELETE", f"/v1/exposures/{job_id}", expect=202)

    def artifact_bytes(self, artifact_id: str) -> tuple[bytes, dict]:
        return self.call("GET", f"/v1/artifacts/{artifact_id}", raw=True)

    def artifact_headers(self, artifact_id: str) -> dict:
        """Fetch the artifact and return its HTTP metadata (ETag, scene digest).

        The public endpoint returns the image itself, so the only authoritative
        artifact metadata available over HTTP is the header set plus the body.
        """
        _, headers = self.artifact_bytes(artifact_id)
        return {
            "etag": headers.get("etag"),
            "scene_digest": headers.get("x-scene-digest"),
            "content_type": headers.get("content-type"),
        }

    def wait_job(self, job_id: str, states=("succeeded", "cancelled", "failed"),
                 timeout: float = 60.0, poll: float = 0.002) -> tuple[dict, list]:
        """Poll a job until a terminal state, returning every observed state."""
        seen = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.job(job_id)
            state = job["state"]
            if not seen or seen[-1] != state:
                seen.append(state)
            if state in states:
                return job, seen
            time.sleep(poll)
        raise RuntimeError(f"job {job_id} did not finish; last={seen}")


def build_binary(destination: Path) -> str:
    go = go_toolchain()
    env = environment()
    result = subprocess.run(
        [go, "build", "-trimpath", "-o", str(destination), "./cmd/prismplate"],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        raise BuildError(result.stderr[-4000:])
    return str(destination)


def environment() -> dict:
    env = dict(os.environ)
    env.setdefault("GOFLAGS", "-mod=mod")
    env.setdefault("GOCACHE", str(Path(tempfile.gettempdir()) / "prism-go-cache"))
    env.setdefault("GOTOOLCHAIN", "local")
    return env
