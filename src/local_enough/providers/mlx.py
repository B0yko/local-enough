"""Start and stop ``mlx_lm.server`` for a pinned Hugging Face snapshot.

Only ``huggingface_hub`` is imported at module scope; ``mlx_lm`` itself is never imported in this
process -- the server runs as ``sys.executable -m mlx_lm.server`` in a subprocess, so a Linux
install (which never installs the ``mlx``/``mlx-lm`` marker dependencies) does not need it either.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import TracebackType

import httpx
from huggingface_hub import snapshot_download


class MlxServerError(RuntimeError):
    """The server failed to start or never became ready."""


class MlxServer:
    """Context manager around one ``mlx_lm.server`` process bound to 127.0.0.1.

    The pinned snapshot is resolved from the local Hugging Face cache (``local_files_only=True``:
    ``local-enough models pull`` is what actually downloads weights). The snapshot directory's
    name is the resolved commit sha, which becomes ``display_model = "<repo>@<sha>"``.
    """

    def __init__(
        self,
        repo: str,
        revision: str = "main",
        *,
        host: str = "127.0.0.1",
        port: int = 8081,
        startup_timeout_s: float = 300.0,
        poll_interval_s: float = 1.0,
    ) -> None:
        self.repo = repo
        self.revision = revision
        self.host = host
        self.port = port
        self.startup_timeout_s = startup_timeout_s
        self.poll_interval_s = poll_interval_s

        self.process: subprocess.Popen[bytes] | None = None
        self.sha: str = ""
        self.snapshot_dir: Path | None = None

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def display_model(self) -> str:
        return f"{self.repo}@{self.sha}"

    def _resolve_snapshot(self) -> Path:
        path = snapshot_download(self.repo, revision=self.revision, local_files_only=True)
        return Path(path)

    def start(self) -> MlxServer:
        self.snapshot_dir = self._resolve_snapshot()
        self.sha = self.snapshot_dir.name

        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        args = [
            sys.executable,
            "-m",
            "mlx_lm.server",
            "--model",
            str(self.snapshot_dir),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        self.process = subprocess.Popen(
            args,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_ready()
        return self

    def _wait_ready(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + self.startup_timeout_s
        last_error: Exception | None = None
        with httpx.Client(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise MlxServerError(
                        f"mlx_lm.server for {self.repo!r} exited during startup (code {self.process.returncode})"
                    )
                try:
                    response = client.get(f"{self.base_url}/models")
                    if response.status_code == 200:
                        return
                except httpx.HTTPError as exc:
                    last_error = exc
                time.sleep(self.poll_interval_s)
        self.stop()
        raise MlxServerError(
            f"mlx_lm.server for {self.repo!r} did not become ready within {self.startup_timeout_s}s"
        ) from last_error

    def stop(self, *, timeout_s: float = 10.0) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout_s)
        self.process = None

    def __enter__(self) -> MlxServer:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
