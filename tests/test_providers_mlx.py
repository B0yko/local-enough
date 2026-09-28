from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from local_enough.providers.mlx import MlxServer, MlxServerError


class FakeProcess:
    def __init__(self, pid: int = 4242, exits_immediately: bool = False) -> None:
        self.pid = pid
        self.returncode: int | None = 1 if exits_immediately else None
        self.terminate_calls = 0
        self.kill_calls = 0
        self._wait_timeouts_left = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self._wait_timeouts_left > 0:
            self._wait_timeouts_left -= 1
            raise subprocess.TimeoutExpired(cmd="mlx_lm.server", timeout=timeout or 0)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    d = tmp_path / "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
    d.mkdir()
    return d


def _patch_snapshot_download(monkeypatch: pytest.MonkeyPatch, snapshot_dir: Path) -> None:
    def fake_snapshot_download(repo: str, revision: str = "main", **kwargs: Any) -> str:
        assert kwargs.get("local_files_only") is True
        return str(snapshot_dir)

    monkeypatch.setattr("local_enough.providers.mlx.snapshot_download", fake_snapshot_download)


def test_start_resolves_snapshot_and_waits_for_ready(monkeypatch: pytest.MonkeyPatch, snapshot_dir: Path) -> None:
    _patch_snapshot_download(monkeypatch, snapshot_dir)
    fake = FakeProcess()
    monkeypatch.setattr("local_enough.providers.mlx.subprocess.Popen", lambda *a, **k: fake)

    server = MlxServer("mlx-community/Qwen3-4B-Instruct-2507-4bit", revision="main", port=8081, poll_interval_s=0.01)
    with respx.mock(assert_all_called=True) as mock:
        mock.get("http://127.0.0.1:8081/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
        server.start()

    assert server.sha == snapshot_dir.name
    assert server.display_model == f"mlx-community/Qwen3-4B-Instruct-2507-4bit@{snapshot_dir.name}"
    assert server.base_url == "http://127.0.0.1:8081/v1"
    assert server.pid == 4242
    server.process = fake  # for stop()
    server.stop()
    assert fake.terminate_calls == 1


def test_start_raises_if_process_exits_immediately(monkeypatch: pytest.MonkeyPatch, snapshot_dir: Path) -> None:
    _patch_snapshot_download(monkeypatch, snapshot_dir)
    fake = FakeProcess(exits_immediately=True)
    monkeypatch.setattr("local_enough.providers.mlx.subprocess.Popen", lambda *a, **k: fake)

    server = MlxServer("mlx-community/Qwen3-4B-Instruct-2507-4bit", port=8082, poll_interval_s=0.01)
    with pytest.raises(MlxServerError, match="exited during startup"):
        server.start()


def test_start_raises_after_timeout_and_stops_process(monkeypatch: pytest.MonkeyPatch, snapshot_dir: Path) -> None:
    _patch_snapshot_download(monkeypatch, snapshot_dir)
    fake = FakeProcess()
    monkeypatch.setattr("local_enough.providers.mlx.subprocess.Popen", lambda *a, **k: fake)

    server = MlxServer(
        "mlx-community/Qwen3-4B-Instruct-2507-4bit", port=8083, startup_timeout_s=0.05, poll_interval_s=0.01
    )
    with respx.mock(assert_all_called=False) as mock:
        mock.get("http://127.0.0.1:8083/v1/models").mock(return_value=httpx.Response(500))
        with pytest.raises(MlxServerError, match="did not become ready"):
            server.start()
    assert fake.terminate_calls == 1


def test_stop_kills_after_terminate_timeout(monkeypatch: pytest.MonkeyPatch, snapshot_dir: Path) -> None:
    _patch_snapshot_download(monkeypatch, snapshot_dir)
    fake = FakeProcess()
    fake._wait_timeouts_left = 1
    monkeypatch.setattr("local_enough.providers.mlx.subprocess.Popen", lambda *a, **k: fake)

    server = MlxServer("mlx-community/Qwen3-4B-Instruct-2507-4bit", port=8084, poll_interval_s=0.01)
    with respx.mock(assert_all_called=True) as mock:
        mock.get("http://127.0.0.1:8084/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
        server.start()
    server.stop(timeout_s=0.01)
    assert fake.terminate_calls == 1
    assert fake.kill_calls == 1


def test_context_manager_starts_and_stops(monkeypatch: pytest.MonkeyPatch, snapshot_dir: Path) -> None:
    _patch_snapshot_download(monkeypatch, snapshot_dir)
    fake = FakeProcess()
    monkeypatch.setattr("local_enough.providers.mlx.subprocess.Popen", lambda *a, **k: fake)

    with respx.mock(assert_all_called=True) as mock:
        mock.get("http://127.0.0.1:8085/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
        with MlxServer("mlx-community/Qwen3-4B-Instruct-2507-4bit", port=8085, poll_interval_s=0.01) as server:
            assert server.pid == 4242
    assert fake.terminate_calls == 1
