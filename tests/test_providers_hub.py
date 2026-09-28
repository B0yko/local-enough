from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from local_enough.cli import app
from local_enough.config import MlxLaunch
from local_enough.providers.hub import (
    ModelBudgetExceeded,
    ModelDownload,
    pull_models,
    write_downloads_json,
)


@dataclass
class _FakeSibling:
    size: int | None


@dataclass
class _FakeCardData:
    license: str | None


@dataclass
class _FakeModelInfo:
    sha: str
    siblings: list[_FakeSibling]
    card_data: _FakeCardData | None


class _FakeHfApi:
    def __init__(self, infos: dict[str, _FakeModelInfo]) -> None:
        self._infos = infos

    def model_info(self, repo: str, revision: str = "main", files_metadata: bool = False) -> _FakeModelInfo:
        return self._infos[repo]


@dataclass
class _FakeRevision:
    commit_hash: str
    size_on_disk: int


@dataclass
class _FakeRepoInfo:
    repo_id: str
    revisions: list[_FakeRevision]


@dataclass
class _FakeCacheInfo:
    repos: list[_FakeRepoInfo]


QWEN_INFO = _FakeModelInfo(
    sha="50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
    siblings=[_FakeSibling(size=2_000_000_000), _FakeSibling(size=278_972_236)],
    card_data=_FakeCardData(license="apache-2.0"),
)
SMALL_INFO = _FakeModelInfo(
    sha="8b403126fc14f14cfc99bb4cfa72ecbc129ea677",
    siblings=[_FakeSibling(size=880_172_064)],
    card_data=_FakeCardData(license="apache-2.0"),
)


def _now() -> datetime:
    return datetime(2026, 9, 28, 10, 0, 0, tzinfo=UTC)


def test_pull_models_downloads_and_records_bytes_new() -> None:
    api = _FakeHfApi({"mlx-community/Qwen3-4B": QWEN_INFO, "mlx-community/Qwen2.5-1.5B": SMALL_INFO})
    downloaded: list[tuple[str, dict[str, Any]]] = []

    def fake_snapshot_download(repo: str, revision: str = "main") -> str:
        downloaded.append((repo, {"revision": revision}))
        return f"/cache/{repo}/{revision}"

    def fake_scan_cache() -> _FakeCacheInfo:
        return _FakeCacheInfo(repos=[])  # nothing cached yet

    models = [
        ("local-qwen3-4b", MlxLaunch(repo="mlx-community/Qwen3-4B", revision="main", port=8081)),
        ("local-qwen2.5-1.5b", MlxLaunch(repo="mlx-community/Qwen2.5-1.5B", revision="main", port=8082)),
    ]
    results = pull_models(
        models,
        max_gb=4.0,
        hf_api=api,
        snapshot_download_fn=fake_snapshot_download,
        scan_cache_fn=fake_scan_cache,
        now_fn=_now,
    )

    assert len(downloaded) == 2
    assert [model_id for model_id, _ in results] == ["local-qwen3-4b", "local-qwen2.5-1.5b"]
    first = dict(results)["local-qwen3-4b"]
    assert first.repo == "mlx-community/Qwen3-4B"
    assert first.revision_sha == "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
    assert first.bytes == 2_278_972_236
    assert first.bytes_new == 2_278_972_236  # nothing cached before
    assert first.licence == "apache-2.0"
    assert first.pulled_at_utc == "2026-09-28T10:00:00Z"


def test_pull_models_bytes_new_excludes_already_cached() -> None:
    api = _FakeHfApi({"mlx-community/Qwen2.5-1.5B": SMALL_INFO})

    def fake_scan_cache() -> _FakeCacheInfo:
        return _FakeCacheInfo(
            repos=[
                _FakeRepoInfo(
                    repo_id="mlx-community/Qwen2.5-1.5B",
                    revisions=[_FakeRevision(commit_hash=SMALL_INFO.sha, size_on_disk=880_172_064)],
                )
            ]
        )

    results = pull_models(
        [("local-small", MlxLaunch(repo="mlx-community/Qwen2.5-1.5B", revision="main", port=8082))],
        max_gb=4.0,
        hf_api=api,
        snapshot_download_fn=lambda repo, revision="main": f"/cache/{repo}",
        scan_cache_fn=fake_scan_cache,
        now_fn=_now,
    )
    _, download = results[0]
    assert download.bytes_new == 0


def test_pull_models_refuses_over_budget_before_downloading() -> None:
    api = _FakeHfApi({"mlx-community/Qwen3-4B": QWEN_INFO, "mlx-community/Qwen2.5-1.5B": SMALL_INFO})
    downloaded: list[str] = []

    def fake_snapshot_download(repo: str, revision: str = "main") -> str:
        downloaded.append(repo)
        return f"/cache/{repo}"

    models = [
        ("a", MlxLaunch(repo="mlx-community/Qwen3-4B", revision="main", port=8081)),
        ("b", MlxLaunch(repo="mlx-community/Qwen2.5-1.5B", revision="main", port=8082)),
    ]
    with pytest.raises(ModelBudgetExceeded):
        pull_models(
            models,
            max_gb=1.0,
            hf_api=api,
            snapshot_download_fn=fake_snapshot_download,
            scan_cache_fn=lambda: _FakeCacheInfo(repos=[]),
            now_fn=_now,
        )
    assert downloaded == []


def test_write_downloads_json_is_sorted_and_keyed_by_model_id(tmp_path: Path) -> None:
    downloads = [
        (
            "local-qwen3-4b",
            ModelDownload(
                repo="mlx-community/Qwen3-4B",
                revision_sha="50d4277",
                bytes=2_278_972_236,
                bytes_new=2_278_972_236,
                licence="apache-2.0",
                pulled_at_utc="2026-09-28T10:00:00Z",
            ),
        )
    ]
    path = write_downloads_json(tmp_path, downloads)
    assert path == tmp_path / "downloads.json"
    text = path.read_text(encoding="utf-8")
    assert '"local-qwen3-4b"' in text
    assert "/Users" not in text
    import json

    payload = json.loads(text)
    assert payload["local-qwen3-4b"]["bytes"] == 2_278_972_236


CONFIG_WITH_LOCAL_MODELS = """
project: test-project
budget_usd: 5.0
models:
  - id: local-qwen3-4b
    kind: local
    launch: {repo: mlx-community/Qwen3-4B, revision: main, port: 8081}
  - id: local-qwen2.5-1.5b
    kind: local
    launch: {repo: mlx-community/Qwen2.5-1.5B, revision: main, port: 8082}
  - id: open-large
    kind: cloud
    model: deepseek/deepseek-chat-v3.1
"""

CONFIG_CLOUD_ONLY = """
project: test-project
budget_usd: 5.0
models:
  - id: open-large
    kind: cloud
    model: deepseek/deepseek-chat-v3.1
"""


def test_cli_models_pull_writes_downloads_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_WITH_LOCAL_MODELS, encoding="utf-8")
    run_dir = tmp_path / "run"

    captured_models: list[str] = []

    def fake_pull_models(models: list[tuple[str, MlxLaunch]], *, max_gb: float) -> list[tuple[str, ModelDownload]]:
        captured_models.extend(model_id for model_id, _ in models)
        assert max_gb == 5.0 or max_gb == 4.0  # default local_models_max_gb, whatever it is configured to
        return [
            (
                model_id,
                ModelDownload(
                    repo=launch.repo,
                    revision_sha="deadbeef",
                    bytes=1_000_000_000,
                    bytes_new=1_000_000_000,
                    licence="apache-2.0",
                    pulled_at_utc="2026-09-28T10:00:00Z",
                ),
            )
            for model_id, launch in models
        ]

    monkeypatch.setattr("local_enough.cli.pull_models", fake_pull_models)

    runner = CliRunner()
    result = runner.invoke(app, ["models", "pull", "--models", str(config_path), "--run", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert captured_models == ["local-qwen3-4b", "local-qwen2.5-1.5b"]
    assert (run_dir / "downloads.json").exists()
    assert "GB" in result.output


def test_cli_models_pull_exits_1_on_budget_exceeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_WITH_LOCAL_MODELS, encoding="utf-8")

    def fake_pull_models(models: list[tuple[str, MlxLaunch]], *, max_gb: float) -> list[tuple[str, ModelDownload]]:
        raise ModelBudgetExceeded(total_gb=5.0, max_gb=max_gb)

    monkeypatch.setattr("local_enough.cli.pull_models", fake_pull_models)

    runner = CliRunner()
    result = runner.invoke(app, ["models", "pull", "--models", str(config_path)])

    assert result.exit_code == 1
    assert "exceeding" in result.output


def test_cli_models_pull_with_no_local_models_is_a_noop(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_CLOUD_ONLY, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["models", "pull", "--models", str(config_path)])

    assert result.exit_code == 0
    assert "nothing to pull" in result.output.lower()
