"""Download configured local models into the Hugging Face cache, under a hard size cap.

Refuses (before downloading anything) when the combined size of every ``launch: mlx`` model would
exceed ``local_models_max_gb``. Sibling projects may share the same Hugging Face cache, so the cap
counts only the bytes newly downloaded for this project's models, recorded in ``downloads.json``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, scan_cache_dir
from huggingface_hub import snapshot_download as _hf_snapshot_download

from local_enough.config import MlxLaunch

GIGABYTE = 1_000_000_000


class ModelBudgetExceeded(RuntimeError):
    def __init__(self, total_gb: float, max_gb: float) -> None:
        super().__init__(
            f"configured local models total {total_gb:.2f} GB, exceeding local_models_max_gb={max_gb:.2f} GB"
        )
        self.total_gb = total_gb
        self.max_gb = max_gb


@dataclass(frozen=True)
class ModelDownload:
    repo: str
    revision_sha: str
    bytes: int
    bytes_new: int
    licence: str | None
    pulled_at_utc: str


@dataclass(frozen=True)
class _RepoInfo:
    revision_sha: str
    bytes: int
    licence: str | None


def _repo_info(repo: str, revision: str, api: HfApi) -> _RepoInfo:
    info = api.model_info(repo, revision=revision, files_metadata=True)
    size = sum((getattr(f, "size", None) or 0) for f in (info.siblings or []))
    licence = None
    card = getattr(info, "card_data", None)
    if card is not None:
        licence = getattr(card, "license", None)
        if licence is None and isinstance(card, dict):
            licence = card.get("license")
    sha = getattr(info, "sha", None) or revision
    return _RepoInfo(revision_sha=str(sha), bytes=int(size), licence=licence)


def _cached_bytes(repo: str, revision_sha: str, scan_cache_fn: Callable[[], Any]) -> int:
    try:
        cache_info = scan_cache_fn()
    except Exception:
        return 0
    for repo_info in getattr(cache_info, "repos", []):
        if getattr(repo_info, "repo_id", None) != repo:
            continue
        for revision in getattr(repo_info, "revisions", []):
            commit_hash = getattr(revision, "commit_hash", "") or ""
            if commit_hash == revision_sha or commit_hash.startswith(revision_sha):
                return int(getattr(revision, "size_on_disk", 0))
    return 0


def pull_models(
    models: Sequence[tuple[str, MlxLaunch]],
    *,
    max_gb: float,
    hf_api: HfApi | None = None,
    snapshot_download_fn: Callable[..., Any] = _hf_snapshot_download,
    scan_cache_fn: Callable[[], Any] = scan_cache_dir,
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> list[tuple[str, ModelDownload]]:
    """Size-check then download every ``launch: mlx`` model. Refuses before downloading any of them."""
    api = hf_api or HfApi()
    infos = {model_id: _repo_info(launch.repo, launch.revision, api) for model_id, launch in models}

    total_gb = sum(info.bytes for info in infos.values()) / GIGABYTE
    if total_gb > max_gb:
        raise ModelBudgetExceeded(total_gb, max_gb)

    results: list[tuple[str, ModelDownload]] = []
    for model_id, launch in models:
        info = infos[model_id]
        cached_before = _cached_bytes(launch.repo, info.revision_sha, scan_cache_fn)
        snapshot_download_fn(launch.repo, revision=launch.revision)
        results.append(
            (
                model_id,
                ModelDownload(
                    repo=launch.repo,
                    revision_sha=info.revision_sha,
                    bytes=info.bytes,
                    bytes_new=max(0, info.bytes - cached_before),
                    licence=info.licence,
                    pulled_at_utc=now_fn().strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            )
        )
    return results


def write_downloads_json(run_dir: Path, downloads: Sequence[tuple[str, ModelDownload]]) -> Path:
    """Write ``downloads.json`` keyed by model id, sorted keys, UTF-8."""
    path = run_dir / "downloads.json"
    payload = {model_id: asdict(download) for model_id, download in downloads}
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path
