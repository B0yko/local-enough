"""Read and write a run directory: JSON documents and append-only gzip JSONL record files."""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

PREDICTIONS = "predictions.jsonl.gz"


def dumps(data: Any) -> str:
    """Canonical JSON used for every run file: sorted keys, 2-space indent, trailing newline."""
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class RunDir:
    """A run directory such as ``runs/20260928T101500Z`` or the bundled reference run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def file(self, name: str) -> Path:
        return self.path / name

    def exists(self, name: str) -> bool:
        return self.file(name).exists()

    def ensure(self) -> RunDir:
        self.path.mkdir(parents=True, exist_ok=True)
        return self

    def read_json(self, name: str, default: Any = None) -> Any:
        path = self.file(name)
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def write_json(self, name: str, data: Any) -> None:
        self.ensure()
        self.file(name).write_text(dumps(data), encoding="utf-8")

    def merge_json(self, name: str, updates: dict[str, Any]) -> dict[str, Any]:
        merged = _deep_merge(self.read_json(name, {}) or {}, updates)
        self.write_json(name, merged)
        return merged

    def append_records(self, name: str, records: Iterable[dict[str, Any]]) -> int:
        """Append records as one new gzip member (mtime 0, no file name) so files stay reproducible."""
        lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records]
        if not lines:
            return 0
        self.ensure()
        buf = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
            gz.write("".join(lines).encode("utf-8"))
        with self.file(name).open("ab") as fh:
            fh.write(buf.getvalue())
        return len(lines)

    def iter_records(self, name: str) -> Iterator[dict[str, Any]]:
        """Yield records from a (possibly multi-member) gzip JSONL file; a missing file yields nothing."""
        path = self.file(name)
        if not path.exists():
            return
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def predictions(self, pass_: str | None = "A") -> list[dict[str, Any]]:
        """Prediction records, by default only Pass A (the only pass that is scored)."""
        return [r for r in self.iter_records(PREDICTIONS) if pass_ is None or r.get("pass") == pass_]

    def completed_keys(self, name: str = PREDICTIONS) -> set[tuple[str, str, str, str, str]]:
        """(model_id, task, split, item_id, pass) of every recorded call, for ``--resume``."""
        return {
            (r["model_id"], r["task"], r["split"], str(r["item_id"]), r.get("pass", "A"))
            for r in self.iter_records(name)
        }
