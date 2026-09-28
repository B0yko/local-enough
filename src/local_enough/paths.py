"""Filesystem locations: the state home, bundled package data and run directories."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

REFERENCE = "reference"


def home() -> Path:
    """``$LOCAL_ENOUGH_HOME`` or ``~/.local/share/local-enough``."""
    env = os.environ.get("LOCAL_ENOUGH_HOME")
    return Path(env).expanduser() if env else Path.home() / ".local" / "share" / "local-enough"


def ledger_path(project: str) -> Path:
    return home() / "ledgers" / f"{project}.jsonl"


def data_dir() -> Path:
    return Path(str(resources.files("local_enough") / "data"))


def datasets_dir() -> Path:
    return data_dir() / "datasets"


def init_templates_dir() -> Path:
    return data_dir() / "init"


def reference_run_dir() -> Path:
    return data_dir() / REFERENCE


def new_run_dir(base: Path | None = None) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (base or Path("runs")) / stamp


def resolve_run(run: str | None, base: Path | None = None) -> Path:
    """``reference`` -> bundled run; None -> latest ``./runs/*``; otherwise the given directory."""
    if run == REFERENCE:
        return reference_run_dir()
    if run:
        return Path(run)
    root = base or Path("runs")
    runs = sorted(p for p in root.glob("*") if p.is_dir()) if root.exists() else []
    if not runs:
        raise FileNotFoundError("no run directory found under ./runs; pass --run <dir> or --run reference")
    return runs[-1]
