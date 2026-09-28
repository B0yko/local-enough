"""Shared test fixtures: an isolated LOCAL_ENOUGH_HOME, a clean budget/key environment, no real caffeinate."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LOCAL_ENOUGH_HOME", str(tmp_path))
    monkeypatch.delenv("LOCAL_ENOUGH_BUDGET_USD", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    yield


@pytest.fixture(autouse=True)
def _no_caffeinate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never spawn the real ``caffeinate``; the commands that do are exercised with fakes."""
    for module in ("runner", "probe", "soak"):
        monkeypatch.setattr(f"local_enough.bench.{module}.spawn_caffeinate", lambda pid: None)
