"""Shared test fixtures: an isolated LOCAL_ENOUGH_HOME and a clean budget/key environment."""

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
