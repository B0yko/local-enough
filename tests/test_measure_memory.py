from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from local_enough.bench.memory import (
    footprint_bytes,
    sample_footprint,
    sample_system_delta,
    system_used_bytes,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fake_run(stdout: str):
    def _run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert args[0] == "/usr/bin/footprint"
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")

    return _run


def test_footprint_bytes_prefers_phys_footprint(monkeypatch: pytest.MonkeyPatch) -> None:
    text = (FIXTURES / "footprint_output.txt").read_text(encoding="utf-8")
    monkeypatch.setattr("local_enough.bench.memory.subprocess.run", _fake_run(text))
    result = footprint_bytes(54321)
    assert result == 2183254 * 1024


def test_footprint_bytes_falls_back_to_header_total(monkeypatch: pytest.MonkeyPatch) -> None:
    text = (FIXTURES / "footprint_output_header_only.txt").read_text(encoding="utf-8")
    monkeypatch.setattr("local_enough.bench.memory.subprocess.run", _fake_run(text))
    result = footprint_bytes(9999)
    assert result == 512000 * 1024


def test_footprint_bytes_raises_on_unparseable_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_enough.bench.memory.subprocess.run", _fake_run("nothing useful here"))
    with pytest.raises(ValueError, match="could not parse"):
        footprint_bytes(1)


def test_sample_footprint_wraps_method_label(monkeypatch: pytest.MonkeyPatch) -> None:
    text = (FIXTURES / "footprint_output.txt").read_text(encoding="utf-8")
    monkeypatch.setattr("local_enough.bench.memory.subprocess.run", _fake_run(text))
    sample = sample_footprint(54321)
    assert sample.method == "footprint"
    assert sample.bytes == 2183254 * 1024


def test_system_used_bytes_is_total_minus_available(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeVM:
        total = 16_000_000_000
        available = 4_000_000_000

    monkeypatch.setattr("local_enough.bench.memory.psutil.virtual_memory", lambda: _FakeVM())
    assert system_used_bytes() == 12_000_000_000


def test_sample_system_delta_subtracts_idle_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeVM:
        total = 16_000_000_000
        available = 4_000_000_000

    monkeypatch.setattr("local_enough.bench.memory.psutil.virtual_memory", lambda: _FakeVM())
    sample = sample_system_delta(idle_baseline_bytes=10_000_000_000)
    assert sample.method == "system_delta"
    assert sample.bytes == 2_000_000_000
