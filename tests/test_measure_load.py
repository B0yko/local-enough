from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from local_enough.bench.load import LoadSample, LoadSampler


class FakeProcess:
    def __init__(self, pid: int, cpu_values: list[float], rss: int) -> None:
        self.pid = pid
        self._cpu_values = iter(cpu_values)
        self._rss = rss

    def cpu_percent(self, interval: float | None = None) -> float:
        return next(self._cpu_values, 0.0)

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=self._rss)


def _system(monkeypatch: pytest.MonkeyPatch, cpu_pct: float, cores: int, used_bytes: int) -> None:
    monkeypatch.setattr("local_enough.bench.load.psutil.cpu_percent", lambda interval=None: cpu_pct)
    monkeypatch.setattr("local_enough.bench.load.psutil.cpu_count", lambda: cores)
    monkeypatch.setattr("local_enough.bench.load.psutil.virtual_memory", lambda: SimpleNamespace(used=used_bytes))


def test_other_load_is_system_wide_minus_our_own_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_enough.bench.load.os.getloadavg", lambda: (2.5, 2.0, 1.5))
    # 16 cores at 12.5% = 200% of one core system-wide; our process uses 80% of it.
    _system(monkeypatch, cpu_pct=12.5, cores=16, used_bytes=10_000_000_000)
    own = {100: FakeProcess(100, [0.0, 80.0], rss=4_000_000_000)}
    monkeypatch.setattr("local_enough.bench.load.psutil.Process", lambda pid: own[pid])

    sampler = LoadSampler(own_pids_fn=lambda: {100}, prime_s=0.0)
    sample = sampler.sample_once()

    assert sample.load1 == pytest.approx(2.5)
    assert sample.other_cpu_pct == pytest.approx(120.0)
    assert sample.other_mem_gb == pytest.approx(6.0)
    assert sampler.samples == [sample]


def test_processes_of_other_users_are_counted_even_when_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another user's VM is invisible per process (AccessDenied) but shows up in the system-wide counter."""
    monkeypatch.setattr("local_enough.bench.load.os.getloadavg", lambda: (1.0, 1.0, 1.0))
    _system(monkeypatch, cpu_pct=10.0, cores=16, used_bytes=2_000_000_000)

    def process(pid: int) -> FakeProcess:
        raise AssertionError("only our own processes may be inspected")

    monkeypatch.setattr("local_enough.bench.load.psutil.Process", process)
    sample = LoadSampler(own_pids_fn=lambda: set(), prime_s=0.0).sample_once()
    assert sample.other_cpu_pct == pytest.approx(160.0)


def test_sample_once_never_touches_process_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_enough.bench.load.os.getloadavg", lambda: (0.0, 0.0, 0.0))
    _system(monkeypatch, cpu_pct=0.0, cores=4, used_bytes=0)

    class NameExplodingProcess(FakeProcess):
        def name(self) -> str:  # pragma: no cover - must never be called
            raise AssertionError("LoadSampler must never read process names")

    monkeypatch.setattr(
        "local_enough.bench.load.psutil.Process",
        lambda pid: NameExplodingProcess(pid, [0.0, 5.0], rss=500_000_000),
    )
    sampler = LoadSampler(own_pids_fn=lambda: {200}, prime_s=0.0)
    sampler.sample_once()  # must not raise


def test_contaminated_true_over_ten_percent_threshold() -> None:
    sampler = LoadSampler(own_pids_fn=lambda: set())
    sampler.samples = [LoadSample(0.0, 0.0, 50.0, 0.0) for _ in range(9)] + [
        LoadSample(0.0, 0.0, 150.0, 0.0) for _ in range(2)
    ]
    assert sampler.contaminated is True


def test_contaminated_false_under_threshold() -> None:
    sampler = LoadSampler(own_pids_fn=lambda: set())
    sampler.samples = [LoadSample(0.0, 0.0, 50.0, 0.0) for _ in range(19)] + [LoadSample(0.0, 0.0, 150.0, 0.0)]
    assert sampler.contaminated is False


def test_contaminated_false_with_no_samples() -> None:
    sampler = LoadSampler(own_pids_fn=lambda: set())
    assert sampler.contaminated is False


@pytest.mark.asyncio
async def test_start_stop_collects_samples_periodically(monkeypatch: pytest.MonkeyPatch) -> None:
    sampler = LoadSampler(own_pids_fn=lambda: set(), interval_s=0.01)
    calls = 0

    def fake_sample_once() -> LoadSample:
        nonlocal calls
        calls += 1
        sample = LoadSample(t=0.0, load1=0.0, other_cpu_pct=0.0, other_mem_gb=0.0)
        sampler.samples.append(sample)
        return sample

    monkeypatch.setattr(sampler, "sample_once", fake_sample_once)
    await sampler.start()
    await asyncio.sleep(0.05)
    await sampler.stop()

    assert calls >= 2
    assert sampler._task is None


@pytest.mark.asyncio
async def test_stop_without_start_is_safe() -> None:
    sampler = LoadSampler(own_pids_fn=lambda: set())
    await sampler.stop()  # must not raise
    assert sampler.samples == []
