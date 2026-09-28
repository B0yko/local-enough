"""Background-load sampling for the contamination check: numbers only, never process names.

Other work may be running on the same machine during a measurement window. ``LoadSampler`` records
the 1-minute load average plus the CPU% and memory in use by everything that is not ours (system-wide counters,
so other users' processes are included), so a run can be flagged ``contaminated`` and rerun once the machine is
quiet.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import psutil

_CONTAMINATION_CPU_PCT = 100.0  # one core-equivalent
_CONTAMINATION_SAMPLE_FRACTION = 0.10


@dataclass(frozen=True)
class LoadSample:
    t: float
    load1: float
    other_cpu_pct: float
    other_mem_gb: float


class LoadSampler:
    """Periodic sampler of load average and other-process CPU/memory, numbers only."""

    def __init__(self, own_pids_fn: Callable[[], set[int]], interval_s: float = 60.0, prime_s: float = 1.0) -> None:
        self.own_pids_fn = own_pids_fn
        self.interval_s = interval_s
        self.prime_s = prime_s
        self._primed = False
        self.samples: list[LoadSample] = []
        self._processes: dict[int, psutil.Process] = {}
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    def sample_once(self) -> LoadSample:
        """Other work = system-wide CPU and memory in use minus our own processes.

        Per-process accounting would miss processes of other users (on macOS they raise AccessDenied without
        root), so the system-wide counters, which cover every user, are the reference and only our own processes
        (which are always readable) are subtracted.
        """
        if not self._primed:
            # CPU counters report usage since their previous read, and a process's first read is always 0.0, so
            # the first sample would charge everything since the last read (e.g. our own model loading) to "other".
            # Prime every counter, then measure over a short fresh interval.
            self._track(self.own_pids_fn())  # primes each of our processes
            psutil.cpu_percent(None)
            time.sleep(self.prime_s)
            self._primed = True
        load1 = os.getloadavg()[0]
        own_pids = self.own_pids_fn()
        self._track(own_pids)
        system_cpu_pct = psutil.cpu_percent(None) * (psutil.cpu_count() or 1)
        own_cpu_pct = 0.0
        own_mem_bytes = 0
        for pid, proc in list(self._processes.items()):
            try:
                own_cpu_pct += proc.cpu_percent(None)
                own_mem_bytes += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                del self._processes[pid]
        other_cpu_pct = max(0.0, system_cpu_pct - own_cpu_pct)
        other_mem_bytes = max(0, psutil.virtual_memory().used - own_mem_bytes)

        sample = LoadSample(t=time.time(), load1=load1, other_cpu_pct=other_cpu_pct, other_mem_gb=other_mem_bytes / 1e9)
        self.samples.append(sample)
        return sample

    def _track(self, own_pids: set[int]) -> None:
        """Keep a primed ``psutil.Process`` for each of our pids (a new process's first read is 0.0)."""
        for pid in list(self._processes):
            if pid not in own_pids:
                del self._processes[pid]
        for pid in own_pids - set(self._processes):
            try:
                self._processes[pid] = psutil.Process(pid)
                self._processes[pid].cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    async def start(self) -> None:
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        # Wait first: callers take their "before" sample explicitly, and a sample right after another one would
        # measure a near-zero interval.
        while not self._stop_event.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_s)
            if not self._stop_event.is_set():
                self.sample_once()

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task
            self._task = None

    @property
    def contaminated(self) -> bool:
        if not self.samples:
            return False
        bad = sum(1 for s in self.samples if s.other_cpu_pct > _CONTAMINATION_CPU_PCT)
        return (bad / len(self.samples)) > _CONTAMINATION_SAMPLE_FRACTION
