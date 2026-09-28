"""Peak memory of a model server process: macOS ``/usr/bin/footprint``, with a system-delta fallback.

``footprint_bytes`` is the preferred method -- it reports the process's physical footprint,
including Metal/GPU allocations on this machine (verified by comparing it against the model's size
on disk). When that comparison fails on a given machine, ``system_used_bytes`` supports the
alternative idle-baseline delta method instead; either way, the caller records which ``method`` was
used.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Literal

import psutil

Method = Literal["footprint", "system_delta"]

_UNIT_MULTIPLIER = {"B": 1.0, "KB": 1024.0, "MB": 1024.0**2, "GB": 1024.0**3}
_PHYS_FOOTPRINT_RE = re.compile(r"phys_footprint:\s*([\d.]+)\s*(B|KB|MB|GB)", re.IGNORECASE)
_FOOTPRINT_HEADER_RE = re.compile(r"Footprint:\s*([\d.]+)\s*(B|KB|MB|GB)", re.IGNORECASE)


@dataclass(frozen=True)
class MemorySample:
    bytes: int
    method: Method


def _parse_size(amount: str, unit: str) -> int:
    return round(float(amount) * _UNIT_MULTIPLIER[unit.upper()])


def footprint_bytes(pid: int) -> int:
    """Parse ``/usr/bin/footprint <pid>`` output to a byte count.

    Prefers the ``Auxiliary data: phys_footprint:`` line; falls back to the header's
    ``Footprint:`` total if that line is absent.
    """
    result = subprocess.run(["/usr/bin/footprint", str(pid)], capture_output=True, text=True, timeout=30, check=True)
    match = _PHYS_FOOTPRINT_RE.search(result.stdout) or _FOOTPRINT_HEADER_RE.search(result.stdout)
    if not match:
        raise ValueError(f"could not parse /usr/bin/footprint output for pid {pid}")
    return _parse_size(match.group(1), match.group(2))


def system_used_bytes() -> int:
    """System-wide used memory (total - available), for the idle-baseline delta fallback method."""
    vm = psutil.virtual_memory()
    return int(vm.total) - int(vm.available)


def sample_footprint(pid: int) -> MemorySample:
    return MemorySample(bytes=footprint_bytes(pid), method="footprint")


def sample_system_delta(idle_baseline_bytes: int) -> MemorySample:
    return MemorySample(bytes=system_used_bytes() - idle_baseline_bytes, method="system_delta")
