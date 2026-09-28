"""Keep the machine awake for the duration of a measurement command, on macOS only."""

from __future__ import annotations

import subprocess
import sys


def spawn_caffeinate(pid: int) -> subprocess.Popen[bytes] | None:
    """Spawn ``caffeinate -i -w <pid>`` so the machine stays awake while ``pid`` is alive.

    No-op (returns ``None``) on any platform other than macOS.
    """
    if sys.platform != "darwin":
        return None
    return subprocess.Popen(
        ["caffeinate", "-i", "-w", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
