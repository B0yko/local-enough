"""``env.json``: an allowlisted snapshot of the run environment. Nothing outside the allowlist.

Never stores the serial number, hardware UUID, provisioning UDID, host or computer name, username,
battery serial, process names or arguments, or any filesystem path. ``ioreg``, ``sysctl`` and
``sw_vers`` output is parsed down to plain numbers, strings and booleans before it is kept.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from typing import Any

from local_enough.bench.power import PowerTelemetryUnavailable, read_battery

ALLOWED_TOP_LEVEL_KEYS = frozenset(
    {
        "machine_model",
        "chip",
        "memory_gb",
        "macos_version",
        "macos_build",
        "python_version",
        "mlx_version",
        "mlx_lm_version",
        "local_enough_version",
        "models",
        "power",
        "datasets_sha256",
        "templates_sha256",
        "hardware_price",
        "start_utc",
        "end_utc",
    }
)

ALLOWED_POWER_KEYS = frozenset({"ac", "charge_pct", "charging", "low_power_mode"})
ALLOWED_HARDWARE_PRICE_KEYS = frozenset({"usd", "label", "source_url", "date"})


def _run(args: list[str]) -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()
    except Exception:
        return None


def _machine_model() -> str | None:
    return _run(["sysctl", "-n", "hw.model"])


def _chip() -> str | None:
    return _run(["sysctl", "-n", "machdep.cpu.brand_string"])


def _memory_gb() -> float | None:
    raw = _run(["sysctl", "-n", "hw.memsize"])
    if raw is None:
        return None
    try:
        return round(int(raw) / 2**30, 1)  # GiB, as Apple quotes unified memory (24 GB = 24 GiB)
    except ValueError:
        return None


def _macos_version() -> str | None:
    return _run(["sw_vers", "-productVersion"])


def _macos_build() -> str | None:
    return _run(["sw_vers", "-buildVersion"])


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _low_power_mode() -> bool | None:
    raw = _run(["pmset", "-g"])
    if raw is None:
        return None
    for line in raw.splitlines():
        if "lowpowermode" in line.lower():
            parts = line.split()
            if parts:
                try:
                    return bool(int(parts[-1]))
                except ValueError:
                    return None
    return None


def _power_block() -> dict[str, Any]:
    low_power_mode = _low_power_mode()
    if sys.platform != "darwin":
        return {"ac": None, "charge_pct": None, "charging": None, "low_power_mode": None}
    try:
        reading = read_battery()
    except PowerTelemetryUnavailable:
        return {"ac": None, "charge_pct": None, "charging": None, "low_power_mode": low_power_mode}
    return {
        "ac": reading.ac,
        "charge_pct": round(reading.charge_pct, 1),
        "charging": reading.charging,
        "low_power_mode": low_power_mode,
    }


def collect_env(
    *,
    models: dict[str, str],
    dataset_sha256: dict[str, str],
    template_sha256: dict[str, str],
    hardware_price: dict[str, Any],
    start_utc: str,
    end_utc: str | None = None,
) -> dict[str, Any]:
    """Build the allowlisted ``env.json`` payload for one run (or one pass within a run).

    ``models`` maps model id to ``"<repo>@<sha>"``; ``dataset_sha256``/``template_sha256`` map a
    logical dataset or template name to its sha256 hex digest (never a filesystem path);
    ``hardware_price`` is ``{usd, label, source_url, date}`` from ``config.yaml``'s hardware block.
    """
    from local_enough import __version__ as local_enough_version

    return {
        "machine_model": _machine_model(),
        "chip": _chip(),
        "memory_gb": _memory_gb(),
        "macos_version": _macos_version(),
        "macos_build": _macos_build(),
        "python_version": platform.python_version(),
        "mlx_version": _package_version("mlx"),
        "mlx_lm_version": _package_version("mlx-lm"),
        "local_enough_version": local_enough_version,
        "models": dict(models),
        "power": _power_block(),
        "datasets_sha256": dict(dataset_sha256),
        "templates_sha256": dict(template_sha256),
        "hardware_price": {k: v for k, v in hardware_price.items() if k in ALLOWED_HARDWARE_PRICE_KEYS},
        "start_utc": start_utc,
        "end_utc": end_utc,
    }
