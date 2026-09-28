"""Incremental-watts measurement from ``ioreg -a -rn AppleSmartBattery``, without sudo.

The fields read here (``PowerTelemetryData.SystemLoad``, ``InstantAmperage``, ...) are observed on
the reference machine but undocumented by Apple, so every result is an estimate: it is sanity-range
checked and, when the fields are missing entirely (a desktop Mac with no battery, or a non-Apple-
Silicon ``ioreg``), :class:`PowerTelemetryUnavailable` is raised with a clear message rather than
returning a silently wrong number.
"""

from __future__ import annotations

import plistlib
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean

_TWO_POW_64 = 1 << 64
_TWO_POW_63 = 1 << 63


class PowerTelemetryUnavailable(RuntimeError):
    """Raised when a required battery-telemetry field is missing."""


@dataclass(frozen=True)
class BatteryReading:
    ac: bool
    charging: bool
    fully_charged: bool
    charge_pct: float
    system_load_mw: float | None
    voltage_mv: float
    instant_amperage_ma: float


def _to_signed_64(value: int) -> int:
    """Convert an unsigned 64-bit ``InstantAmperage`` reading to its signed two's-complement value."""
    value &= _TWO_POW_64 - 1
    return value - _TWO_POW_64 if value >= _TWO_POW_63 else value


def read_battery(raw_plist: bytes | None = None) -> BatteryReading:
    """Parse one ``ioreg -a -rn AppleSmartBattery`` snapshot into numbers only.

    Pass ``raw_plist`` (the raw XML/binary plist bytes ``ioreg`` prints) in tests; otherwise it is
    read live from the system. Raises :class:`PowerTelemetryUnavailable` if a required field --
    ``ExternalConnected``, ``IsCharging``, ``CurrentCapacity``, ``Voltage`` or ``InstantAmperage`` --
    is missing.
    """
    if raw_plist is None:
        raw_plist = subprocess.run(
            ["ioreg", "-a", "-rn", "AppleSmartBattery"], capture_output=True, timeout=10, check=True
        ).stdout

    try:
        data = plistlib.loads(raw_plist)
    except Exception as exc:
        raise PowerTelemetryUnavailable(f"could not parse ioreg AppleSmartBattery output: {exc}") from exc

    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict) or not data:
        raise PowerTelemetryUnavailable("no AppleSmartBattery entry found (desktop Mac or unsupported machine?)")

    required = ("ExternalConnected", "IsCharging", "CurrentCapacity", "Voltage", "InstantAmperage")
    missing = [key for key in required if key not in data]
    if missing:
        raise PowerTelemetryUnavailable(f"AppleSmartBattery is missing fields: {', '.join(missing)}")

    max_capacity = data.get("MaxCapacity") or 100
    charge_pct = float(data["CurrentCapacity"]) / float(max_capacity) * 100.0

    telemetry = data.get("PowerTelemetryData") or {}
    system_load = telemetry.get("SystemLoad")

    return BatteryReading(
        ac=bool(data["ExternalConnected"]),
        charging=bool(data["IsCharging"]),
        fully_charged=bool(data.get("FullyCharged", False)),
        charge_pct=charge_pct,
        system_load_mw=float(system_load) if system_load is not None else None,
        voltage_mv=float(data["Voltage"]),
        instant_amperage_ma=float(_to_signed_64(int(data["InstantAmperage"]))),
    )


def watts(sample: BatteryReading) -> float:
    """Instantaneous watts: AC uses ``SystemLoad``; battery uses ``V x I``, signed so discharge > 0."""
    if sample.ac:
        if sample.system_load_mw is None:
            raise PowerTelemetryUnavailable("PowerTelemetryData.SystemLoad is missing while on AC power")
        return sample.system_load_mw / 1000.0
    # InstantAmperage was observed positive while charging (current flowing into the battery), so
    # unplugged discharge is negative in that convention; negate to report discharge as positive.
    return -(sample.voltage_mv * sample.instant_amperage_ma) / 1e6


@dataclass(frozen=True)
class PowerProbeResult:
    idle_watts: float
    load_watts: float
    incremental_watts: float
    samples_n: int
    charge_pct_start: float
    charge_pct_end: float
    flagged_not_full: bool
    flagged_out_of_range: bool
    mode: str = "measured"


def _collect(
    sampler: Callable[[], BatteryReading], duration_s: float, interval_s: float, sleep_fn: Callable[[float], None]
) -> list[BatteryReading]:
    samples: list[BatteryReading] = []
    elapsed = 0.0
    while elapsed < duration_s:
        samples.append(sampler())
        sleep_fn(interval_s)
        elapsed += interval_s
    return samples


def probe(
    idle_s: float,
    load_s: float,
    start_load: Callable[[], None],
    stop_load: Callable[[], None],
    sampler: Callable[[], BatteryReading],
    *,
    sample_interval_s: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> PowerProbeResult:
    """60 s idle then sustained load, per the measurement protocol: incremental = mean(load) - mean(idle)."""
    start_reading = sampler()
    idle_samples = _collect(sampler, idle_s, sample_interval_s, sleep_fn)

    start_load()
    try:
        load_samples = _collect(sampler, load_s, sample_interval_s, sleep_fn)
    finally:
        stop_load()
    end_reading = sampler()

    idle_watts = mean(watts(s) for s in idle_samples)
    load_watts = mean(watts(s) for s in load_samples)
    incremental = load_watts - idle_watts

    out_of_range = not (0 < idle_watts < 100) or not (0 < load_watts < 100)
    not_full = start_reading.ac and start_reading.charge_pct < 100.0

    return PowerProbeResult(
        idle_watts=idle_watts,
        load_watts=load_watts,
        incremental_watts=incremental,
        samples_n=len(idle_samples) + len(load_samples),
        charge_pct_start=start_reading.charge_pct,
        charge_pct_end=end_reading.charge_pct,
        flagged_not_full=not_full,
        flagged_out_of_range=out_of_range,
        mode="measured",
    )
