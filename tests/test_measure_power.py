from __future__ import annotations

import plistlib

import pytest

from local_enough.bench.power import (
    BatteryReading,
    PowerTelemetryUnavailable,
    probe,
    read_battery,
    watts,
)


def _plist(entry: dict[str, object]) -> bytes:
    return plistlib.dumps([entry])


AC_ENTRY = {
    "ExternalConnected": True,
    "IsCharging": True,
    "FullyCharged": False,
    "CurrentCapacity": 96,
    "MaxCapacity": 100,
    "Voltage": 13342,
    "InstantAmperage": 1215,
    "PowerTelemetryData": {"SystemLoad": 11828},
}


def _battery_entry(*, amperage: int) -> dict[str, object]:
    return {
        "ExternalConnected": False,
        "IsCharging": False,
        "FullyCharged": False,
        "CurrentCapacity": 80,
        "MaxCapacity": 100,
        "Voltage": 12800,
        "InstantAmperage": amperage,
        "PowerTelemetryData": {"SystemLoad": 0},
    }


def test_read_battery_ac_numeric_fields() -> None:
    reading = read_battery(_plist(AC_ENTRY))
    assert reading.ac is True
    assert reading.charging is True
    assert reading.fully_charged is False
    assert reading.charge_pct == pytest.approx(96.0)
    assert reading.system_load_mw == pytest.approx(11828.0)
    assert reading.voltage_mv == pytest.approx(13342.0)
    assert reading.instant_amperage_ma == pytest.approx(1215.0)


def test_read_battery_converts_twos_complement_negative_amperage() -> None:
    # 2**64 - 500 represents -500 mA in unsigned 64-bit form.
    raw = (1 << 64) - 500
    reading = read_battery(_plist(_battery_entry(amperage=raw)))
    assert reading.instant_amperage_ma == pytest.approx(-500.0)


def test_read_battery_missing_fields_raises() -> None:
    entry = dict(AC_ENTRY)
    del entry["InstantAmperage"]
    with pytest.raises(PowerTelemetryUnavailable, match="InstantAmperage"):
        read_battery(_plist(entry))


def test_read_battery_no_entries_raises() -> None:
    with pytest.raises(PowerTelemetryUnavailable):
        read_battery(plistlib.dumps([]))


def test_watts_on_ac_uses_system_load() -> None:
    reading = read_battery(_plist(AC_ENTRY))
    assert watts(reading) == pytest.approx(11.828)


def test_watts_on_ac_without_system_load_raises() -> None:
    entry = dict(AC_ENTRY)
    entry["PowerTelemetryData"] = {}
    reading = read_battery(_plist(entry))
    with pytest.raises(PowerTelemetryUnavailable):
        watts(reading)


def test_watts_on_battery_discharge_is_positive() -> None:
    # Negative amperage (discharging, per the charging-positive convention) -> positive watts.
    raw = (1 << 64) - 2000  # -2000 mA
    reading = read_battery(_plist(_battery_entry(amperage=raw)))
    result = watts(reading)
    assert result > 0
    assert result == pytest.approx(12800 * 2000 / 1e6)


def test_watts_on_battery_charging_is_negative() -> None:
    reading = read_battery(_plist(_battery_entry(amperage=1500)))
    assert watts(reading) < 0


def test_probe_computes_incremental_watts_and_flags() -> None:
    idle_reading = BatteryReading(
        ac=True,
        charging=False,
        fully_charged=True,
        charge_pct=100.0,
        system_load_mw=5000.0,
        voltage_mv=13000.0,
        instant_amperage_ma=0.0,
    )
    load_reading = BatteryReading(
        ac=True,
        charging=False,
        fully_charged=True,
        charge_pct=100.0,
        system_load_mw=20000.0,
        voltage_mv=13000.0,
        instant_amperage_ma=0.0,
    )
    calls: list[str] = []
    # start_reading + 2 idle samples + 2 load samples + end_reading = 6 sampler() calls.
    readings = iter([idle_reading, idle_reading, idle_reading, load_reading, load_reading, idle_reading])

    def sampler() -> BatteryReading:
        return next(readings)

    result = probe(
        idle_s=2.0,
        load_s=2.0,
        start_load=lambda: calls.append("start"),
        stop_load=lambda: calls.append("stop"),
        sampler=sampler,
        sample_interval_s=1.0,
        sleep_fn=lambda _s: None,
    )

    assert result.idle_watts == pytest.approx(5.0)
    assert result.load_watts == pytest.approx(20.0)
    assert result.incremental_watts == pytest.approx(15.0)
    assert result.samples_n == 4
    assert result.mode == "measured"
    assert result.flagged_not_full is False
    assert result.flagged_out_of_range is False
    assert calls == ["start", "stop"]


def test_probe_flags_not_full_when_ac_and_under_100_pct() -> None:
    reading = BatteryReading(
        ac=True,
        charging=True,
        fully_charged=False,
        charge_pct=87.0,
        system_load_mw=5000.0,
        voltage_mv=13000.0,
        instant_amperage_ma=0.0,
    )
    result = probe(
        idle_s=1.0,
        load_s=1.0,
        start_load=lambda: None,
        stop_load=lambda: None,
        sampler=lambda: reading,
        sample_interval_s=1.0,
        sleep_fn=lambda _s: None,
    )
    assert result.flagged_not_full is True


def test_probe_stops_load_even_when_load_phase_raises() -> None:
    idle_reading = BatteryReading(
        ac=True,
        charging=False,
        fully_charged=True,
        charge_pct=100.0,
        system_load_mw=5000.0,
        voltage_mv=13000.0,
        instant_amperage_ma=0.0,
    )
    calls: list[str] = []
    sample_count = 0

    def sampler() -> BatteryReading:
        nonlocal sample_count
        sample_count += 1
        if sample_count > 2:  # first call (start reading) + one idle sample succeed, then load fails
            raise RuntimeError("boom")
        return idle_reading

    with pytest.raises(RuntimeError, match="boom"):
        probe(
            idle_s=1.0,
            load_s=2.0,
            start_load=lambda: calls.append("start"),
            stop_load=lambda: calls.append("stop"),
            sampler=sampler,
            sample_interval_s=1.0,
            sleep_fn=lambda _s: None,
        )
    assert calls == ["start", "stop"]


def test_probe_flags_out_of_range_watts() -> None:
    reading = BatteryReading(
        ac=True,
        charging=False,
        fully_charged=True,
        charge_pct=100.0,
        system_load_mw=150000.0,
        voltage_mv=13000.0,
        instant_amperage_ma=0.0,  # 150 W: implausible
    )
    result = probe(
        idle_s=1.0,
        load_s=1.0,
        start_load=lambda: None,
        stop_load=lambda: None,
        sampler=lambda: reading,
        sample_interval_s=1.0,
        sleep_fn=lambda _s: None,
    )
    assert result.flagged_out_of_range is True
