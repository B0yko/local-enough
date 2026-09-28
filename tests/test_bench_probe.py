from __future__ import annotations

from pathlib import Path

import pytest
import respx

import fakes
from local_enough.bench.power import BatteryReading, PowerTelemetryUnavailable
from local_enough.bench.probe import run_probe
from local_enough.bench.rundir import RunDir
from local_enough.config import Config, LocalModel, RouteConfig

LOCAL_URL = "https://local.test/v1"


def _route_cfg() -> RouteConfig:
    return RouteConfig(workload_mix={"classification": 1.0}, reference_monthly_volume=1000)


def _config() -> Config:
    return Config(
        project="p", budget_usd=5.0, models=[LocalModel(id="local-test", base_url=LOCAL_URL, model="local-model")]
    )


def _reading(system_load_mw: float, *, charge_pct: float = 100.0) -> BatteryReading:
    return BatteryReading(
        ac=True,
        charging=False,
        fully_charged=True,
        charge_pct=charge_pct,
        system_load_mw=system_load_mw,
        voltage_mv=13000.0,
        instant_amperage_ma=0.0,
    )


@pytest.mark.asyncio
async def test_probe_writes_power_json(tmp_path: Path) -> None:
    cfg = _config()
    run_dir = tmp_path / "run"
    # The first reading is the telemetry pre-check that runs before any model is loaded.
    readings = iter([_reading(5000.0), _reading(5000.0), _reading(5000.0), _reading(20000.0), _reading(20000.0)])

    def sampler() -> BatteryReading:
        return next(readings, _reading(20000.0))

    with respx.mock(assert_all_called=False) as mock:
        route = fakes.register_chat(mock, LOCAL_URL, fakes.fixed_content_handler("card_swallowed"))
        await run_probe(
            cfg, "local-test", _route_cfg(), run_dir, idle_s=1.0, load_s=1.0, concurrency=2, sampler_fn=sampler
        )

    assert route.call_count >= 1  # at least the warm-up call

    rd = RunDir(run_dir)
    power = rd.read_json("power.json")
    entry = power["local-test"]
    assert entry["idle_watts"] == pytest.approx(5.0)
    assert entry["load_watts"] == pytest.approx(20.0)
    assert entry["incremental_watts"] == pytest.approx(15.0)
    assert entry["mode"] == "measured"
    assert "charge_pct_start" in entry
    assert "flagged_not_full" in entry

    load = rd.read_json("load_samples.json", {})
    assert any(w["name"] == "power-probe:local-test" for w in load.get("windows", []))


@pytest.mark.asyncio
async def test_probe_reports_unavailable_telemetry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _config()
    run_dir = tmp_path / "run"

    def sampler() -> BatteryReading:
        raise PowerTelemetryUnavailable("no AppleSmartBattery entry found")

    with respx.mock(assert_all_called=False) as mock:
        route = fakes.register_chat(mock, LOCAL_URL, fakes.fixed_content_handler("card_swallowed"))
        await run_probe(cfg, "local-test", _route_cfg(), run_dir, idle_s=0.01, load_s=0.01, sampler_fn=sampler)
    assert route.call_count == 0  # no model is loaded or called when there is no telemetry

    rd = RunDir(run_dir)
    power = rd.read_json("power.json")
    assert power["local-test"]["mode"] == "unavailable"
    assert "reason" in power["local-test"]
    out = capsys.readouterr().out
    assert "configured watts" in out


@pytest.mark.asyncio
async def test_probe_rejects_non_local_model(tmp_path: Path) -> None:
    from local_enough.config import CloudModel

    cfg = Config(
        project="p", budget_usd=5.0, models=[CloudModel(id="cloud-a", base_url=LOCAL_URL, api_key_env="X", model="m")]
    )
    with pytest.raises(ValueError, match="local"):
        await run_probe(cfg, "cloud-a", _route_cfg(), tmp_path / "run", idle_s=0.01, load_s=0.01)
