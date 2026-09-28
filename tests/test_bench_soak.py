from __future__ import annotations

from pathlib import Path

import pytest
import respx

import fakes
from local_enough.bench.rundir import RunDir
from local_enough.bench.soak import run_soak
from local_enough.config import Config, LocalModel, RouteConfig

LOCAL_URL = "https://local.test/v1"


def _route_cfg() -> RouteConfig:
    return RouteConfig(
        workload_mix={"classification": 0.6, "entity_matching": 0.4},
        reference_monthly_volume=1000,
    )


def _config() -> Config:
    return Config(
        project="test-proj",
        budget_usd=5.0,
        models=[LocalModel(id="local-test", base_url=LOCAL_URL, model="local-model")],
    )


class _FakeClock:
    """Advances by a fixed step on every call, so the soak loop ends without real sleeping."""

    def __init__(self, step: float) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


@pytest.mark.asyncio
async def test_soak_writes_expected_shape(tmp_path: Path) -> None:
    cfg = _config()
    route_cfg = _route_cfg()
    run_dir = tmp_path / "run"
    clock = _FakeClock(step=0.05)

    with respx.mock(assert_all_called=False) as mock:
        fakes.register_chat(mock, LOCAL_URL, fakes.fixed_content_handler("card_swallowed"))
        await run_soak(cfg, "local-test", route_cfg, run_dir, minutes=0.05, concurrency=2, seed=7, clock=clock)

    rd = RunDir(run_dir)
    soak = rd.read_json("soak.json")
    assert "local-test" in soak
    entry = soak["local-test"]
    assert entry["minutes"] == 0.05
    assert entry["concurrency"] == 2
    assert entry["mix"] == {"classification": 0.6, "entity_matching": 0.4}
    assert entry["completions_n"] > 0
    assert isinstance(entry["per_minute_tph"], list)
    assert "first5_tph" in entry and "last5_tph" in entry and "throttle_factor" in entry

    load = rd.read_json("load_samples.json", {})
    assert any(w["name"] == "soak:local-test" for w in load.get("windows", []))


@pytest.mark.asyncio
async def test_soak_rejects_non_local_model(tmp_path: Path) -> None:
    from local_enough.config import CloudModel

    cfg = Config(
        project="p",
        budget_usd=5.0,
        models=[CloudModel(id="cloud-a", base_url=LOCAL_URL, api_key_env="X", model="m")],
    )
    with pytest.raises(ValueError, match="local"):
        await run_soak(cfg, "cloud-a", _route_cfg(), tmp_path / "run", minutes=0.01)


@pytest.mark.asyncio
async def test_soak_period_respects_mix_proportions() -> None:
    from local_enough.bench.soak import _period

    seq = _period({"a": 0.7, "b": 0.3}, seed=7, length=100)
    assert len(seq) == 100
    assert seq.count("a") == 70
    assert seq.count("b") == 30
    # deterministic for the same seed
    assert _period({"a": 0.7, "b": 0.3}, seed=7, length=100) == seq


def test_soak_windows_ignore_the_trailing_partial_minute() -> None:
    from local_enough.bench.soak import soak_windows

    # 20 full minutes at 6,000 tasks/hour plus the requests that finished after the deadline (4 completions).
    per_minute = [6000.0] * 20 + [240.0]
    first5, last5, factor = soak_windows(per_minute, 20)
    assert (first5, last5, factor) == (6000.0, 6000.0, 1.0)
    # a real slowdown in the last full minutes still shows
    slowed = [6000.0] * 15 + [4800.0] * 5 + [240.0]
    assert soak_windows(slowed, 20)[2] == 0.8
