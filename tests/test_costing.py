import math

from local_enough import costing
from local_enough.bench.rundir import PREDICTIONS, RunDir
from local_enough.config import RouteConfig

HW = {
    "name": "box",
    "purchase_price_usd": 3600.0,
    "price_label": "configured",
    "lifetime_years": 3.0,
    "allocation": 1.0,
    "busy_hours_per_day": 8.0,
    "electricity_usd_per_kwh": 0.30,
    "ops_usd_per_month": 0.0,
    "power": {"mode": "measured", "incremental_watts": 36.0, "idle_watts": 10.0},
}


def _run(tmp_path, power=None, soak=None):
    run = RunDir(tmp_path / "run")
    run.write_json(
        "config.json",
        {
            "project": "p",
            "hardware": HW,
            "models": [
                {"id": "loc", "kind": "local"},
                {"id": "big", "kind": "cloud", "model": "vendor/big"},
                {"id": "tfidf", "kind": "baseline", "task": "classification"},
            ],
        },
    )
    run.write_json(
        "latency.json",
        {"loc": {"classification": {"test": {"A": {"tasks_per_hour": 1800.0}, "B": {"tasks_per_hour": 3600.0}}}}},
    )
    if soak is not None:
        run.write_json("soak.json", soak)
    if power is not None:
        run.write_json("power.json", power)
    base = {"task": "classification", "split": "calib", "pass": "A"}
    run.append_records(
        PREDICTIONS,
        [
            {**base, "model_id": "big", "item_id": "1", "cost_usd": 0.002},
            {**base, "model_id": "big", "item_id": "2", "cost_usd": 0.004},
            {**base, "model_id": "tfidf", "item_id": "1", "cost_usd": 0.0},
        ],
    )
    return run


ROUTE = RouteConfig(workload_mix={"classification": 1.0}, reference_monthly_volume=10_000)


def test_local_cost_uses_pass_b_throttle_and_configured_watts_when_probe_missing(tmp_path):
    run = _run(
        tmp_path, power={"loc": {"mode": "unavailable", "reason": "no battery"}}, soak={"loc": {"throttle_factor": 0.5}}
    )
    cost = costing.local_costs(run, ROUTE)[("loc", "classification")]
    assert cost.sustained_tasks_per_hour == 1800.0
    assert cost.watts_label == "configured"
    # fixed = 3600 / 36 + 10 W * 720 h / 1000 * 0.30 = 100 + 2.16
    assert math.isclose(cost.fixed_usd_per_month, 102.16)
    # energy = 36 W / (0.5 tasks/s) = 72 J = 2e-5 kWh -> 6e-6 USD
    assert math.isclose(cost.energy_usd_per_task, 6e-6)
    assert math.isclose(cost.usd_per_task, 102.16 / 10_000 + 6e-6)
    assert math.isclose(cost.capacity_per_month, 1800.0 * 8 * 30)


def test_measured_power_and_missing_soak(tmp_path):
    run = _run(tmp_path, power={"loc": {"mode": "measured", "incremental_watts": 18.0, "idle_watts": 4.0}})
    cost = costing.local_costs(run, ROUTE)[("loc", "classification")]
    assert (cost.watts_label, cost.incremental_watts, cost.idle_watts) == ("measured", 18.0, 4.0)
    assert cost.throttle_factor == 1.0 and "assumed" in cost.throttle_source


def test_cloud_and_baseline_costs(tmp_path):
    run = _run(tmp_path)
    assert math.isclose(costing.cloud_usd_per_task(run)[("big", "classification")], 0.003)
    costs = costing.usd_per_task(run, ROUTE)
    assert costs[("tfidf", "classification")] == 0.0
    assert ("loc", "classification") in costs
    assert costing.provider_of({"kind": "cloud", "model": "vendor/big"}) == "vendor"
