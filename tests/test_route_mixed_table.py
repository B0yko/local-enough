"""mixed_table's five comparison rows and the shared-machine cost formula's arithmetic."""

from __future__ import annotations

import json
import math

from local_enough import costmodel
from local_enough.bench.rundir import PREDICTIONS, RunDir
from local_enough.config import RouteConfig
from local_enough.route import planner, simulate
from local_enough.tasks.base import TaskSpec

HW = {
    "name": "test-box",
    "purchase_price_usd": 3600.0,
    "price_label": "configured",
    "lifetime_years": 3.0,
    "allocation": 1.0,
    "busy_hours_per_day": 8.0,
    "electricity_usd_per_kwh": 0.30,
    "ops_usd_per_month": 0.0,
    "power": {"mode": "configured", "incremental_watts": 20.0, "idle_watts": 5.0},
}
LOCAL_THROUGHPUT_TPH = 3600.0
REFERENCE_VOLUME = 100_000.0


def _write_jsonl(path, items) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")


def _spec(tmp_path) -> TaskSpec:
    root = tmp_path / "cls"
    root.mkdir()
    spec = TaskSpec(name="classification", kind="classification", labels=["a", "b"])
    spec.root = root
    calib = [{"id": f"c{i}", "text": f"apple calib {i}", "label": "a"} for i in range(4)]
    _write_jsonl(root / "calib.jsonl", calib)
    test = [{"id": f"t{i}", "text": f"apple test {i}", "label": "a"} for i in range(4)]
    _write_jsonl(root / "test.jsonl", test)
    return spec, calib, test


def _setup(tmp_path):
    spec, calib, test = _spec(tmp_path)
    specs = {"classification": spec}
    run = RunDir(tmp_path / "run")
    run.write_json(
        "config.json",
        {
            "project": "p",
            "hardware": HW,
            "models": [
                {"id": "cloud-frontier", "kind": "cloud", "model": "acme/frontier-v1", "role": "frontier"},
                {"id": "cloud-mid", "kind": "cloud", "model": "other/mid-v1"},
                {"id": "local-llm", "kind": "local"},
            ],
        },
    )
    run.write_json(
        "latency.json",
        {"local-llm": {"classification": {"test": {"A": {"tasks_per_hour": LOCAL_THROUGHPUT_TPH}}}}},
    )

    base_calib = {"task": "classification", "split": "calib", "pass": "A", "valid": True, "latency_s": 0.05}
    calib_records = []
    for item in calib:
        calib_records.append(
            {**base_calib, "model_id": "cloud-frontier", "item_id": item["id"], "raw": "a", "cost_usd": 0.02}
        )
        calib_records.append(
            {**base_calib, "model_id": "cloud-mid", "item_id": item["id"], "raw": "a", "cost_usd": 0.005}
        )
        calib_records.append(
            {**base_calib, "model_id": "local-llm", "item_id": item["id"], "raw": "a", "cost_usd": 0.0}
        )
    run.append_records(PREDICTIONS, calib_records)

    base_test = {"task": "classification", "split": "test", "pass": "A", "valid": True, "latency_s": 0.05}
    test_records = []
    for item in test:
        test_records.append(
            {**base_test, "model_id": "cloud-frontier", "item_id": item["id"], "raw": "a", "cost_usd": 0.02}
        )
        test_records.append(
            {**base_test, "model_id": "cloud-mid", "item_id": item["id"], "raw": "a", "cost_usd": 0.005}
        )
        test_records.append({**base_test, "model_id": "local-llm", "item_id": item["id"], "raw": "a", "cost_usd": 0.0})
    run.append_records(PREDICTIONS, test_records)

    route_cfg = RouteConfig(workload_mix={"classification": 1.0}, reference_monthly_volume=REFERENCE_VOLUME)
    plan = planner.build_plan(run, specs, route_cfg, gates=True)
    return run, specs, route_cfg, plan


def test_mixed_table_has_five_labelled_rows(tmp_path):
    run, specs, route_cfg, plan = _setup(tmp_path)
    rows = simulate.mixed_table(run, plan, specs, route_cfg, split="test")
    assert len(rows) == 5
    assert rows[0].label == "all-frontier"
    assert rows[1].label == "cheapest-single-cloud"
    assert "no gates" in rows[2].label
    assert "gates" in rows[3].label and "no constraints" in rows[3].label
    assert "gates" in rows[4].label and "route.yaml" in rows[4].label


def test_all_frontier_row_uses_the_frontier_models_measured_cost(tmp_path):
    run, specs, route_cfg, plan = _setup(tmp_path)
    rows = simulate.mixed_table(run, plan, specs, route_cfg, split="test")
    row = rows[0]
    assert math.isclose(row.usd_per_1k, 0.02 * 1000)
    assert math.isclose(row.per_task["classification"]["metric"], 1.0)
    assert row.per_task["classification"]["meets_bar"]


def test_cheapest_single_cloud_row_picks_the_cheaper_qualifying_model(tmp_path):
    run, specs, route_cfg, plan = _setup(tmp_path)
    rows = simulate.mixed_table(run, plan, specs, route_cfg, split="test")
    row = rows[1]
    # cloud-mid is also perfectly accurate and cheaper than the frontier model.
    assert math.isclose(row.usd_per_1k, 0.005 * 1000)


def test_router_row_prefers_the_cheap_local_model_and_matches_shared_machine_formula(tmp_path):
    run, specs, route_cfg, plan = _setup(tmp_path)
    tp = plan.tasks["classification"]
    assert tp.primary is not None and tp.primary.model_id == "local-llm"

    rows = simulate.mixed_table(run, plan, specs, route_cfg, split="test")
    gated_row = rows[4]
    assert gated_row.served_locally_pct == 100.0
    assert gated_row.escalation_pct == 0.0

    fixed = costmodel.fixed_usd_per_month(
        purchase_price_usd=HW["purchase_price_usd"],
        allocation=HW["allocation"],
        lifetime_years=HW["lifetime_years"],
        idle_watts=HW["power"]["idle_watts"],
        electricity_usd_per_kwh=HW["electricity_usd_per_kwh"],
        ops_usd_per_month=HW["ops_usd_per_month"],
    )
    energy_per_task = costmodel.energy_usd_per_task(
        incremental_watts=HW["power"]["incremental_watts"],
        sustained_tasks_per_hour=LOCAL_THROUGHPUT_TPH,
        electricity_usd_per_kwh=HW["electricity_usd_per_kwh"],
    )
    capacity = costmodel.capacity_per_month(LOCAL_THROUGHPUT_TPH, HW["busy_hours_per_day"])
    monthly_calls = REFERENCE_VOLUME  # all 4 test items served locally, scaled 1:1 to the task's monthly volume
    utilisation = monthly_calls / capacity
    machines = max(1, math.ceil(utilisation))
    expected_usd_per_1k = (machines * fixed + monthly_calls * energy_per_task) / REFERENCE_VOLUME * 1000

    assert gated_row.usd_per_1k is not None
    assert math.isclose(gated_row.usd_per_1k, expected_usd_per_1k, rel_tol=1e-9)
    assert gated_row.saving_vs_all_frontier_pct is not None and gated_row.saving_vs_all_frontier_pct > 0


def test_mixed_table_reports_none_cheapest_cloud_when_no_model_meets_every_bar(tmp_path):
    spec, calib, test = _spec(tmp_path)
    specs = {"classification": spec}
    run = RunDir(tmp_path / "run")
    run.write_json(
        "config.json",
        {
            "project": "p",
            "hardware": HW,
            "models": [{"id": "cloud-frontier", "kind": "cloud", "model": "acme/frontier-v1", "role": "frontier"}],
        },
    )
    base_calib = {"task": "classification", "split": "calib", "pass": "A", "valid": True, "latency_s": 0.05}
    records = [
        {**base_calib, "model_id": "cloud-frontier", "item_id": item["id"], "raw": "b", "cost_usd": 0.02}
        for item in calib
    ]
    run.append_records(PREDICTIONS, records)
    base_test = {"task": "classification", "split": "test", "pass": "A", "valid": True, "latency_s": 0.05}
    records = [
        {**base_test, "model_id": "cloud-frontier", "item_id": item["id"], "raw": "b", "cost_usd": 0.02}
        for item in test
    ]
    run.append_records(PREDICTIONS, records)

    route_cfg = RouteConfig(
        workload_mix={"classification": 1.0},
        reference_monthly_volume=REFERENCE_VOLUME,
        quality_bar={"default": {"absolute": 0.9}},
    )
    plan = planner.build_plan(run, specs, route_cfg, gates=False)
    rows = simulate.mixed_table(run, plan, specs, route_cfg, split="test")
    assert rows[1].usd_per_1k is None
    # A row without a model has no shares either: n/a, not 0.0%.
    assert rows[1].model_id is None
    assert rows[1].served_locally_pct is None and rows[1].escalation_pct is None


def _two_task_setup(tmp_path):
    """``classification`` is served locally; ``triage`` must stay local but the local model gets it wrong (503)."""
    specs = {}
    items: dict[str, tuple[list, list]] = {}
    for name in ("classification", "triage"):
        root = tmp_path / name
        root.mkdir()
        spec = TaskSpec(name=name, kind="classification", labels=["a", "b"])
        spec.root = root
        calib = [{"id": f"c{i}", "text": f"apple calib {i}", "label": "a"} for i in range(4)]
        test = [{"id": f"t{i}", "text": f"apple test {i}", "label": "a"} for i in range(4)]
        _write_jsonl(root / "calib.jsonl", calib)
        _write_jsonl(root / "test.jsonl", test)
        specs[name] = spec
        items[name] = (calib, test)
    run = RunDir(tmp_path / "run")
    run.write_json(
        "config.json",
        {
            "project": "p",
            "hardware": HW,
            "models": [
                {"id": "cloud-frontier", "kind": "cloud", "model": "acme/frontier-v1", "role": "frontier"},
                {"id": "local-llm", "kind": "local"},
            ],
        },
    )
    run.write_json(
        "latency.json",
        {"local-llm": {n: {"test": {"A": {"tasks_per_hour": LOCAL_THROUGHPUT_TPH}}} for n in specs}},
    )
    records = []
    for name, (calib, test) in items.items():
        for split, rows in (("calib", calib), ("test", test)):
            for item in rows:
                base = {"task": name, "split": split, "pass": "A", "valid": True, "item_id": item["id"]}
                records.append({**base, "model_id": "cloud-frontier", "raw": "a", "cost_usd": 0.02, "latency_s": 2.0})
                local_answer = "a" if name == "classification" else "b"
                records.append(
                    {**base, "model_id": "local-llm", "raw": local_answer, "cost_usd": 0.0, "latency_s": 0.1}
                )
    run.append_records(PREDICTIONS, records)
    route_cfg = RouteConfig(
        workload_mix={"classification": 0.5, "triage": 0.5},
        reference_monthly_volume=REFERENCE_VOLUME,
        quality_bar={"default": {"absolute": 0.9}},
        constraints={"data_must_stay_local": ["triage"]},
    )
    plan = planner.build_plan(run, specs, route_cfg, gates=True)
    return run, specs, route_cfg, plan


def test_unservable_task_is_excluded_from_cost_latency_and_shares(tmp_path):
    run, specs, route_cfg, plan = _two_task_setup(tmp_path)
    assert plan.tasks["triage"].status == planner.STATUS_UNSERVABLE
    rows = simulate.mixed_table(run, plan, specs, route_cfg, split="test")
    constrained = rows[4]

    assert constrained.per_task["triage"]["unservable"] is True
    assert constrained.per_task["triage"]["meets_bar"] is False
    assert constrained.per_task["classification"]["unservable"] is False
    # Only the served task counts: served locally 100% (not 50%), no escalation, latency of the local model only.
    assert constrained.served_locally_pct == 100.0
    assert constrained.escalation_pct == 0.0
    assert constrained.p50_s is not None and math.isclose(constrained.p50_s, 0.1)
    assert constrained.p95_s is not None and math.isclose(constrained.p95_s, 0.1)
    # USD per 1k is per served task: the classification task alone, not diluted by the refused task's volume.
    fixed = costmodel.fixed_usd_per_month(
        purchase_price_usd=HW["purchase_price_usd"],
        allocation=HW["allocation"],
        lifetime_years=HW["lifetime_years"],
        idle_watts=HW["power"]["idle_watts"],
        electricity_usd_per_kwh=HW["electricity_usd_per_kwh"],
        ops_usd_per_month=HW["ops_usd_per_month"],
    )
    energy = costmodel.energy_usd_per_task(
        incremental_watts=HW["power"]["incremental_watts"],
        sustained_tasks_per_hour=LOCAL_THROUGHPUT_TPH,
        electricity_usd_per_kwh=HW["electricity_usd_per_kwh"],
    )
    served_volume = REFERENCE_VOLUME * 0.5
    assert constrained.usd_per_1k is not None
    assert math.isclose(constrained.usd_per_1k, (fixed + served_volume * energy) / served_volume * 1000, rel_tol=1e-9)
    # The saving compares like with like: the frontier over the served task only ($0.02 a task).
    assert constrained.saving_vs_all_frontier_pct is not None
    expected = (20.0 - constrained.usd_per_1k) / 20.0 * 100
    assert math.isclose(constrained.saving_vs_all_frontier_pct, expected, rel_tol=1e-9)
