"""simulate() chain-walk mechanics: escalation on error/invalid/gate-fail, the all-fail fallback, and
per-task aggregation (metric, bar, served-locally and escalation rates, latency percentiles)."""

from __future__ import annotations

import json

from local_enough.bench.rundir import PREDICTIONS, RunDir
from local_enough.config import Constraints, RouteConfig
from local_enough.route import planner, simulate
from local_enough.stats import p50_p95
from local_enough.tasks.base import TaskSpec

LATENCY_S = 0.1


def _write_jsonl(path, items) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")


def _classification_spec(tmp_path) -> TaskSpec:
    root = tmp_path / "cls"
    root.mkdir()
    spec = TaskSpec(name="classification", kind="classification", labels=["a", "b"], train="train.jsonl")
    spec.root = root

    train = [{"id": f"tr{i}", "text": f"apple report {i}", "label": "a"} for i in range(15)]
    train += [{"id": f"tr{i}", "text": f"banana update {i}", "label": "b"} for i in range(15, 30)]
    _write_jsonl(root / "train.jsonl", train)

    calib = [{"id": f"c{i}", "text": f"apple calib {i}", "label": "a"} for i in range(10)]
    calib += [{"id": f"c{i}", "text": f"banana calib {i}", "label": "b"} for i in range(10, 20)]
    _write_jsonl(root / "calib.jsonl", calib)

    test = [
        {"id": "t1", "text": "apple report one", "label": "a"},
        {"id": "t2", "text": "banana update two", "label": "b"},
        {"id": "t3", "text": "apple special three", "label": "a"},
        {"id": "t4", "text": "apple special four", "label": "a"},
    ]
    _write_jsonl(root / "test.jsonl", test)
    return spec, calib


def _setup(tmp_path):
    spec, calib = _classification_spec(tmp_path)
    specs = {"classification": spec}
    run = RunDir(tmp_path / "run")
    run.write_json(
        "config.json",
        {
            "project": "p",
            "models": [
                {"id": "cloud-p", "kind": "cloud", "model": "vendorp/model-p"},
                {"id": "cloud-f", "kind": "cloud", "model": "vendorf/model-f"},
            ],
        },
    )

    base_calib = {"task": "classification", "split": "calib", "pass": "A", "latency_s": LATENCY_S, "valid": True}
    calib_records = []
    for i, item in enumerate(calib):
        # primary ("cloud-p") is wrong on exactly one calib item (19/20 = 0.95, right at the 0.95 bar);
        # fallback ("cloud-f") is correct on all 20 (1.0, the best score and the calib-only tie-break winner).
        primary_label = "b" if (i == 0 and item["label"] == "a") else item["label"]
        calib_records.append(
            {**base_calib, "model_id": "cloud-p", "item_id": item["id"], "raw": primary_label, "cost_usd": 0.001}
        )
        calib_records.append(
            {**base_calib, "model_id": "cloud-f", "item_id": item["id"], "raw": item["label"], "cost_usd": 0.002}
        )
    run.append_records(PREDICTIONS, calib_records)

    base_test = {"task": "classification", "split": "test", "pass": "A", "latency_s": LATENCY_S, "valid": True}
    test_records = [
        # t1: primary answers correctly and agrees with the TF-IDF baseline -> served directly, no escalation.
        {**base_test, "model_id": "cloud-p", "item_id": "t1", "raw": "a", "cost_usd": 0.001},
        # t2: primary's call recorded an error -> escalates to the fallback, which answers correctly.
        {
            **base_test,
            "model_id": "cloud-p",
            "item_id": "t2",
            "raw": "",
            "cost_usd": 0.001,
            "error": "http_500",
            "valid": False,
        },
        {**base_test, "model_id": "cloud-f", "item_id": "t2", "raw": "b", "cost_usd": 0.002},
        # t3: both primary and fallback disagree with the baseline ("apple" -> "a") -> gate fails on both;
        # the chain is exhausted, so the fallback's (the best-quality candidate's) own wrong answer is served.
        {**base_test, "model_id": "cloud-p", "item_id": "t3", "raw": "b", "cost_usd": 0.001},
        {**base_test, "model_id": "cloud-f", "item_id": "t3", "raw": "b", "cost_usd": 0.002},
        # t4: primary disagrees with the baseline (gate fails, escalates); the fallback agrees and is served.
        {**base_test, "model_id": "cloud-p", "item_id": "t4", "raw": "b", "cost_usd": 0.001},
        {**base_test, "model_id": "cloud-f", "item_id": "t4", "raw": "a", "cost_usd": 0.002},
    ]
    run.append_records(PREDICTIONS, test_records)

    route_cfg = RouteConfig(workload_mix={"classification": 1.0}, reference_monthly_volume=1000)
    plan = planner.build_plan(run, specs, route_cfg, gates=True)
    return run, specs, route_cfg, plan


def test_simulate_walks_the_chain_and_scores_the_served_answer(tmp_path):
    run, specs, route_cfg, plan = _setup(tmp_path)
    tp = plan.tasks["classification"]
    assert tp.primary is not None and tp.primary.model_id == "cloud-p"
    assert tp.best_quality_model_id == "cloud-f"

    result = simulate.simulate(run, plan, specs, route_cfg, split="test", gates=True)
    task_result = result.tasks["classification"]
    assert task_result.n == 4
    # t1, t2, t4 are answered correctly; t3 is served (as a last resort) with the wrong answer.
    assert abs(task_result.metric - 0.75) < 1e-9
    assert task_result.served_locally_rate == 0.0  # both candidates are cloud models
    assert abs(task_result.escalation_rate - 0.75) < 1e-9  # only t1 avoids escalation

    expected_p50, expected_p95 = p50_p95([LATENCY_S, LATENCY_S * 2, LATENCY_S * 2, LATENCY_S * 2])
    assert abs(task_result.p50_s - expected_p50) < 1e-9
    assert abs(task_result.p95_s - expected_p95) < 1e-9

    assert task_result.calls_by_model == {"cloud-p": 4, "cloud-f": 3}
    assert abs(task_result.cloud_cost_by_model["cloud-p"] - 4 * 0.001) < 1e-9
    assert abs(task_result.cloud_cost_by_model["cloud-f"] - 3 * 0.002) < 1e-9


def test_simulate_reports_meets_bar_against_the_calib_derived_bar(tmp_path):
    run, specs, route_cfg, plan = _setup(tmp_path)
    result = simulate.simulate(run, plan, specs, route_cfg, split="test", gates=True)
    task_result = result.tasks["classification"]
    # bar = 0.95 * 1.0 (fallback's calib score); test metric 0.75 falls short of it.
    assert abs(task_result.bar - 0.95) < 1e-9
    assert not task_result.meets_bar


def test_simulate_without_gates_only_escalates_on_error_or_invalid_parse(tmp_path):
    run, specs, route_cfg, plan = _setup(tmp_path)
    result = simulate.simulate(run, plan, specs, route_cfg, split="test", gates=False)
    task_result = result.tasks["classification"]
    # t3 and t4 no longer escalate (the gate is off), only t2 (a recorded error) does.
    assert abs(task_result.escalation_rate - 0.25) < 1e-9
    # t1, t2 correct; t3, t4 now served directly by the (disagreeing) primary and scored wrong.
    assert abs(task_result.metric - 0.5) < 1e-9


def test_simulate_unservable_task_is_scored_as_wrong_with_no_calls(tmp_path):
    run, specs, route_cfg, _plan = _setup(tmp_path)
    # Both candidates are cloud models; declaring the task local-only leaves no survivor at all.
    route_cfg = route_cfg.model_copy(update={"constraints": Constraints(data_must_stay_local=["classification"])})
    from local_enough.route.planner import build_plan

    strict_plan = build_plan(run, specs, route_cfg, gates=True)
    assert strict_plan.tasks["classification"].status == "unservable"

    result = simulate.simulate(run, strict_plan, specs, route_cfg, split="test", gates=True)
    task_result = result.tasks["classification"]
    assert task_result.metric == 0.0
    assert task_result.calls_by_model == {}
    assert task_result.served_locally_rate == 0.0
