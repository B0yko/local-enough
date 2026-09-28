"""Planner rules: bar, latency cap, local-only, fallback provider diversity, baseline format-only,
allow_below_bar_local."""

from __future__ import annotations

import json

from local_enough.bench.rundir import PREDICTIONS, RunDir
from local_enough.config import RouteConfig
from local_enough.route import planner
from local_enough.tasks import registry

SPECS = {s.name: s for s in registry.bundled_tasks()}
CLASSIFICATION_ITEMS = registry.load_items(SPECS["classification"], "calib")[:40]
PII_ITEM = registry.load_items(SPECS["pii_redaction"], "calib")[0]


def _other_label(item, labels: list[str]) -> str:
    gold = item["label"]
    return next(lbl for lbl in labels if lbl != gold)


def _classification_run(tmp_path):
    labels = SPECS["classification"].labels or []
    run = RunDir(tmp_path / "run")
    run.write_json(
        "config.json",
        {
            "project": "p",
            "models": [
                {"id": "cloud-quality", "kind": "cloud", "model": "best/quality-v1"},
                {"id": "cloud-cheap", "kind": "cloud", "model": "acme/cheap-v1"},
                {"id": "cloud-other", "kind": "cloud", "model": "other/mid-v1"},
                {"id": "tfidf", "kind": "baseline", "task": "classification"},
            ],
        },
    )
    base = {"task": "classification", "split": "calib", "pass": "A"}
    records = []
    for i, item in enumerate(CLASSIFICATION_ITEMS):
        records.append(
            {**base, "model_id": "cloud-quality", "item_id": item["id"], "raw": item["label"], "cost_usd": 0.05}
        )
        wrong_cheap = i >= 39  # 39/40 correct
        wrong_other = i >= 38  # 38/40 correct
        records.append(
            {
                **base,
                "model_id": "cloud-cheap",
                "item_id": item["id"],
                "raw": _other_label(item, labels) if wrong_cheap else item["label"],
                "cost_usd": 0.001,
            }
        )
        records.append(
            {
                **base,
                "model_id": "cloud-other",
                "item_id": item["id"],
                "raw": _other_label(item, labels) if wrong_other else item["label"],
                "cost_usd": 0.002,
            }
        )
        wrong_baseline = i >= 10  # 10/40 correct: well below bar
        records.append(
            {
                **base,
                "model_id": "tfidf",
                "item_id": item["id"],
                "raw": _other_label(item, labels) if wrong_baseline else item["label"],
                "cost_usd": 0.0,
            }
        )
    run.append_records(PREDICTIONS, records)
    return run


ROUTE = RouteConfig(workload_mix={"classification": 1.0}, reference_monthly_volume=1000)


def test_build_plan_bar_and_fallback_chain(tmp_path):
    run = _classification_run(tmp_path)
    plan = planner.build_plan(run, SPECS, ROUTE, gates=False)
    tp = plan.tasks["classification"]

    assert tp.status == "served"
    assert tp.bar_kind == "relative_to_best" and abs(tp.bar_value - 0.95) < 1e-9
    # tfidf (10/40 = 0.25) is well below the 0.95 bar.
    assert any(d.model_id == "tfidf" and d.reason == "below_bar" for d in tp.dropped)
    # cheapest survivor becomes primary.
    assert tp.primary is not None and tp.primary.model_id == "cloud-cheap"
    # first fallback: next survivor from a different provider (cloud-other, cheaper than cloud-quality).
    assert tp.fallbacks[0].model_id == "cloud-other"
    # third member: the best-quality survivor (cloud-quality, strictly higher score than the primary).
    assert tp.fallbacks[1].model_id == "cloud-quality"
    assert tp.best_quality_model_id == "cloud-quality"
    assert not tp.format_only_gate


def test_build_plan_marks_baseline_primary_as_format_only(tmp_path):
    run = RunDir(tmp_path / "run")
    labels = SPECS["classification"].labels or []
    items = CLASSIFICATION_ITEMS[:10]
    run.write_json(
        "config.json",
        {
            "project": "p",
            "models": [
                {"id": "cloud-x", "kind": "cloud", "model": "vendor/x"},
                {"id": "tfidf", "kind": "baseline", "task": "classification"},
            ],
        },
    )
    base = {"task": "classification", "split": "calib", "pass": "A"}
    records = []
    for item in items:
        records.append({**base, "model_id": "cloud-x", "item_id": item["id"], "raw": item["label"], "cost_usd": 0.01})
        records.append({**base, "model_id": "tfidf", "item_id": item["id"], "raw": item["label"], "cost_usd": 0.0})
    run.append_records(PREDICTIONS, records)

    plan = planner.build_plan(run, SPECS, ROUTE, gates=False)
    tp = plan.tasks["classification"]
    assert tp.primary is not None and tp.primary.model_id == "tfidf"
    assert tp.format_only_gate
    assert tp.gate_label == "format-only"
    _ = labels  # unused except for readability of intent


def test_build_plan_drops_candidates_above_latency_cap(tmp_path):
    run = RunDir(tmp_path / "run")
    items = CLASSIFICATION_ITEMS[:10]
    run.write_json(
        "config.json",
        {
            "project": "p",
            "models": [
                {"id": "fast", "kind": "cloud", "model": "vendor/fast"},
                {"id": "slow", "kind": "cloud", "model": "vendor2/slow"},
            ],
        },
    )
    run.write_json(
        "latency.json",
        {
            "fast": {"classification": {"calib": {"A": {"p50_s": 0.2, "p95_s": 0.5}}}},
            "slow": {"classification": {"calib": {"A": {"p50_s": 5.0, "p95_s": 12.0}}}},
        },
    )
    base = {"task": "classification", "split": "calib", "pass": "A"}
    records = []
    for item in items:
        records.append({**base, "model_id": "fast", "item_id": item["id"], "raw": item["label"], "cost_usd": 0.01})
        records.append({**base, "model_id": "slow", "item_id": item["id"], "raw": item["label"], "cost_usd": 0.001})
    run.append_records(PREDICTIONS, records)

    route = RouteConfig(
        workload_mix={"classification": 1.0},
        reference_monthly_volume=1000,
        latency_p95_max_s={"default": None, "classification": 5.0},
    )
    plan = planner.build_plan(run, SPECS, route, gates=False)
    tp = plan.tasks["classification"]
    assert tp.primary is not None and tp.primary.model_id == "fast"
    assert any(d.model_id == "slow" and d.reason == "latency_p95_exceeds_cap" for d in tp.dropped)


def _pii_run(tmp_path, *, cloud_score_perfect: bool, baseline_score_perfect: bool):
    run = RunDir(tmp_path / "run")
    run.write_json(
        "config.json",
        {
            "project": "p",
            "models": [
                {"id": "cloud-pii", "kind": "cloud", "model": "vendor/pii"},
                {"id": "regex", "kind": "baseline", "task": "pii_redaction"},
            ],
        },
    )
    correct_answer = json.dumps(
        [{"type": s["type"], "text": PII_ITEM["text"][s["start"] : s["end"]]} for s in PII_ITEM["spans"]]
    )
    base = {"task": "pii_redaction", "split": "calib", "pass": "A"}
    records = [
        {
            **base,
            "model_id": "cloud-pii",
            "item_id": PII_ITEM["id"],
            "raw": correct_answer if cloud_score_perfect else "[]",
            "cost_usd": 0.0001,
        },
        {
            **base,
            "model_id": "regex",
            "item_id": PII_ITEM["id"],
            "raw": correct_answer if baseline_score_perfect else "[]",
            "cost_usd": 0.0,
        },
    ]
    run.append_records(PREDICTIONS, records)
    return run


PII_ROUTE = RouteConfig(
    workload_mix={"pii_redaction": 1.0},
    reference_monthly_volume=1000,
    constraints={"data_must_stay_local": ["pii_redaction"]},
)


def test_build_plan_excludes_cloud_candidates_for_local_only_tasks(tmp_path):
    run = _pii_run(tmp_path, cloud_score_perfect=True, baseline_score_perfect=True)
    plan = planner.build_plan(run, SPECS, PII_ROUTE, gates=False)
    tp = plan.tasks["pii_redaction"]
    assert tp.local_only
    assert any(d.model_id == "cloud-pii" and d.reason == "cloud_excluded_local_only" for d in tp.dropped)
    assert tp.primary is not None and tp.primary.model_id == "regex"
    assert tp.format_only_gate  # the surviving primary is itself the baseline the gate compares against


def test_build_plan_marks_local_only_task_unservable_below_bar(tmp_path):
    run = _pii_run(tmp_path, cloud_score_perfect=True, baseline_score_perfect=False)
    plan = planner.build_plan(run, SPECS, PII_ROUTE, gates=False)
    tp = plan.tasks["pii_redaction"]
    assert tp.status == "unservable"
    assert tp.primary is None


def test_build_plan_allow_below_bar_local_serves_best_local_candidate(tmp_path):
    run = _pii_run(tmp_path, cloud_score_perfect=True, baseline_score_perfect=False)
    route = PII_ROUTE.model_copy(update={"allow_below_bar_local": True})
    plan = planner.build_plan(run, SPECS, route, gates=False)
    tp = plan.tasks["pii_redaction"]
    assert tp.status == "served"
    assert tp.below_bar
    assert tp.gate_label == "below-bar"
    assert tp.primary is not None and tp.primary.model_id == "regex"


def test_print_plan_renders_a_readable_table(tmp_path):
    run = _classification_run(tmp_path)
    plan = planner.build_plan(run, SPECS, ROUTE, gates=False)
    text = planner.print_plan(plan)
    lines = text.splitlines()
    assert lines[0].split()[:2] == ["TASK", "STATUS"]
    assert any("classification" in line and "served" in line for line in lines)
    assert "cloud-cheap" in text
