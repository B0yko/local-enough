import math

from local_enough.bench.rundir import PREDICTIONS, RunDir
from local_enough.candidates import candidate_table
from local_enough.config import RouteConfig
from local_enough.tasks import registry


def test_candidate_table_scores_costs_and_marks_unjudged_summaries(tmp_path):
    specs = {s.name: s for s in registry.bundled_tasks()}
    run = RunDir(tmp_path / "run")
    run.write_json(
        "config.json",
        {
            "project": "p",
            "models": [
                {"id": "good", "kind": "cloud", "model": "vendor/good"},
                {"id": "tfidf", "kind": "baseline", "task": "classification"},
            ],
        },
    )
    items = registry.load_items(specs["classification"], "calib")[:4]
    summaries = registry.load_items(specs["summarisation"], "calib")[:2]
    base = {"split": "calib", "pass": "A", "latency_s": 1.0}
    records = [
        {**base, "model_id": "good", "task": "classification", "item_id": i["id"], "raw": i["label"], "cost_usd": 0.001}
        for i in items
    ]
    records += [
        {**base, "model_id": "tfidf", "task": "classification", "item_id": i["id"], "raw": "nonsense", "cost_usd": 0.0}
        for i in items
    ]
    records += [
        {**base, "model_id": "good", "task": "summarisation", "item_id": s["id"], "raw": "Short.", "cost_usd": 0.002}
        for s in summaries
    ]
    run.append_records(PREDICTIONS, records)
    route = RouteConfig(workload_mix={"classification": 0.5, "summarisation": 0.5}, reference_monthly_volume=1000)

    table = candidate_table(run, specs, route, "calib")
    by_id = {c.model_id: c for c in table["classification"]}
    assert by_id["good"].primary == 1.0 and math.isclose(by_id["good"].usd_per_task, 0.001)
    assert by_id["tfidf"].primary == 0.0 and by_id["tfidf"].invalid_output_rate == 1.0
    assert by_id["tfidf"].is_local and by_id["tfidf"].usd_per_task == 0.0
    summ = table["summarisation"][0]
    assert not summ.judged and math.isnan(summ.primary)
