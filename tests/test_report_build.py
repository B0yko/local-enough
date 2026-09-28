"""Determinism, HTML self-containment and partial-run robustness of the generated report.

Builds a small but complete synthetic run directory (2 local models, 2 cloud models, 3 baselines,
all 5 bundled tasks, both splits, plus latency/soak/power/memory/judge/downloads/live-check files)
using :class:`RunDir` directly -- the same technique ``tests/test_candidates.py`` and
``tests/test_costing.py`` use -- rather than depending on a real bench run.
"""

from __future__ import annotations

import json
from pathlib import Path

import fakes
from local_enough.bench.rundir import PREDICTIONS, RunDir
from local_enough.config import Constraints, RouteConfig
from local_enough.report.build import build_report, readme_blocks
from local_enough.report.markdown import BLOCK_NAMES
from local_enough.tasks import registry
from local_enough.tasks.base import TaskSpec

HW = {
    "name": "test-box",
    "purchase_price_usd": 1500.0,
    "price_label": "configured",
    "lifetime_years": 3.0,
    "allocation": 1.0,
    "busy_hours_per_day": 8.0,
    "electricity_usd_per_kwh": 0.30,
    "ops_usd_per_month": 0.0,
    "power": {"mode": "measured", "incremental_watts": 22.0, "idle_watts": 6.0},
}

LOCAL_MODELS = ("loc-a", "loc-b")
CLOUD_MODELS = ("cloud-frontier", "cloud-small")
BASELINE_FOR_KIND = {
    "classification": "tfidf-baseline",
    "pii_redaction": "regex-baseline",
    "entity_matching": "rapidfuzz-baseline",
}


def route_cfg() -> RouteConfig:
    return RouteConfig(
        workload_mix={
            "classification": 0.30,
            "extraction": 0.25,
            "pii_redaction": 0.20,
            "entity_matching": 0.15,
            "summarisation": 0.10,
        },
        reference_monthly_volume=50_000,
        constraints=Constraints(data_must_stay_local=["pii_redaction"]),
    )


def _config_json() -> dict:
    return {
        "project": "synthetic",
        "budget_usd": 15.0,
        "budget_warn_usd": 12.0,
        "hardware": HW,
        "judge": {"candidates": ["judge-a"]},
        "models": [
            {"id": "loc-a", "kind": "local", "launch": {"repo": "org/model-a", "revision": "aaa111"}},
            {"id": "loc-b", "kind": "local", "launch": {"repo": "org/model-b", "revision": "bbb222"}},
            {"id": "cloud-frontier", "kind": "cloud", "model": "vendor/frontier", "role": "frontier"},
            {"id": "cloud-small", "kind": "cloud", "model": "vendor/small", "role": "small-closed"},
            {"id": "tfidf-baseline", "kind": "baseline", "task": "classification"},
            {"id": "regex-baseline", "kind": "baseline", "task": "pii_redaction"},
            {"id": "rapidfuzz-baseline", "kind": "baseline", "task": "entity_matching"},
        ],
    }


def _predictions(specs: dict[str, TaskSpec]) -> list[dict]:
    records = []
    for task_name, spec in sorted(specs.items()):
        content = fakes.default_valid_answer(spec)
        models = [*LOCAL_MODELS, *CLOUD_MODELS]
        baseline = BASELINE_FOR_KIND.get(spec.kind)
        if baseline:
            models.append(baseline)
        for split in ("calib", "test"):
            for item in registry.load_items(spec, split)[:3]:
                for model_id in models:
                    is_local = model_id in LOCAL_MODELS or model_id == baseline
                    records.append(
                        {
                            "model_id": model_id,
                            "task": task_name,
                            "split": split,
                            "pass": "A",
                            "item_id": item["id"],
                            "raw": content,
                            "latency_s": 0.01 if model_id == baseline else (0.4 if is_local else 0.7),
                            "cost_usd": 0.0 if is_local else 0.003,
                        }
                    )
    return records


def _judge_files(run: RunDir, specs: dict[str, TaskSpec]) -> None:
    holdout_items = [{"id": f"h{i}", "label_pass": i % 2 == 0, "predicted_pass": i % 2 == 0} for i in range(20)]
    run.write_json(
        "judge_calibration.json",
        {
            "selected": "judge-a",
            "target_balanced_accuracy": 0.90,
            "meets_target": True,
            "judge_calib_n": 40,
            "created_utc": "2026-09-20T09:30:00Z",
            "candidates": {
                "judge-a": {"calib": {"tpr": 0.93, "tnr": 0.91, "balanced_accuracy": 0.92, "kappa": 0.85, "n": 40}}
            },
            "holdout": {
                "tpr": 0.92,
                "tnr": 0.90,
                "balanced_accuracy": 0.91,
                "kappa": 0.82,
                "n": 20,
                "invalid_rate": 0.0,
                "per_item": holdout_items,
            },
        },
    )
    spec = specs["summarisation"]
    records = []
    for model_id in (*LOCAL_MODELS, *CLOUD_MODELS):
        for split in ("calib", "test"):
            for i, item in enumerate(registry.load_items(spec, split)[:3]):
                passed = (i % 2) == 0
                records.append(
                    {
                        "model_id": model_id,
                        "split": split,
                        "item_id": item["id"],
                        "judge_model": "judge-a",
                        "verdict": {
                            "facts": [{"index": 0, "present": True, "correct": passed}],
                            "unsupported_claims": [],
                        },
                        "passed": passed,
                        "fact_recall": 1.0 if passed else 0.5,
                        "within_limit": True,
                        "key_token": [{"index": 0, "check": passed, "judge": passed, "agree": True}],
                    }
                )
    run.append_records("judge_scores.jsonl.gz", records)


def write_full_synthetic_run(run_dir: Path) -> tuple[RunDir, dict[str, TaskSpec]]:
    """A small but complete run: every optional file present, so every report block renders real data."""
    run = RunDir(run_dir)
    specs = {s.name: s for s in registry.bundled_tasks()}

    run.write_json("config.json", _config_json())
    run.write_json(
        "env.json",
        {
            "machine_model": "Mac99,9",
            "chip": "Apple M99",
            "memory_gb": 24.0,
            "macos_version": "26.1",
            "macos_build": "25Z1",
            "mlx_version": "0.32.0",
            "mlx_lm_version": "0.31.0",
            "models": {"loc-a": "org/model-a@aaa111", "loc-b": "org/model-b@bbb222"},
            "hardware_price": {"usd": 1500.0, "label": "configured", "source_url": None, "date": "2026-01-01"},
            "start_utc": "2026-09-20T09:00:00Z",
            "end_utc": "2026-09-20T12:00:00Z",
        },
    )
    run.write_json("price_snapshot.json", {"taken_at_utc": "2026-09-20T08:00:00Z", "models": {}})
    run.write_json(
        "downloads.json",
        {
            "loc-a": {
                "repo": "org/model-a",
                "revision_sha": "aaa111",
                "bytes": 2_300_000_000,
                "bytes_new": 2_300_000_000,
                "licence": "Apache-2.0",
                "pulled_at_utc": "2026-09-19T00:00:00Z",
            },
            "loc-b": {
                "repo": "org/model-b",
                "revision_sha": "bbb222",
                "bytes": 900_000_000,
                "bytes_new": 900_000_000,
                "licence": "Apache-2.0",
                "pulled_at_utc": "2026-09-19T00:05:00Z",
            },
        },
    )
    run.append_records(PREDICTIONS, _predictions(specs))

    latency = {
        model_id: {
            task_name: {
                "test": {"A": {"p50_s": 0.4, "p95_s": 0.6, "tasks_per_hour": 1200.0}, "B": {"tasks_per_hour": 2100.0}}
            }
            for task_name in specs
        }
        for model_id in LOCAL_MODELS
    }
    run.write_json("latency.json", latency)
    run.write_json(
        "soak.json",
        {model_id: {"first5_tph": 2000.0, "last5_tph": 1700.0, "throttle_factor": 0.85} for model_id in LOCAL_MODELS},
    )
    run.write_json(
        "power.json",
        {model_id: {"mode": "measured", "incremental_watts": 20.0, "idle_watts": 5.0} for model_id in LOCAL_MODELS},
    )
    run.write_json("memory.json", {"combined": {"bytes": 6_000_000_000, "method": "footprint"}})
    run.write_json("route.json", json.loads(route_cfg().model_dump_json()))
    run.write_json(
        "live_check.json",
        {
            "n": 100,
            "seed": 7,
            "created_utc": "2026-09-20T12:30:00Z",
            "overhead_p50_ms": 12.5,
            "direct_p50_ms": 410.0,
            "router_p50_ms": 422.5,
            "decision_match_rate": 0.98,
            "mismatches": [],
            "local_only_cloud_calls": 0,
            "per_task": {},
        },
    )
    _judge_files(run, specs)
    return run, specs


def write_partial_synthetic_run(run_dir: Path) -> tuple[RunDir, dict[str, TaskSpec]]:
    """The bare minimum: config.json plus a handful of predictions, nothing else."""
    run = RunDir(run_dir)
    specs = {s.name: s for s in registry.bundled_tasks()}
    run.write_json("config.json", _config_json())
    run.append_records(PREDICTIONS, _predictions(specs))
    return run, specs


def test_report_is_byte_for_byte_deterministic(tmp_path: Path) -> None:
    run, specs = write_full_synthetic_run(tmp_path / "run")
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    paths1 = build_report(run, out1, specs=specs, route_cfg=route_cfg())
    paths2 = build_report(run, out2, specs=specs, route_cfg=route_cfg())

    rel1 = sorted(p.relative_to(out1) for p in paths1)
    rel2 = sorted(p.relative_to(out2) for p in paths2)
    assert rel1 == rel2
    assert rel1 == [
        Path("ADR-local-vs-cloud.md"),
        Path("img/classification.png"),
        Path("img/entity_matching.png"),
        Path("img/extraction.png"),
        Path("img/pii_redaction.png"),
        Path("img/summarisation.png"),
        Path("index.html"),
        Path("report.md"),
    ]
    for rel in rel1:
        assert (out1 / rel).read_bytes() == (out2 / rel).read_bytes(), f"{rel} differs between two builds"


def test_index_html_is_self_contained_with_inline_svg(tmp_path: Path) -> None:
    run, specs = write_full_synthetic_run(tmp_path / "run")
    out = tmp_path / "out"
    build_report(run, out, specs=specs, route_cfg=route_cfg())
    html_text = (out / "index.html").read_text(encoding="utf-8")

    assert "<svg" in html_text
    # No externally-loadable resource: no script/link/stylesheet tags and no http(s) src/href.
    lowered = html_text.lower()
    assert "<script" not in lowered
    assert "<link" not in lowered
    assert 'src="http' not in lowered
    assert 'href="http' not in lowered
    assert "@import" not in lowered
    # The two mandatory SVG namespace declarations are not resource loads; nothing else should
    # mention http(s) at all.
    residual = html_text.replace('xmlns:xlink="http://www.w3.org/1999/xlink"', "").replace(
        'xmlns="http://www.w3.org/2000/svg"', ""
    )
    assert "http://" not in residual
    assert "https://" not in residual


def test_partial_run_renders_without_crashing(tmp_path: Path) -> None:
    run, specs = write_partial_synthetic_run(tmp_path / "run")
    out = tmp_path / "out"
    written = build_report(run, out, specs=specs, route_cfg=route_cfg())
    assert written  # produced files rather than raising

    report_md = (out / "report.md").read_text(encoding="utf-8")
    assert "not measured" in report_md  # soak/power/memory absent
    assert "not judged" in report_md or "*(not judged" in report_md  # no judge_calibration.json

    blocks = readme_blocks(run, specs=specs, route_cfg=route_cfg())
    assert set(blocks) == set(BLOCK_NAMES)
    assert "router table unavailable" in blocks["router"]


def test_readme_blocks_match_build_report_content(tmp_path: Path) -> None:
    """readme_blocks() and build_report()'s report.md must never say different things."""
    run, specs = write_full_synthetic_run(tmp_path / "run")
    blocks = readme_blocks(run, specs=specs, route_cfg=route_cfg())
    out = tmp_path / "out"
    build_report(run, out, specs=specs, route_cfg=route_cfg())
    report_md = (out / "report.md").read_text(encoding="utf-8")
    assert blocks["spend"].strip() in report_md
    assert blocks["verdicts"].strip() in report_md
