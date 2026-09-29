"""Determinism, HTML self-containment and partial-run robustness of the generated report.

Builds a small but complete synthetic run directory (2 local models, 2 cloud models, 3 baselines,
all 5 bundled tasks, both splits, plus latency/soak/power/memory/judge/downloads/live-check files)
using :class:`RunDir` directly -- the same technique ``tests/test_candidates.py`` and
``tests/test_costing.py`` use -- rather than depending on a real bench run.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import fakes
from local_enough.bench.rundir import PREDICTIONS, RunDir
from local_enough.config import Constraints, RouteConfig
from local_enough.report.build import build_report, readme_blocks
from local_enough.report.markdown import BLOCK_NAMES, render_router, render_sensitivity
from local_enough.report.tables import build_report_data
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
    gib = 2**30
    run.write_json(
        "memory.json",
        {
            "loc-a": {"peak_bytes": 4_000_000_000, "method": "footprint", "disk_bytes": 2_300_000_000},
            "loc-b": {"peak_bytes": 2_000_000_000, "method": "footprint", "disk_bytes": 900_000_000},
            "combined": {
                "method": "footprint (phys_footprint, includes Metal allocations)",
                "models_bytes": {"loc-a": 3 * gib, "loc-b": 1 * gib},
                "router_bytes": gib // 2,
                "ours_bytes": 4 * gib + gib // 2,
                "system_used_before_bytes": 10 * gib,
                "system_used_bytes": 15 * gib,
            },
        },
    )
    run.write_json(
        "load_samples.json",
        {
            "windows": [
                {"name": "A:loc-a", "contaminated": False, "samples": []},
                {"name": "A:loc-b", "contaminated": True, "samples": []},
                {"name": "B:loc-a", "contaminated": False, "samples": []},
            ]
        },
    )
    ledger = [
        {"command": "bench", "actual_usd": 1.5},
        {"command": "bench", "actual_usd": 0.5},
        {"command": "judge calibrate", "actual_usd": 0.25},
        {"command": "judge score", "actual_usd": 0.125},
        {"command": "route replay", "actual_usd": 0.0625},
    ]
    run.file("cost_ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger), encoding="utf-8")
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
        Path("img/classification-dark.png"),
        Path("img/classification.png"),
        Path("img/entity_matching-dark.png"),
        Path("img/entity_matching.png"),
        Path("img/extraction-dark.png"),
        Path("img/extraction.png"),
        Path("img/pii_redaction-dark.png"),
        Path("img/pii_redaction.png"),
        Path("img/summarisation-dark.png"),
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
    assert "Scenario: shared machine" in blocks["router"]


def test_readme_blocks_match_build_report_content(tmp_path: Path) -> None:
    """readme_blocks() and build_report()'s report.md must never say different things."""
    run, specs = write_full_synthetic_run(tmp_path / "run")
    blocks = readme_blocks(run, specs=specs, route_cfg=route_cfg())
    out = tmp_path / "out"
    build_report(run, out, specs=specs, route_cfg=route_cfg())
    report_md = (out / "report.md").read_text(encoding="utf-8")
    assert blocks["spend"].strip() in report_md
    assert blocks["verdicts"].strip() in report_md


def _blocks(tmp_path: Path) -> dict[str, str]:
    run, specs = write_full_synthetic_run(tmp_path / "run")
    return readme_blocks(run, specs=specs, route_cfg=route_cfg())


def test_local_performance_shows_peak_memory_per_model_and_the_combined_footprint(tmp_path: Path) -> None:
    block = _blocks(tmp_path)["local_perf"]
    assert "| peak memory | 4.00 GB (footprint) |" in block
    assert "| peak memory | 2.00 GB (footprint) |" in block
    assert "not measured" not in block.split("| peak memory")[1].split("\n")[0]
    assert "Both local models + router: 4.50 GiB (footprint; loc-a 3.00 GiB, loc-b 1.00 GiB, router 0.50 GiB)" in block
    assert "system memory in use at the time: 15.00 GiB of 24 GiB, 10.00 GiB before loading them" in block
    assert "(includes other workloads)" in block


def test_local_cost_cell_shows_the_fully_loaded_cost_with_the_energy_only_cost_beside_it(tmp_path: Path) -> None:
    block = _blocks(tmp_path)["results"]
    local_rows = [line for line in block.splitlines() if line.startswith("| loc-a |")]
    assert local_rows
    assert all("(energy $" in line for line in local_rows)
    cloud_rows = [line for line in block.splitlines() if line.startswith("| cloud-small |")]
    assert cloud_rows and all("energy" not in line for line in cloud_rows)


def test_spend_shows_the_budget_and_splits_the_ledger_by_command(tmp_path: Path) -> None:
    block = _blocks(tmp_path)["spend"]
    assert "| total API spend | $2.44 |" in block
    for row in (
        "| spend: bench | $2.00 |",
        "| spend: judge calibrate | $0.25 |",
        "| spend: judge score | $0.12 |",
        "| spend: route replay | $0.06 |",
        "| budget cap | $15.00 |",
        "| budget warning level | $12.00 |",
    ):
        assert row in block
    assert block.index("spend: bench") < block.index("spend: judge calibrate") < block.index("spend: route replay")


def test_spend_says_when_the_run_recorded_no_budget(tmp_path: Path) -> None:
    run, specs = write_full_synthetic_run(tmp_path / "run")
    cfg = json.loads(run.file("config.json").read_text(encoding="utf-8"))
    del cfg["budget_usd"], cfg["budget_warn_usd"]
    run.write_json("config.json", cfg)
    block = readme_blocks(run, specs=specs, route_cfg=route_cfg())["spend"]
    assert "| budget cap | not recorded in this run |" in block


def test_setup_shows_the_machine_power_and_measurement_windows(tmp_path: Path) -> None:
    run, specs = write_full_synthetic_run(tmp_path / "run")
    env = json.loads(run.file("env.json").read_text(encoding="utf-8"))
    env.update({"machine_model": "Mac16,9", "chip": "Apple M4 Max", "memory_gb": 128.0})
    run.write_json("env.json", env)
    cfg = json.loads(run.file("config.json").read_text(encoding="utf-8"))
    cfg["hardware"]["power"] = {"mode": "configured", "incremental_watts": 139.0, "idle_watts": 6.0}
    run.write_json("config.json", cfg)
    run.write_json(
        "power.json",
        {m: {"mode": "unavailable", "reason": "AppleSmartBattery is missing fields"} for m in LOCAL_MODELS},
    )
    setup = readme_blocks(run, specs=specs, route_cfg=route_cfg())["setup"]
    assert "| machine | Mac Studio (Mac16,9), Apple M4 Max, 128 GB |" in setup
    assert "128.00" not in setup
    assert (
        "| power | configured: 6 W idle, 139 W incremental, from Apple's published figures; "
        "no battery telemetry on a desktop |" in setup
    )
    assert "| measurement windows | 3 load-sample windows, 1 flagged contaminated |" in setup


def test_setup_falls_back_to_the_identifier_and_reports_missing_files(tmp_path: Path) -> None:
    run, specs = write_partial_synthetic_run(tmp_path / "run")
    setup = readme_blocks(run, specs=specs, route_cfg=route_cfg())["setup"]
    assert "| measurement windows | not measured |" in setup
    assert "| machine | n/a |" in setup


def test_sensitivity_says_why_it_is_not_applicable_when_no_cloud_model_meets_the_bar(tmp_path: Path) -> None:
    run, specs = write_full_synthetic_run(tmp_path / "run")
    data = build_report_data(run, specs=specs, route_cfg=route_cfg())
    empty = dataclasses.replace(
        data,
        break_even={
            **data.break_even,
            "sensitivity": [],
            "headline_task": "classification",
            "sensitivity_note": (
                "not applicable: no cloud model meets the classification bar, so there is no break-even volume to vary"
            ),
        },
    )
    assert render_sensitivity(empty) == (
        "*(sensitivity grid not applicable: no cloud model meets the classification bar, "
        "so there is no break-even volume to vary)*\n"
    )


def test_router_table_marks_refused_tasks_and_keeps_rows_without_a_model_at_n_a(tmp_path: Path) -> None:
    run, specs = write_full_synthetic_run(tmp_path / "run")
    data = build_report_data(run, specs=specs, route_cfg=route_cfg())
    per_task = {
        "classification": {"metric": 0.9, "bar": 0.8, "meets_bar": True, "unservable": False},
        "pii_redaction": {"metric": 0.0, "bar": 0.95, "meets_bar": False, "unservable": True},
    }
    rows = [
        {
            "label": "router (gates, route.yaml constraints)",
            "usd_per_1k": 1.0,
            "per_task": per_task,
            "served_locally_pct": 100.0,
            "escalation_pct": 0.0,
            "p50_s": 0.5,
            "p95_s": 0.9,
            "saving_vs_all_frontier_pct": 50.0,
            "saving_vs_cheapest_cloud_pct": None,
        },
        {
            "label": "cheapest-single-cloud",
            "usd_per_1k": None,
            "per_task": {},
            "served_locally_pct": None,
            "escalation_pct": None,
            "p50_s": None,
            "p95_s": None,
            "saving_vs_all_frontier_pct": None,
            "saving_vs_cheapest_cloud_pct": None,
        },
    ]
    text = render_router(dataclasses.replace(data, router=dataclasses.replace(data.router, mixed_rows=rows)))
    assert "Router with gates + data_must_stay_local (1/2 tasks served)" in text
    assert "unservable (503)" in text
    assert (
        "| Cheapest single cloud model meeting every bar: none | n/a | n/a | n/a | n/a | n/a / n/a | n/a | n/a |"
        in text
    )
    assert "0.0%" not in text.split("Per-task")[0].split("Cheapest single cloud")[1]


def test_headline_saving_of_a_real_build_is_a_sane_percentage(tmp_path: Path) -> None:
    run, specs = write_full_synthetic_run(tmp_path / "run")
    data = build_report_data(run, specs=specs, route_cfg=route_cfg())
    for word in data.headline.split():
        if word.endswith("%") and word[:-1].replace(".", "").replace("+", "").isdigit():
            assert abs(float(word[:-1])) <= 100
