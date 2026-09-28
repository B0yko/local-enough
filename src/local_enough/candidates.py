"""One row per (model, task, split): quality with CI, validity, latency and USD per task.

The route planner and the report both rank candidates from this table, so the quality bar and the verdicts are
decided on exactly the same numbers. Summarisation's primary score is the bias-corrected judged pass rate when the
run is judged, and NaN ("not judged") otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from local_enough import costing, evaluate
from local_enough.bench.rundir import RunDir
from local_enough.config import RouteConfig
from local_enough.judge.stats import judge_summary
from local_enough.tasks.base import TaskSpec


@dataclass(frozen=True)
class Candidate:
    model_id: str
    task: str
    split: str
    kind: str
    provider: str
    role: str | None
    n: int
    primary: float
    primary_ci: tuple[float, float]
    metrics: dict[str, float]
    ci: dict[str, tuple[float, float]]
    invalid_output_rate: float
    p50_s: float
    p95_s: float
    usd_per_task: float
    judged: bool = False
    raw_pass_rate: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        """Local LLMs and in-process baselines count as local for data_must_stay_local."""
        return self.kind in ("local", "baseline")


def _latency(latency: dict[str, Any], model_id: str, task: str, split: str) -> tuple[float, float]:
    node = latency.get(model_id, {}).get(task, {}).get(split, {}).get("A") or {}
    return float(node.get("p50_s", math.nan)), float(node.get("p95_s", math.nan))


_RUN_FILES = (
    "predictions.jsonl.gz",
    "latency.json",
    "config.json",
    "soak.json",
    "power.json",
    "judge_scores.jsonl.gz",
    "judge_calibration.json",
)


def run_fingerprint(run: RunDir) -> tuple[Any, ...]:
    """Resolved path plus size and mtime of every input file: a cache key that changes when the run does."""
    stats = []
    for name in _RUN_FILES:
        path = run.file(name)
        stats.append((name, path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else (name, None, None))
    return (str(run.path.resolve()), *stats)


_TABLE_CACHE: dict[tuple[Any, ...], dict[str, list[Candidate]]] = {}


def candidate_table(
    run: RunDir, specs: dict[str, TaskSpec], route_cfg: RouteConfig, split: str
) -> dict[str, list[Candidate]]:
    """Candidates per task for ``split`` (Pass A), sorted by model id for stable output.

    Scoring and bootstrapping are the expensive part of a report, and the planner, the simulation and the report
    all ask for the same tables, so results are cached per run fingerprint, split, route config and task set.
    """
    key = (run_fingerprint(run), split, route_cfg.model_dump_json(), tuple(sorted(specs)))
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = _candidate_table(run, specs, route_cfg, split)
    return {task: list(cands) for task, cands in _TABLE_CACHE[key].items()}


def _candidate_table(
    run: RunDir, specs: dict[str, TaskSpec], route_cfg: RouteConfig, split: str
) -> dict[str, list[Candidate]]:
    models = costing.run_models(run)
    latency = run.read_json("latency.json", {}) or {}
    costs = costing.usd_per_task(run, route_cfg, split=split)
    judged = judge_summary(run)
    table: dict[str, list[Candidate]] = {}
    for (model_id, task), scores in sorted(evaluate.score_run(run, specs, split).items()):
        spec = specs[task]
        model = models.get(model_id, {"id": model_id, "kind": "cloud"})
        summary = evaluate.summarise(spec, scores, ci_metrics=("primary",))
        metrics: dict[str, float] = summary["metrics"]
        ci: dict[str, tuple[float, float]] = {k: (float(v[0]), float(v[1])) for k, v in summary["ci"].items()}
        primary, primary_ci = float(metrics["primary"]), ci.get("primary", (math.nan, math.nan))
        is_judged, raw = False, None
        if spec.kind == "summarisation":
            j = judged.get((model_id, split))
            if j is not None:
                is_judged, raw = True, float(j["raw_pass_rate"])
                primary = float(j["corrected_pass_rate"])
                primary_ci = (float(j["ci"][0]), float(j["ci"][1]))
            else:
                primary, primary_ci = math.nan, (math.nan, math.nan)
        p50, p95 = _latency(latency, model_id, task, split)
        table.setdefault(task, []).append(
            Candidate(
                model_id=model_id,
                task=task,
                split=split,
                kind=str(model.get("kind")),
                provider=costing.provider_of(model),
                role=model.get("role"),
                n=int(summary["n"]),
                primary=primary,
                primary_ci=primary_ci,
                metrics=metrics,
                ci=ci,
                invalid_output_rate=float(metrics.get("invalid_output_rate", math.nan)),
                p50_s=p50,
                p95_s=p95,
                usd_per_task=costs.get((model_id, task), math.nan),
                judged=is_judged,
                raw_pass_rate=raw,
            )
        )
    return table
