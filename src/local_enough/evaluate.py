"""Deterministic scoring: re-parse recorded ``raw`` output and score it against gold.

Every function here reads a run directory and a set of task specs; nothing is fetched over the
network and nothing is randomised beyond the fixed-seed bootstrap, so a run directory always
reproduces the same ``metrics.json`` (see :mod:`local_enough.bench.runner`, which calls this module
to write that file, and ``local-enough report``, which recomputes it the same way).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from local_enough.bench.rundir import RunDir
from local_enough.stats import bootstrap_ci
from local_enough.tasks import registry
from local_enough.tasks.base import Item, ItemScore, TaskSpec

JUDGE_SCORES = "judge_scores.jsonl.gz"


def _as_rundir(run: str | Path | RunDir) -> RunDir:
    return run if isinstance(run, RunDir) else RunDir(run)


def task_specs_for_run(run: str | Path | RunDir, extra_tasks: dict[str, TaskSpec] | None = None) -> dict[str, TaskSpec]:
    """Every bundled task by name, plus a run's custom tasks (from ``tasks.json``) or ``extra_tasks``.

    A custom task is loaded from its ``task_yaml_path`` recorded in ``tasks.json``, relative to the
    current working directory (the same directory ``local-enough bench`` was run from, matching how
    ``--tasks all,path/to/task.yaml`` itself resolves), when that file is still present;
    ``extra_tasks`` lets a caller (the CLI, or a test) supply a custom :class:`TaskSpec` directly
    instead, and always wins over a recorded path.
    """
    run_dir = _as_rundir(run)
    specs: dict[str, TaskSpec] = {spec.name: spec for spec in registry.bundled_tasks()}
    tasks_json = run_dir.read_json("tasks.json", {}) or {}
    for name, info in tasks_json.items():
        if not isinstance(info, dict) or info.get("source") != "custom":
            continue
        rel_path = info.get("task_yaml_path")
        if not rel_path:
            continue
        candidate = Path.cwd() / rel_path
        if candidate.exists():
            specs[name] = registry.load_task(candidate)
    if extra_tasks:
        specs.update(extra_tasks)
    return specs


def _judge_verdicts(run_dir: RunDir) -> dict[tuple[str, str, str], dict[str, Any]]:
    verdicts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in run_dir.iter_records(JUDGE_SCORES):
        key = (str(record["model_id"]), str(record["split"]), str(record["item_id"]))
        verdicts[key] = record["verdict"]
    return verdicts


def score_run(
    run: str | Path | RunDir,
    specs: dict[str, TaskSpec],
    split: str,
    *,
    pass_: str = "A",
) -> dict[tuple[str, str], list[ItemScore]]:
    """Re-parse ``raw`` predictions with the task module and score them against gold.

    Returns one list of :class:`ItemScore` per ``(model_id, task)`` pair found in the run for
    ``split``/``pass_``. Summarisation gets ``extra={"judge": verdict}`` from ``judge_scores.jsonl.gz``
    when a verdict for that ``(model_id, split, item_id)`` was recorded, else it scores unjudged.
    """
    run_dir = _as_rundir(run)
    judge_verdicts = _judge_verdicts(run_dir)
    items_by_task: dict[str, dict[str, Item]] = {}
    results: dict[tuple[str, str], list[ItemScore]] = {}

    for record in run_dir.predictions(pass_):
        if record.get("split") != split:
            continue
        task_name = str(record["task"])
        spec = specs.get(task_name)
        if spec is None:
            continue
        item_id = str(record["item_id"])

        if task_name not in items_by_task:
            items_by_task[task_name] = {str(item["id"]): item for item in registry.load_items(spec, split)}
        item = items_by_task[task_name].get(item_id)
        if item is None:
            continue

        module = registry.get_kind(spec.kind)
        parsed = module.parse(spec, str(record.get("raw") or ""))
        extra: dict[str, Any] | None = None
        if spec.kind == "summarisation":
            key = (str(record["model_id"]), split, item_id)
            if key in judge_verdicts:
                extra = {"judge": judge_verdicts[key]}
        score = module.score(spec, item, parsed, extra)
        results.setdefault((str(record["model_id"]), task_name), []).append(score)

    return results


def summarise(spec: TaskSpec, scores: list[ItemScore]) -> dict[str, Any]:
    """``{n, metrics, ci}`` with a 95% bootstrap CI (1000 resamples, seed 7) for every metric."""
    module = registry.get_kind(spec.kind)
    metrics = module.aggregate(spec, scores)
    ci: dict[str, tuple[float, float]] = {}
    for name in metrics:

        def _statistic(resample: Sequence[ItemScore], _name: str = name) -> float:
            return module.aggregate(spec, list(resample))[_name]

        ci[name] = bootstrap_ci(scores, _statistic, n=1000, seed=7)
    return {"n": len(scores), "metrics": metrics, "ci": ci}


def latency_summary(run: str | Path | RunDir) -> dict[str, Any]:
    """The run's recorded ``latency.json``, unchanged (already per model/task/split/pass)."""
    result: dict[str, Any] = _as_rundir(run).read_json("latency.json", {}) or {}
    return result
