"""Replay a plan's recorded predictions (no network) to score the router and cost the mixed workload.

``simulate`` walks each item through a task's fallback chain exactly as the server would: a recorded error,
an invalid parse, or (when ``gates`` is on) a failed gate escalates to the next candidate. The final answer is
scored with the task's own module, so the router's reported metric is computed the same way ``bench``/
``report`` compute it. ``mixed_table`` builds the five comparison rows of spec Evaluation item 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_enough import costing, costmodel
from local_enough.bench.rundir import RunDir
from local_enough.config import Config, RouteConfig
from local_enough.route import gates as gates_module
from local_enough.route.planner import STATUS_UNSERVABLE, Plan, TaskPlan
from local_enough.stats import p50_p95, weighted_percentile
from local_enough.tasks import registry
from local_enough.tasks.base import Item, ItemScore, Parsed, TaskKindModule, TaskSpec

GATE_PASSED = "passed"
GATE_FAILED = "failed"
GATE_FORMAT_ONLY = "format-only"
GATE_NA = "n/a"


@dataclass(frozen=True)
class ItemResult:
    item_id: str
    served_by: str | None
    escalated: bool
    gate: str
    latency_s: float
    chain: list[str]
    score: ItemScore
    calls_by_model: dict[str, int]
    cloud_cost_by_model: dict[str, float]


@dataclass(frozen=True)
class TaskSimResult:
    task: str
    n: int
    metric: float
    bar: float
    meets_bar: bool
    served_locally_rate: float
    escalation_rate: float
    p50_s: float
    p95_s: float
    calls_by_model: dict[str, int]
    cloud_cost_by_model: dict[str, float]
    latencies_s: list[float] = field(default_factory=list, repr=False)
    items: list[ItemResult] = field(default_factory=list, repr=False)
    """Per-item decisions (not serialised by ``to_dict``): used by ``route.replay`` to compare a live run
    against this replay."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "n": self.n,
            "metric": self.metric,
            "bar": self.bar,
            "meets_bar": self.meets_bar,
            "served_locally_rate": self.served_locally_rate,
            "escalation_rate": self.escalation_rate,
            "p50_s": self.p50_s,
            "p95_s": self.p95_s,
            "calls_by_model": dict(self.calls_by_model),
            "cloud_cost_by_model": dict(self.cloud_cost_by_model),
        }


@dataclass(frozen=True)
class SimResult:
    split: str
    tasks: dict[str, TaskSimResult]

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split, "tasks": {name: r.to_dict() for name, r in self.tasks.items()}}


def _as_rundir(run: RunDir | str | Path) -> RunDir:
    return run if isinstance(run, RunDir) else RunDir(run)


def _judge_verdicts(run_dir: RunDir) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(r["model_id"]), str(r["split"]), str(r["item_id"])): r["verdict"]
        for r in run_dir.iter_records("judge_scores.jsonl.gz")
    }


def _walk_chain(
    tp: TaskPlan,
    spec: TaskSpec,
    module: TaskKindModule,
    item: Item,
    predictions: dict[tuple[str, str], dict[str, Any]],
    gate_ctx: gates_module.GateContext | None,
    gates_enabled: bool,
    judge_verdicts: dict[tuple[str, str, str], dict[str, Any]],
    split: str,
) -> ItemResult:
    """Walk one item through ``tp``'s fallback chain and score whichever answer ends up served."""
    item_id = str(item["id"])
    chain = tp.chain_model_ids()
    total_latency = 0.0
    escalated = False
    attempted: list[str] = []
    calls: list[tuple[str, str]] = []  # (model_id, kind) for every attempt made

    served_by: str | None = None
    gate_label = GATE_NA
    served_parsed: Parsed | None = None
    served_extra: dict[str, Any] | None = None
    fallback_parsed: Parsed | None = None
    fallback_extra: dict[str, Any] | None = None

    for model_id in chain:
        kind = tp.candidate_kind(model_id) or "cloud"
        rec = predictions.get((model_id, item_id))
        attempted.append(model_id)
        if rec is None:
            escalated = True
            continue
        calls.append((model_id, kind))
        total_latency += float(rec.get("latency_s") or 0.0)
        if rec.get("error") or not rec.get("valid"):
            escalated = True
            continue

        parsed = module.parse(spec, str(rec.get("raw") or ""))
        if not parsed.ok:
            escalated = True
            continue

        verdict_key = (model_id, split, item_id)
        extra = {"judge": judge_verdicts[verdict_key]} if verdict_key in judge_verdicts else None
        if model_id == tp.best_quality_model_id:
            fallback_parsed, fallback_extra = parsed, extra

        if gates_enabled and gate_ctx is not None:
            gr = gates_module.gate(spec, item, parsed, gate_ctx, format_only=kind == "baseline")
        else:
            gr = gates_module.GateResult(True, None)

        if gr.passed:
            served_by = model_id
            gate_label = GATE_FORMAT_ONLY if (gates_enabled and kind == "baseline") else GATE_PASSED
            served_parsed, served_extra = parsed, extra
            break
        escalated = True

    if served_by is None:
        gate_label = GATE_FAILED
        # A task that isn't local-only serves the best-quality candidate's answer anyway when
        # every candidate fails the gate; a local-only task's chain exhausting instead means the router
        # would 503 (``local_only_unavailable``), so it must not be scored as if served -- mirrors server.py.
        if tp.local_only:
            served_parsed = Parsed(ok=False, error="local_only_unavailable")
            served_extra = None
        else:
            served_by = tp.best_quality_model_id
            served_parsed = fallback_parsed if fallback_parsed is not None else Parsed(ok=False, error="no_answer")
            served_extra = fallback_extra
    assert served_parsed is not None  # every branch above sets it

    score = module.score(spec, item, served_parsed, served_extra)
    calls_by_model: dict[str, int] = {}
    cloud_cost_by_model: dict[str, float] = {}
    for model_id in attempted:
        calls_by_model[model_id] = calls_by_model.get(model_id, 0) + 1
    for model_id, kind in calls:
        if kind == "cloud":
            rec = predictions.get((model_id, item_id)) or {}
            cloud_cost_by_model[model_id] = cloud_cost_by_model.get(model_id, 0.0) + float(rec.get("cost_usd") or 0.0)

    return ItemResult(
        item_id=item_id,
        served_by=served_by,
        escalated=escalated,
        gate=gate_label,
        latency_s=total_latency,
        chain=attempted,
        score=score,
        calls_by_model=calls_by_model,
        cloud_cost_by_model=cloud_cost_by_model,
    )


def _simulate_task(
    tp: TaskPlan,
    spec: TaskSpec,
    run_dir: RunDir,
    gate_ctx: gates_module.GateContext | None,
    gates_enabled: bool,
    judge_verdicts: dict[tuple[str, str, str], dict[str, Any]],
    split: str,
) -> TaskSimResult:
    module = registry.get_kind(spec.kind)
    items = registry.load_items(spec, split)

    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in run_dir.predictions("A"):
        if rec.get("task") == spec.name and rec.get("split") == split:
            predictions[(str(rec["model_id"]), str(rec["item_id"]))] = rec

    scores: list[ItemScore] = []
    latencies: list[float] = []
    served_locally = 0
    escalations = 0
    calls_by_model: dict[str, int] = {}
    cloud_cost_by_model: dict[str, float] = {}
    item_results: list[ItemResult] = []

    if tp.status == STATUS_UNSERVABLE:
        for item in items:
            score = module.score(spec, item, Parsed(ok=False, error="unservable"), None)
            scores.append(score)
            latencies.append(0.0)
            item_results.append(
                ItemResult(
                    str(item["id"]), None, False, GATE_NA, 0.0, [], score, calls_by_model={}, cloud_cost_by_model={}
                )
            )
    else:
        for item in items:
            result = _walk_chain(tp, spec, module, item, predictions, gate_ctx, gates_enabled, judge_verdicts, split)
            item_results.append(result)
            scores.append(result.score)
            latencies.append(result.latency_s)
            if result.escalated:
                escalations += 1
            kind = tp.candidate_kind(result.served_by) if result.served_by else None
            if kind in ("local", "baseline"):
                served_locally += 1
            for model_id, n in result.calls_by_model.items():
                calls_by_model[model_id] = calls_by_model.get(model_id, 0) + n
            for model_id, cost in result.cloud_cost_by_model.items():
                cloud_cost_by_model[model_id] = cloud_cost_by_model.get(model_id, 0.0) + cost

    metrics = module.aggregate(spec, scores)
    metric = float(metrics.get("primary", math.nan))
    n = len(items)
    p50, p95 = p50_p95(latencies)
    bar_value = tp.bar_value
    meets_bar = not math.isnan(metric) and not math.isnan(bar_value) and metric >= bar_value

    return TaskSimResult(
        task=tp.task,
        n=n,
        metric=metric,
        bar=bar_value,
        meets_bar=meets_bar,
        served_locally_rate=(served_locally / n) if n else math.nan,
        escalation_rate=(escalations / n) if n else math.nan,
        p50_s=p50,
        p95_s=p95,
        calls_by_model=calls_by_model,
        cloud_cost_by_model=cloud_cost_by_model,
        latencies_s=latencies,
        items=item_results,
    )


def simulate(
    run: RunDir | str | Path,
    plan: Plan,
    specs: dict[str, TaskSpec],
    route_cfg: RouteConfig,
    cfg: Config | None = None,
    *,
    split: str = "test",
    gates: bool = True,
) -> SimResult:
    """Replay ``plan`` over ``run``'s recorded ``split`` predictions; deterministic, no network calls."""
    del cfg
    run_dir = _as_rundir(run)
    gate_ctx = gates_module.build_gate_context(run_dir, specs) if gates else None
    judge_verdicts = _judge_verdicts(run_dir)

    tasks: dict[str, TaskSimResult] = {}
    for task_name, tp in plan.tasks.items():
        spec = specs.get(task_name)
        if spec is None:
            continue
        tasks[task_name] = _simulate_task(tp, spec, run_dir, gate_ctx, gates, judge_verdicts, split)
    return SimResult(split=split, tasks=tasks)


# -- mixed-workload cost and the report's five comparison rows -----------------------------------------------


@dataclass(frozen=True)
class MixedRow:
    label: str
    usd_per_1k: float | None
    per_task: dict[str, dict[str, Any]]
    served_locally_pct: float | None
    escalation_pct: float | None
    p50_s: float | None
    p95_s: float | None
    saving_vs_all_frontier_pct: float | None
    saving_vs_cheapest_cloud_pct: float | None
    model_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "usd_per_1k": self.usd_per_1k,
            "per_task": self.per_task,
            "served_locally_pct": self.served_locally_pct,
            "escalation_pct": self.escalation_pct,
            "p50_s": self.p50_s,
            "p95_s": self.p95_s,
            "saving_vs_all_frontier_pct": self.saving_vs_all_frontier_pct,
            "saving_vs_cheapest_cloud_pct": self.saving_vs_cheapest_cloud_pct,
        }


def _bar_value(route_cfg: RouteConfig, task_name: str, best: float) -> float:
    bar_cfg = route_cfg.bar_for(task_name)
    if bar_cfg.relative_to_best is not None:
        return bar_cfg.relative_to_best * best
    return float(bar_cfg.absolute or 0.0)


def _weighted_avg(values: dict[str, float], weights: dict[str, float]) -> float | None:
    total_w = math.fsum(weights.get(t, 0.0) for t in values)
    if total_w <= 0:
        return None
    return math.fsum(values[t] * weights.get(t, 0.0) for t in values) / total_w


def _mixed_cost(
    sim: SimResult,
    run_dir: RunDir,
    route_cfg: RouteConfig,
    local_costs: dict[tuple[str, str], costing.LocalCost],
    unservable: frozenset[str] = frozenset(),
) -> float | None:
    """USD per 1,000 mixed tasks under the shared-machine scenario (docs/cost-model.md).

    Tasks in ``unservable`` are refused (HTTP 503), so they carry no cost and their volume is left out of the
    per-1,000 denominator: the figure is per task actually served."""
    hw = costing.hardware_from_run(run_dir)
    fixed_usd_per_month = 0.0
    if hw is not None:
        fixed_usd_per_month = costmodel.fixed_usd_per_month(
            purchase_price_usd=hw.purchase_price_usd,
            allocation=hw.allocation,
            lifetime_years=hw.lifetime_years,
            idle_watts=hw.power.idle_watts,
            electricity_usd_per_kwh=hw.electricity_usd_per_kwh,
            ops_usd_per_month=hw.ops_usd_per_month,
        )

    utilisation = 0.0
    energy_total = 0.0
    cloud_total = 0.0
    any_local_llm_call = False
    models = costing.run_models(run_dir)

    served_volume = 0.0
    for task_name, task_result in sim.tasks.items():
        n = task_result.n
        if n <= 0 or task_name in unservable:
            continue
        volume = route_cfg.monthly_volume(task_name)
        served_volume += volume
        scale = volume / n
        for model_id, call_count in task_result.calls_by_model.items():
            monthly_calls = call_count * scale
            kind = str((models.get(model_id) or {}).get("kind", ""))
            if kind == "local":
                any_local_llm_call = True
                lc = local_costs.get((model_id, task_name))
                if lc is not None and lc.capacity_per_month > 0:
                    utilisation += monthly_calls / lc.capacity_per_month
                if lc is not None and math.isfinite(lc.energy_usd_per_task):
                    energy_total += monthly_calls * lc.energy_usd_per_task
        for model_id, cost_sum in task_result.cloud_cost_by_model.items():
            call_count = task_result.calls_by_model.get(model_id, 0)
            if call_count <= 0:
                continue
            mean_cost = cost_sum / call_count
            cloud_total += mean_cost * call_count * scale

    machines = (max(1, math.ceil(utilisation)) if utilisation > 0 else 1) if any_local_llm_call else 0
    monthly_total = machines * fixed_usd_per_month + energy_total + cloud_total
    if served_volume <= 0:
        return None
    return monthly_total / served_volume * 1000


def _fixed_model_row(
    label: str,
    model_id: str | None,
    run_dir: RunDir,
    specs: dict[str, TaskSpec],
    route_cfg: RouteConfig,
    split: str,
    only_tasks: frozenset[str] | None = None,
) -> MixedRow:
    """A non-router row: every task in ``workload_mix`` (or just ``only_tasks``) served entirely by one fixed
    cloud model.

    The bar is always the calib-decided one (:func:`_bar_value` over the calib candidate table), matching how
    every other verdict in the report is decided, even though the metric and cost shown are measured on
    ``split``.
    """
    if model_id is None:
        return MixedRow(label, None, {}, None, None, None, None, None, None)

    from local_enough import candidates as candidates_module

    calib_table = candidates_module.candidate_table(run_dir, specs, route_cfg, split="calib")
    table = candidates_module.candidate_table(run_dir, specs, route_cfg, split=split)
    per_task: dict[str, dict[str, Any]] = {}
    usd_parts: list[float] = []
    latencies: list[float] = []
    weights: list[float] = []
    item_latency: dict[str, list[float]] = {}
    for rec in run_dir.predictions("A"):
        if rec.get("model_id") == model_id and rec.get("split") == split:
            item_latency.setdefault(str(rec["task"]), []).append(float(rec.get("latency_s") or 0.0))
    mix = {t: w for t, w in sorted(route_cfg.workload_mix.items()) if only_tasks is None or t in only_tasks}
    total_weight = math.fsum(mix.values())
    norm = total_weight if only_tasks is not None and total_weight > 0 else 1.0
    for task_name, weight in mix.items():
        best = max((c.primary for c in calib_table.get(task_name, []) if not math.isnan(c.primary)), default=math.nan)
        bar_value = _bar_value(route_cfg, task_name, best)
        cand = next((c for c in table.get(task_name, []) if c.model_id == model_id), None)
        if cand is None:
            per_task[task_name] = {"metric": math.nan, "bar": bar_value, "meets_bar": False}
            continue
        meets = not math.isnan(cand.primary) and not math.isnan(bar_value) and cand.primary >= bar_value
        per_task[task_name] = {"metric": cand.primary, "bar": bar_value, "meets_bar": meets}
        usd_parts.append(weight * cand.usd_per_task)
        task_latencies = item_latency.get(task_name, [])
        item_weight = weight / len(task_latencies) if task_latencies else 0.0
        latencies.extend(task_latencies)
        weights.extend([item_weight] * len(task_latencies))

    p50 = weighted_percentile(latencies, weights, 50.0) if latencies else None
    p95 = weighted_percentile(latencies, weights, 95.0) if latencies else None
    return MixedRow(label, math.fsum(usd_parts) / norm * 1000, per_task, 0.0, 0.0, p50, p95, None, None, model_id)


def _cheapest_single_cloud(
    run_dir: RunDir, specs: dict[str, TaskSpec], route_cfg: RouteConfig, split: str = "calib"
) -> str | None:
    """The cloud model id that meets every task's bar on calib, cheapest by weighted USD/1k on test."""
    from local_enough import candidates as candidates_module

    calib_table = candidates_module.candidate_table(run_dir, specs, route_cfg, split=split)
    test_table = candidates_module.candidate_table(run_dir, specs, route_cfg, split="test")
    models = costing.run_models(run_dir)
    cloud_ids = [mid for mid, m in models.items() if m.get("kind") == "cloud"]

    best_id: str | None = None
    best_cost = math.inf
    for model_id in cloud_ids:
        qualifies = True
        cost = 0.0
        for task_name, weight in route_cfg.workload_mix.items():
            cands = calib_table.get(task_name, [])
            best = max((c.primary for c in cands if not math.isnan(c.primary)), default=math.nan)
            bar_value = _bar_value(route_cfg, task_name, best)
            cand = next((c for c in cands if c.model_id == model_id), None)
            if cand is None or math.isnan(cand.primary) or math.isnan(bar_value) or cand.primary < bar_value:
                qualifies = False
                break
            test_cand = next((c for c in test_table.get(task_name, []) if c.model_id == model_id), None)
            cost += weight * (test_cand.usd_per_task if test_cand else math.inf)
        if qualifies and cost < best_cost:
            best_cost, best_id = cost, model_id
    return best_id


def _role_model_id(run_dir: RunDir, role: str) -> str | None:
    models = costing.run_models(run_dir)
    for model_id, m in models.items():
        if m.get("kind") == "cloud" and m.get("role") == role:
            return model_id
    return None


def mixed_table(
    run: RunDir | str | Path,
    plan: Plan,
    specs: dict[str, TaskSpec],
    route_cfg: RouteConfig,
    cfg: Config | None = None,
    *,
    split: str = "test",
) -> list[MixedRow]:
    """The five rows of spec Evaluation item 6, computed over the full ``split`` (default ``test``)."""
    del cfg
    run_dir = _as_rundir(run)
    local_costs = costing.local_costs(run_dir, route_cfg)

    frontier_id = _role_model_id(run_dir, "frontier")
    cheapest_id = _cheapest_single_cloud(run_dir, specs, route_cfg)

    row_all_frontier = _fixed_model_row("all-frontier", frontier_id, run_dir, specs, route_cfg, split)
    row_cheapest_cloud = _fixed_model_row("cheapest-single-cloud", cheapest_id, run_dir, specs, route_cfg, split)

    from local_enough.route.planner import build_plan

    plan_no_gates = build_plan(run_dir, specs, route_cfg, constraints=[], gates=False)
    plan_gates_no_constraints = build_plan(run_dir, specs, route_cfg, constraints=[], gates=True)

    def _router_row(label: str, p: Plan, gates_enabled: bool) -> MixedRow:
        sim = simulate(run_dir, p, specs, route_cfg, split=split, gates=gates_enabled)
        refused = frozenset(n for n in sim.tasks if p.tasks[n].status == STATUS_UNSERVABLE)
        per_task = {
            name: {
                "metric": r.metric,
                "bar": r.bar,
                "meets_bar": r.meets_bar and name not in refused,
                "unservable": name in refused,
            }
            for name, r in sim.tasks.items()
        }
        # Refused (503) items are neither served nor escalated and have no latency: leave them out of every share.
        live = {n: r for n, r in sim.tasks.items() if n not in refused}
        served = _weighted_avg({n: r.served_locally_rate for n, r in live.items()}, route_cfg.workload_mix)
        esc = _weighted_avg({n: r.escalation_rate for n, r in live.items()}, route_cfg.workload_mix)
        latencies: list[float] = []
        weights: list[float] = []
        for name, r in live.items():
            w = route_cfg.workload_mix.get(name, 0.0) / r.n if r.n else 0.0
            latencies.extend(r.latencies_s)
            weights.extend([w] * len(r.latencies_s))
        p50 = weighted_percentile(latencies, weights, 50.0) if latencies else None
        p95 = weighted_percentile(latencies, weights, 95.0) if latencies else None
        usd = _mixed_cost(sim, run_dir, route_cfg, local_costs, refused)
        return MixedRow(
            label,
            usd,
            per_task,
            served * 100 if served is not None else None,
            esc * 100 if esc is not None else None,
            p50,
            p95,
            None,
            None,
        )

    row_no_gates = _router_row("router (no gates, no constraints)", plan_no_gates, False)
    row_gates_no_constraints = _router_row("router (gates, no constraints)", plan_gates_no_constraints, True)
    row_gates_constraints = _router_row("router (gates, route.yaml constraints)", plan, True)

    subset_rows: dict[frozenset[str], tuple[MixedRow, MixedRow]] = {}

    def _reference_rows(row: MixedRow) -> tuple[MixedRow, MixedRow]:
        """The two fixed-model rows over the tasks ``row`` actually serves (all of them unless some are refused)."""
        served = frozenset(t for t, v in row.per_task.items() if not v.get("unservable"))
        if len(served) == len(row.per_task):
            return row_all_frontier, row_cheapest_cloud
        if served not in subset_rows:
            subset_rows[served] = (
                _fixed_model_row("all-frontier", frontier_id, run_dir, specs, route_cfg, split, served),
                _fixed_model_row("cheapest-single-cloud", cheapest_id, run_dir, specs, route_cfg, split, served),
            )
        return subset_rows[served]

    def _savings(row: MixedRow) -> MixedRow:
        vs_frontier = None
        vs_cheapest = None
        frontier_ref, cheapest_ref = _reference_rows(row)
        if row.usd_per_1k is not None:
            if frontier_ref.usd_per_1k:
                vs_frontier = (frontier_ref.usd_per_1k - row.usd_per_1k) / frontier_ref.usd_per_1k * 100
            if cheapest_ref.usd_per_1k:
                vs_cheapest = (cheapest_ref.usd_per_1k - row.usd_per_1k) / cheapest_ref.usd_per_1k * 100
        return MixedRow(
            row.label,
            row.usd_per_1k,
            row.per_task,
            row.served_locally_pct,
            row.escalation_pct,
            row.p50_s,
            row.p95_s,
            vs_frontier,
            vs_cheapest,
            row.model_id,
        )

    return [
        row_all_frontier,
        row_cheapest_cloud,
        _savings(row_no_gates),
        _savings(row_gates_no_constraints),
        _savings(row_gates_constraints),
    ]
