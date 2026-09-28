"""Build a routing plan from a run's ``calib`` results and ``route.yaml`` (spec item 15).

Candidates come from :func:`local_enough.candidates.candidate_table` (the same table the report uses), so
the quality bar and the router's decisions are always read off identical numbers. Everything a gate needs
that must be tuned once on ``calib`` -- the extraction grounding ratio, the entity-matching uncertainty band
-- is computed here via :func:`local_enough.route.gates.build_gate_context` and stored in the plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_enough import candidates as candidates_module
from local_enough.bench.rundir import RunDir
from local_enough.candidates import Candidate
from local_enough.config import Config, RouteConfig
from local_enough.route import gates as gates_module
from local_enough.tasks.base import TaskSpec

STATUS_SERVED = "served"
STATUS_UNSERVABLE = "unservable"

GATE_PASSED = "passed"
GATE_FAILED = "failed"
GATE_BELOW_BAR = "below-bar"
GATE_FORMAT_ONLY = "format-only"
GATE_DISABLED = "disabled"


@dataclass(frozen=True)
class CandidateRef:
    """A candidate as it appears in the plan: enough to identify it and rank it, no CI or per-metric detail."""

    model_id: str
    kind: str
    provider: str
    role: str | None
    score: float
    usd_per_task: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "kind": self.kind,
            "provider": self.provider,
            "role": self.role,
            "score": self.score,
            "usd_per_task": self.usd_per_task,
        }


@dataclass(frozen=True)
class DroppedCandidate:
    model_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "reason": self.reason}


@dataclass(frozen=True)
class TaskPlan:
    """The routing decision for one task: primary, fallback chain, gate configuration and bar."""

    task: str
    status: str
    quality: str  # "primary" or "proxy (not judged)"
    bar_kind: str  # "relative_to_best" or "absolute"
    bar_value: float
    best_score: float
    local_only: bool
    latency_p95_max_s: float | None
    primary: CandidateRef | None
    fallbacks: list[CandidateRef]
    survivors: list[CandidateRef]
    dropped: list[DroppedCandidate]
    format_only_gate: bool
    below_bar: bool
    best_quality_model_id: str | None
    extraction_grounding_ratio: float | None
    entity_matching_band: tuple[float, float] | None
    gate_label: str

    def chain_model_ids(self) -> list[str]:
        """Primary then fallbacks, in the order simulate/server should try them."""
        if self.primary is None:
            return []
        seen = [self.primary.model_id]
        seen.extend(f.model_id for f in self.fallbacks if f.model_id not in seen)
        return seen

    def candidate_kind(self, model_id: str) -> str | None:
        for ref in [self.primary, *self.fallbacks, *self.survivors]:
            if ref is not None and ref.model_id == model_id:
                return ref.kind
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "status": self.status,
            "quality": self.quality,
            "bar_kind": self.bar_kind,
            "bar_value": self.bar_value,
            "best_score": self.best_score,
            "local_only": self.local_only,
            "latency_p95_max_s": self.latency_p95_max_s,
            "primary": self.primary.to_dict() if self.primary else None,
            "fallbacks": [f.to_dict() for f in self.fallbacks],
            "survivors": [s.to_dict() for s in self.survivors],
            "dropped": [d.to_dict() for d in self.dropped],
            "format_only_gate": self.format_only_gate,
            "below_bar": self.below_bar,
            "best_quality_model_id": self.best_quality_model_id,
            "extraction_grounding_ratio": self.extraction_grounding_ratio,
            "entity_matching_band": list(self.entity_matching_band) if self.entity_matching_band else None,
            "gate_label": self.gate_label,
        }


@dataclass(frozen=True)
class Plan:
    run: str
    gates_enabled: bool
    local_only_tasks: list[str]
    tasks: dict[str, TaskPlan] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "gates_enabled": self.gates_enabled,
            "local_only_tasks": list(self.local_only_tasks),
            "tasks": {name: tp.to_dict() for name, tp in self.tasks.items()},
        }


def _run_label(run: RunDir | str | Path) -> str:
    if isinstance(run, RunDir):
        return str(run.path)
    return str(run)


def _score_for(spec: TaskSpec | None, candidate: Candidate, quality: str) -> float:
    if spec is not None and spec.kind == "summarisation" and quality != "primary":
        within = candidate.metrics.get("within_word_limit", math.nan)
        invalid = candidate.invalid_output_rate
        if math.isnan(within) or math.isnan(invalid):
            return math.nan
        return within * (1 - invalid)
    return candidate.primary


def _to_ref(candidate: Candidate, score: float) -> CandidateRef:
    return CandidateRef(
        model_id=candidate.model_id,
        kind=candidate.kind,
        provider=candidate.provider,
        role=candidate.role,
        score=score,
        usd_per_task=candidate.usd_per_task,
    )


def _plan_task(
    task_name: str,
    spec: TaskSpec | None,
    cands: list[Candidate],
    route_cfg: RouteConfig,
    local_only_tasks: set[str],
    gate_ctx: gates_module.GateContext | None,
) -> TaskPlan:
    is_local_only = task_name in local_only_tasks
    quality = "primary"
    if spec is not None and spec.kind == "summarisation" and cands and not any(c.judged for c in cands):
        quality = "proxy (not judged)"

    scored = [(_score_for(spec, c, quality), c) for c in cands]
    finite_scores = [s for s, _ in scored if not math.isnan(s)]
    best_score = max(finite_scores) if finite_scores else math.nan

    bar_cfg = route_cfg.bar_for(task_name)
    if bar_cfg.relative_to_best is not None:
        bar_kind, bar_value = "relative_to_best", bar_cfg.relative_to_best * best_score
    else:
        bar_kind, bar_value = "absolute", float(bar_cfg.absolute or 0.0)
    latency_cap = route_cfg.latency_cap_for(task_name)

    survivors: list[tuple[float, Candidate]] = []
    dropped: list[DroppedCandidate] = []
    for score, c in scored:
        if is_local_only and not c.is_local:
            dropped.append(DroppedCandidate(c.model_id, "cloud_excluded_local_only"))
            continue
        if math.isnan(score) or (not math.isnan(bar_value) and score < bar_value):
            dropped.append(DroppedCandidate(c.model_id, "below_bar"))
            continue
        if latency_cap is not None and not math.isnan(c.p95_s) and c.p95_s > latency_cap:
            dropped.append(DroppedCandidate(c.model_id, "latency_p95_exceeds_cap"))
            continue
        survivors.append((score, c))

    extraction_ratio = None
    em_band = None
    if gate_ctx is not None:
        extraction_ratio = gate_ctx.thresholds.extraction_grounding_ratio.get(task_name)
        em_band = gate_ctx.thresholds.entity_matching_band.get(task_name)

    def _build(
        *,
        status: str,
        primary: CandidateRef | None,
        fallbacks: list[CandidateRef],
        survivors_out: list[CandidateRef],
        format_only_gate: bool,
        below_bar: bool,
        best_quality_model_id: str | None,
        gate_label: str,
    ) -> TaskPlan:
        return TaskPlan(
            task=task_name,
            status=status,
            quality=quality,
            bar_kind=bar_kind,
            bar_value=bar_value,
            best_score=best_score,
            local_only=is_local_only,
            latency_p95_max_s=latency_cap,
            primary=primary,
            fallbacks=fallbacks,
            survivors=survivors_out,
            dropped=dropped,
            format_only_gate=format_only_gate,
            below_bar=below_bar,
            best_quality_model_id=best_quality_model_id,
            extraction_grounding_ratio=extraction_ratio,
            entity_matching_band=em_band,
            gate_label=gate_label,
        )

    if not survivors:
        if is_local_only and route_cfg.allow_below_bar_local:
            local_scored = [(s, c) for s, c in scored if c.is_local and not math.isnan(s)]
            if local_scored:
                score, best_local = max(local_scored, key=lambda sc: sc[0])
                ref = _to_ref(best_local, score)
                return _build(
                    status=STATUS_SERVED,
                    primary=ref,
                    fallbacks=[],
                    survivors_out=[ref],
                    format_only_gate=best_local.kind == "baseline",
                    below_bar=True,
                    best_quality_model_id=ref.model_id,
                    gate_label=GATE_BELOW_BAR,
                )
        return _build(
            status=STATUS_UNSERVABLE,
            primary=None,
            fallbacks=[],
            survivors_out=[],
            format_only_gate=False,
            below_bar=False,
            best_quality_model_id=None,
            gate_label=GATE_DISABLED,
        )

    survivors.sort(key=lambda sc: sc[1].usd_per_task if not math.isnan(sc[1].usd_per_task) else math.inf)
    primary_score, primary_c = survivors[0]
    chain_refs = [_to_ref(primary_c, primary_score)]
    chosen_ids = {primary_c.model_id}

    for score, c in survivors[1:]:
        if c.provider != primary_c.provider:
            chain_refs.append(_to_ref(c, score))
            chosen_ids.add(c.model_id)
            break

    best_quality_score, best_quality_c = max(survivors, key=lambda sc: sc[0])
    if best_quality_c.model_id not in chosen_ids:
        chain_refs.append(_to_ref(best_quality_c, best_quality_score))
        chosen_ids.add(best_quality_c.model_id)

    format_only = primary_c.kind == "baseline"
    return _build(
        status=STATUS_SERVED,
        primary=chain_refs[0],
        fallbacks=chain_refs[1:],
        survivors_out=[_to_ref(c, s) for s, c in survivors],
        format_only_gate=format_only,
        below_bar=False,
        best_quality_model_id=best_quality_c.model_id,
        gate_label=GATE_FORMAT_ONLY if format_only else GATE_PASSED,
    )


def build_plan(
    run: RunDir | str | Path,
    specs: dict[str, TaskSpec],
    route_cfg: RouteConfig,
    cfg: Config | None = None,
    *,
    constraints: list[str] | None = None,
    gates: bool = True,
) -> Plan:
    """Build the routing plan for every task the run measured, plus every task in ``workload_mix``.

    ``cfg`` (a loaded ``config.yaml``) is accepted for callers that already have one at hand, but nothing
    here needs it: the gate baselines are rebuilt straight from each task's own dataset
    (:func:`local_enough.route.gates.build_gate_context`), which is deterministic regardless of which models
    happen to be configured for serving. ``constraints`` overrides ``route_cfg.constraints.data_must_stay_local``
    (an empty list disables the local-only constraint entirely, used by the report's unconstrained router row).
    """
    del cfg
    run_dir = run if isinstance(run, RunDir) else RunDir(run)
    local_only_tasks = set(route_cfg.constraints.data_must_stay_local) if constraints is None else set(constraints)

    table = candidates_module.candidate_table(run_dir, specs, route_cfg, split="calib")
    task_names = sorted({*table.keys(), *route_cfg.workload_mix.keys()})
    relevant_specs = {name: specs[name] for name in task_names if name in specs}
    gate_ctx = gates_module.build_gate_context(run_dir, relevant_specs) if gates else None

    tasks = {
        name: _plan_task(name, specs.get(name), table.get(name, []), route_cfg, local_only_tasks, gate_ctx)
        for name in task_names
    }
    return Plan(run=_run_label(run), gates_enabled=gates, local_only_tasks=sorted(local_only_tasks), tasks=tasks)


def _fmt(value: float | None, spec: str = ".3f") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return format(value, spec)


def print_plan(plan: Plan) -> str:
    """A readable fixed-width table: task, status, bar, primary, fallbacks, gates, USD/1k of primary."""
    headers = ["TASK", "STATUS", "BAR", "PRIMARY", "FALLBACKS", "GATE", "USD/1K"]
    rows: list[list[str]] = []
    for name in sorted(plan.tasks):
        tp = plan.tasks[name]
        primary = tp.primary.model_id if tp.primary else "-"
        fallbacks = ",".join(f.model_id for f in tp.fallbacks) or "-"
        gate_label = tp.gate_label if plan.gates_enabled or tp.status == STATUS_UNSERVABLE else "off"
        usd = _fmt(tp.primary.usd_per_task * 1000, ".2f") if tp.primary else "n/a"
        rows.append([name, tp.status, _fmt(tp.bar_value), primary, fallbacks, gate_label, usd])

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i]) for i in range(len(headers))
    ]
    lines = [
        "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)),
        "  ".join("-" * w for w in widths),
    ]
    lines.extend("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)) for row in rows)
    return "\n".join(lines)
