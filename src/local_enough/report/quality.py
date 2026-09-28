"""The quality bar, the per-task verdict rule and the headline sentence.

Kept independent of the route package (which may not be importable while ``m5`` is still landing):
callers pass in whatever router-derived facts they have (hybrid eligibility, router savings), or
``None``/``False`` when the router package is unavailable, and this module degrades the affected
output without failing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from local_enough.candidates import Candidate
from local_enough.config import RouteConfig
from local_enough.costmodel import BreakEven
from local_enough.report.format import NA, fmt_number, fmt_pct, fmt_signed_pct

VERDICT_LOCAL = "local"
VERDICT_LOCAL_CONSTRAINT = "local (constraint)"
VERDICT_LOCAL_BELOW_BAR_CONSTRAINT = "local — below bar (constraint)"
VERDICT_HYBRID = "hybrid"
VERDICT_CLOUD = "cloud"

VERDICT_RULE_TEXT = (
    "Verdict rule, decided on the calib split: **local** when a local candidate meets the quality bar and this "
    "task's monthly volume is at or above the break-even volume against the cheapest cloud model meeting the bar "
    "(itself at or below this machine's capacity); **local (constraint)** when the task is in "
    "`data_must_stay_local` and a local candidate meets the bar, whatever the break-even; "
    "**local — below bar (constraint)** when the task is in `data_must_stay_local` and no local candidate meets "
    "the bar; **hybrid** when local meets the bar only through the router, with escalation to cloud at or below "
    "20%; otherwise **cloud**."
)


def best_score(candidates: Sequence[Candidate]) -> float:
    """The best (highest) primary metric among ``candidates``, ignoring ``NaN`` scores. ``NaN`` if none score."""
    scores = [c.primary for c in candidates if not math.isnan(c.primary)]
    return max(scores) if scores else math.nan


def bar_value(route_cfg: RouteConfig, task: str, calib_candidates: Sequence[Candidate]) -> float:
    """The quality bar for ``task``: ``absolute``, or ``relative_to_best`` times the best calib score."""
    qb = route_cfg.bar_for(task)
    if qb.absolute is not None:
        return qb.absolute
    best = best_score(calib_candidates)
    if math.isnan(best) or qb.relative_to_best is None:
        return math.nan
    return qb.relative_to_best * best


def meets_bar(primary: float, bar: float) -> bool:
    return not math.isnan(primary) and not math.isnan(bar) and primary >= bar


def gap_to_bar(primary: float, bar: float) -> float:
    """How far below the bar ``primary`` is (0 when it meets or beats the bar); ``NaN`` when either is unknown."""
    if math.isnan(primary) or math.isnan(bar):
        return math.nan
    return max(0.0, bar - primary)


def local_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    return [c for c in candidates if c.is_local]


def local_llm_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Local LLMs only (excludes baselines), the population ``costing.local_costs`` prices for break-even."""
    return [c for c in candidates if c.kind == "local"]


@dataclass(frozen=True)
class TaskVerdict:
    task: str
    label: str
    detail: str


def task_verdict(
    task: str,
    *,
    is_local_only: bool,
    local_meets_bar: bool,
    gap: float,
    breaks_even_within_volume: bool,
    break_even_volume: float | None,
    v_t: float,
    hybrid_eligible: bool,
) -> TaskVerdict:
    """The deterministic per-task verdict (``VERDICT_RULE_TEXT``).

    ``breaks_even_within_volume`` is decided by the caller: ``costmodel.break_even`` returning
    ``"break-even"`` with ``V_t`` at or above that volume, or trivially ``True`` when the best local
    candidate is a free in-process baseline (no hardware to amortise, no capacity limit).
    """
    if is_local_only:
        if local_meets_bar:
            return TaskVerdict(task, VERDICT_LOCAL_CONSTRAINT, "data_must_stay_local; a local candidate meets the bar.")
        gap_text = NA if math.isnan(gap) else f"-{fmt_pct(gap)}"
        return TaskVerdict(
            task,
            VERDICT_LOCAL_BELOW_BAR_CONSTRAINT,
            f"data_must_stay_local; no local candidate meets the bar (gap to bar: {gap_text}).",
        )

    if local_meets_bar and breaks_even_within_volume:
        be_text = "immediately (free baseline)"
        if break_even_volume is not None:
            be_text = f"{fmt_number(break_even_volume)} tasks/month"
        return TaskVerdict(
            task,
            VERDICT_LOCAL,
            f"local meets the bar; break-even at {be_text}, at or below this task's volume "
            f"({fmt_number(v_t)} tasks/month) and this machine's capacity.",
        )
    if hybrid_eligible:
        return TaskVerdict(task, VERDICT_HYBRID, "local meets the bar only through the router, escalation <= 20%.")
    return TaskVerdict(
        task,
        VERDICT_CLOUD,
        "no local candidate meets the bar and breaks even within volume/capacity, and the hybrid router path "
        "does not qualify either.",
    )


def headline_task(met_bar_tasks: Sequence[str], workload_mix: Mapping[str, float]) -> tuple[str, bool]:
    """The task with the largest ``workload_mix`` weight among ``met_bar_tasks``, else ``("classification", False)``."""
    if not met_bar_tasks:
        return "classification", False
    chosen = max(sorted(met_bar_tasks), key=lambda t: workload_mix.get(t, 0.0))
    return chosen, True


def headline_text(
    *,
    met_bar_tasks: Sequence[str],
    total_tasks: int,
    workload_mix: Mapping[str, float],
    break_even_by_task: Mapping[str, BreakEven],
    v_t_by_task: Mapping[str, float],
    router_available: bool,
    router_saving_pct: float | None,
    router_saving_vs: str | None,
) -> str:
    """The 2-3 sentence headline verdict computed from the run."""
    task, local_met_a_bar = headline_task(met_bar_tasks, workload_mix)

    if local_met_a_bar:
        others = ", ".join(sorted(met_bar_tasks))
        sentence1 = f"Local met the quality bar on {len(met_bar_tasks)} of {total_tasks} tasks (calib): {others}."
    else:
        sentence1 = f"Local met the quality bar on none of the {total_tasks} measured tasks (calib)."
    if len(met_bar_tasks) * 2 < total_tasks:
        sentence1 += " Local failed the bar on most tasks."

    be = break_even_by_task.get(task)
    v_t = v_t_by_task.get(task, math.nan)
    if be is not None and be.verdict == "break-even" and be.volume is not None:
        sentence2 = (
            f"For the headline task ({task}), local breaks even against the cheapest cloud model meeting the bar "
            f"at {fmt_number(be.volume)} tasks/month, against this task's {fmt_number(v_t)} tasks/month volume."
        )
    else:
        verdict_label = be.verdict if be is not None else NA
        sentence2 = (
            f'For the headline task ({task}), the break-even verdict is "{verdict_label}" '
            f"at its {fmt_number(v_t)} tasks/month volume."
        )

    if not router_available:
        sentence3 = (
            "Router savings on the mixed workload are not available in this build (the route package is not present)."
        )
    elif router_saving_pct is None:
        sentence3 = (
            "No cloud model meets every task's bar, and the router table could not compute a mixed-workload saving."
        )
    else:
        vs = router_saving_vs or "the cheapest single cloud model meeting every bar"
        sentence3 = f"The router saves {fmt_signed_pct(router_saving_pct)} on the mixed workload against {vs}."

    return " ".join([sentence1, sentence2, sentence3])
