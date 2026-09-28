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
from local_enough.costmodel import VERDICT_NO_CLOUD_BAR, BreakEven
from local_enough.report.format import NA, fmt_number, fmt_pct

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


BASELINE_NAMES: dict[str, str] = {
    "tfidf-baseline": "the TF-IDF baseline",
    "rapidfuzz-baseline": "the RapidFuzz baseline",
    "regex-baseline": "the regex baseline",
}


def baseline_name(model_id: str) -> str:
    """How prose names a non-LLM baseline: ``"the TF-IDF baseline"`` for ``tfidf-baseline``."""
    return BASELINE_NAMES.get(model_id, f"the {model_id}")


def local_winner_text(model_id: str, kind: str) -> str:
    """A local winner as prose: a baseline says so ("the TF-IDF baseline, no LLM"), a local model is its id."""
    return f"{baseline_name(model_id)}, no LLM" if kind == "baseline" else model_id


@dataclass(frozen=True)
class LocalWinner:
    """The best local candidate on a task where local met the bar."""

    model_id: str
    kind: str


@dataclass(frozen=True)
class Refusal:
    """A task the router refuses (HTTP 503): it must stay local and no local candidate meets its bar."""

    task: str
    metric_name: str
    best_local: float
    bar: float


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


def _escalation_text(rate: float) -> str:
    """An escalation share as whole percent, but never a misleading ``0%`` for a share that is not zero."""
    return "<1%" if 0 < rate < 0.005 else f"{rate * 100:.0f}%"


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
    free_baseline: str | None = None,
    local_alone_reason: str | None = None,
    hybrid_escalation_rate: float | None = None,
    hybrid_escalates_to: str | None = None,
) -> TaskVerdict:
    """The deterministic per-task verdict (``VERDICT_RULE_TEXT``).

    ``breaks_even_within_volume`` is decided by the caller: ``costmodel.break_even`` returning
    ``"break-even"`` with ``V_t`` at or above that volume, or trivially ``True`` when the best local
    candidate is a free in-process baseline (no hardware to amortise, no capacity limit); ``free_baseline`` then
    names it. ``local_alone_reason`` says why local alone did not apply, for the ``hybrid`` detail.
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
        if break_even_volume is None:
            who = f"{free_baseline} (no LLM)" if free_baseline else "the local baseline (no LLM)"
            return TaskVerdict(
                task,
                VERDICT_LOCAL,
                f"{who} meets the bar at about $0 per task, so there is no hardware to pay back at this task's "
                f"volume ({fmt_number(v_t)} tasks/month).",
            )
        return TaskVerdict(
            task,
            VERDICT_LOCAL,
            f"local meets the bar; break-even at {fmt_number(break_even_volume)} tasks/month, at or below this "
            f"task's volume ({fmt_number(v_t)} tasks/month) and this machine's capacity.",
        )
    if hybrid_eligible:
        detail = "local meets the bar only through the router, escalation <= 20%."
        if hybrid_escalation_rate is not None:
            via = f" to {hybrid_escalates_to}" if hybrid_escalates_to else ""
            through = (
                f"with gates and escalation{via} the router meets the bar "
                f"with {_escalation_text(hybrid_escalation_rate)} escalation"
            )
            detail = (
                f"{local_alone_reason}; {through}." if local_alone_reason else f"{through[0].upper()}{through[1:]}."
            )
        return TaskVerdict(task, VERDICT_HYBRID, detail)
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


def _refusal_sentence(r: Refusal) -> str:
    return (
        f"Keeping {r.task} local is not possible at its bar: best local {r.metric_name} {fmt_pct(r.best_local)} "
        f"vs {fmt_pct(r.bar)}, so the router refuses it with HTTP 503."
    )


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
    router_saving_scope: str | None = None,
    local_winners: Mapping[str, LocalWinner] | None = None,
    refusals: Sequence[Refusal] = (),
) -> str:
    """The 2-3 sentence headline verdict computed from the run.

    ``router_saving_pct`` is already a percentage (``76.9`` means 76.9%), as ``mixed_table`` reports it.
    """
    task, local_met_a_bar = headline_task(met_bar_tasks, workload_mix)

    # Sentence 1: where local met the bar (calib), and any task that must stay local but cannot.
    n_met = len(met_bar_tasks)
    if local_met_a_bar:
        winners = local_winners or {}
        parts = [
            f"{t} ({local_winner_text(winners[t].model_id, winners[t].kind)})" if t in winners else t
            for t in sorted(met_bar_tasks)
        ]
        joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"
        if n_met * 2 < total_tasks:
            sentence1 = (
                f"Local failed the quality bar on most tasks: on the calib split a local candidate met it on only "
                f"{n_met} of {total_tasks}, {joined}"
            )
        else:
            sentence1 = (
                f"On the calib split a local candidate met the quality bar on {n_met} of {total_tasks} tasks: {joined}"
            )
    else:
        sentence1 = (
            f"Local failed the quality bar on every task: on the calib split no local candidate met it on any of the "
            f"{total_tasks} tasks"
        )
    refusal_clauses = [
        f"{r.task}, which must stay local, missed its bar (best local {r.metric_name} {fmt_pct(r.best_local)} vs "
        f"{fmt_pct(r.bar)}), so the router refuses it"
        for r in refusals
    ]
    sentence1 += ("; " + "; ".join(refusal_clauses) if refusal_clauses else "") + "."

    # Sentence 2: the headline task's break-even.
    be = break_even_by_task.get(task)
    v_t = v_t_by_task.get(task, math.nan)
    prefix = f"The headline task, {task}," if local_met_a_bar else f"The fallback headline task, {task},"
    if be is not None and be.verdict == "break-even" and be.volume is not None:
        sentence2 = (
            f"{prefix} breaks even against the cheapest cloud model meeting the bar at {fmt_number(be.volume)} "
            f"tasks/month (its volume: {fmt_number(v_t)} tasks/month)."
        )
    elif be is not None and be.verdict == VERDICT_NO_CLOUD_BAR:
        sentence2 = (
            f"{prefix} has no break-even volume to compute: no cloud model met its bar "
            f"(volume: {fmt_number(v_t)} tasks/month)."
        )
    else:
        verdict_label = be.verdict if be is not None else NA
        sentence2 = f'{prefix} gets the break-even verdict "{verdict_label}" at {fmt_number(v_t)} tasks/month.'

    # Sentence 3: the router's saving on the mixed workload.
    if not router_available:
        sentence3 = "The router table is not available for this run."
    elif router_saving_pct is None:
        sentence3 = "No cloud model meets every task's bar, and the router table could not compute a saving."
    else:
        vs = router_saving_vs or "the cheapest single cloud model meeting every bar"
        who = f"The router ({router_saving_scope})" if router_saving_scope else "The router"
        if router_saving_pct >= 0:
            sentence3 = f"{who} costs {router_saving_pct:.1f}% less on the mixed workload than {vs}."
        else:
            sentence3 = f"{who} costs {-router_saving_pct:.1f}% more on the mixed workload than {vs}."
    return " ".join([sentence1, sentence2, sentence3])
