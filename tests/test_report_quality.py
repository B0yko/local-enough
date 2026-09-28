"""The deterministic per-task verdict rule and the headline sentence."""

from __future__ import annotations

import math

from local_enough.candidates import Candidate
from local_enough.config import RouteConfig
from local_enough.costmodel import VERDICT_NO_CLOUD_BAR, BreakEven
from local_enough.report.quality import (
    VERDICT_CLOUD,
    VERDICT_HYBRID,
    VERDICT_LOCAL,
    VERDICT_LOCAL_BELOW_BAR_CONSTRAINT,
    VERDICT_LOCAL_CONSTRAINT,
    LocalWinner,
    Refusal,
    bar_value,
    best_score,
    gap_to_bar,
    headline_task,
    headline_text,
    meets_bar,
    task_verdict,
)


def test_verdict_local_when_bar_met_and_breaks_even_within_volume() -> None:
    v = task_verdict(
        "classification",
        is_local_only=False,
        local_meets_bar=True,
        gap=0.0,
        breaks_even_within_volume=True,
        break_even_volume=1000.0,
        v_t=2000.0,
        hybrid_eligible=False,
    )
    assert v.label == VERDICT_LOCAL
    assert "1,000" in v.detail


def test_verdict_local_constraint_when_data_must_stay_local_and_bar_met() -> None:
    v = task_verdict(
        "pii_redaction",
        is_local_only=True,
        local_meets_bar=True,
        gap=0.0,
        breaks_even_within_volume=False,
        break_even_volume=None,
        v_t=2000.0,
        hybrid_eligible=False,
    )
    assert v.label == VERDICT_LOCAL_CONSTRAINT


def test_verdict_local_below_bar_constraint_when_data_must_stay_local_and_bar_missed() -> None:
    v = task_verdict(
        "pii_redaction",
        is_local_only=True,
        local_meets_bar=False,
        gap=0.05,
        breaks_even_within_volume=False,
        break_even_volume=None,
        v_t=2000.0,
        hybrid_eligible=False,
    )
    assert v.label == VERDICT_LOCAL_BELOW_BAR_CONSTRAINT
    assert "5.0%" in v.detail


def test_verdict_hybrid_when_router_escalation_is_low() -> None:
    v = task_verdict(
        "extraction",
        is_local_only=False,
        local_meets_bar=False,
        gap=0.1,
        breaks_even_within_volume=False,
        break_even_volume=None,
        v_t=2000.0,
        hybrid_eligible=True,
    )
    assert v.label == VERDICT_HYBRID


def test_verdict_cloud_otherwise() -> None:
    v = task_verdict(
        "summarisation",
        is_local_only=False,
        local_meets_bar=False,
        gap=0.2,
        breaks_even_within_volume=False,
        break_even_volume=None,
        v_t=2000.0,
        hybrid_eligible=False,
    )
    assert v.label == VERDICT_CLOUD


def test_verdict_local_not_reached_when_volume_below_break_even() -> None:
    """A local candidate meeting the bar but not yet at break-even volume is not `local`."""
    v = task_verdict(
        "classification",
        is_local_only=False,
        local_meets_bar=True,
        gap=0.0,
        breaks_even_within_volume=False,
        break_even_volume=5000.0,
        v_t=2000.0,
        hybrid_eligible=False,
    )
    assert v.label == VERDICT_CLOUD


def test_headline_task_picks_largest_workload_mix_weight_among_bar_meeting_tasks() -> None:
    task, met = headline_task(["extraction", "classification"], {"extraction": 0.25, "classification": 0.30})
    assert (task, met) == ("classification", True)


def test_headline_task_defaults_to_classification_when_none_met_bar() -> None:
    task, met = headline_task([], {"extraction": 0.25, "classification": 0.30})
    assert (task, met) == ("classification", False)


def test_headline_text_when_some_tasks_meet_the_bar() -> None:
    text = headline_text(
        met_bar_tasks=["classification", "entity_matching", "extraction"],
        total_tasks=5,
        workload_mix={"classification": 0.30, "extraction": 0.25, "entity_matching": 0.15},
        break_even_by_task={"classification": BreakEven(volume=10_000.0, verdict="break-even", machines_needed=1)},
        v_t_by_task={"classification": 15_000.0},
        router_available=False,
        router_saving_pct=None,
        router_saving_vs=None,
    )
    assert "Local met the quality bar on 3 of 5 tasks (calib)" in text
    assert "classification" in text
    assert "Local failed the bar on most tasks" not in text
    assert "10,000 tasks/month" in text
    assert "not available in this build" in text


def test_headline_text_when_no_task_meets_the_bar() -> None:
    text = headline_text(
        met_bar_tasks=[],
        total_tasks=5,
        workload_mix={"classification": 0.30},
        break_even_by_task={},
        v_t_by_task={"classification": 15_000.0},
        router_available=False,
        router_saving_pct=None,
        router_saving_vs=None,
    )
    assert "Local met the quality bar on none of the 5 measured tasks (calib)" in text
    assert "Local failed the bar on most tasks" in text
    assert "classification" in text  # the fallback headline task


def test_headline_text_reports_router_saving_when_available() -> None:
    text = headline_text(
        met_bar_tasks=["classification"],
        total_tasks=5,
        workload_mix={"classification": 1.0},
        break_even_by_task={"classification": BreakEven(volume=10_000.0, verdict="break-even", machines_needed=1)},
        v_t_by_task={"classification": 15_000.0},
        router_available=True,
        router_saving_pct=32.0,
        router_saving_vs="the cheapest single cloud model meeting every bar",
    )
    # The saving is already a percentage: it must not be scaled a second time (76.9 used to read "+7690.0%").
    assert "saves 32.0% on the mixed workload" in text
    assert "%" in text and "3200" not in text


def test_bar_value_relative_to_best_ignores_nan_scores() -> None:
    def _cand(model_id: str, primary: float) -> Candidate:
        return Candidate(
            model_id=model_id,
            task="classification",
            split="calib",
            kind="cloud",
            provider="vendor",
            role=None,
            n=10,
            primary=primary,
            primary_ci=(math.nan, math.nan),
            metrics={},
            ci={},
            invalid_output_rate=0.0,
            p50_s=1.0,
            p95_s=1.0,
            usd_per_task=0.001,
        )

    route_cfg = RouteConfig(workload_mix={"classification": 1.0}, reference_monthly_volume=1000)
    candidates = [_cand("a", 0.9), _cand("b", math.nan)]
    assert best_score(candidates) == 0.9
    bar = bar_value(route_cfg, "classification", candidates)
    assert math.isclose(bar, 0.95 * 0.9)
    assert meets_bar(0.8, bar) is False
    assert meets_bar(1.0, bar) is True
    assert math.isclose(gap_to_bar(0.5, bar), bar - 0.5)
    assert gap_to_bar(1.0, bar) == 0.0


def _headline(**overrides):
    kwargs = dict(
        met_bar_tasks=["classification", "entity_matching"],
        total_tasks=5,
        workload_mix={"classification": 0.30, "entity_matching": 0.15},
        break_even_by_task={"classification": BreakEven(volume=None, verdict=VERDICT_NO_CLOUD_BAR, machines_needed=1)},
        v_t_by_task={"classification": 15_000.0},
        router_available=True,
        router_saving_pct=76.9,
        router_saving_vs="all traffic to the frontier model (no single cloud model meets every bar)",
    )
    kwargs.update(overrides)
    return headline_text(**kwargs)


def test_headline_saving_is_not_scaled_twice() -> None:
    text = _headline(router_saving_pct=94.98)
    assert "saves 95.0% on the mixed workload" in text
    assert "9498" not in text


def test_headline_names_the_local_winner_per_task_and_flags_a_non_llm_baseline() -> None:
    text = _headline(
        local_winners={
            "classification": LocalWinner("tfidf-baseline", "baseline"),
            "entity_matching": LocalWinner("local-qwen2.5-1.5b", "local"),
        }
    )
    assert "classification (the TF-IDF baseline, no LLM)" in text
    assert "entity_matching (local-qwen2.5-1.5b)" in text


def test_headline_says_no_break_even_when_no_cloud_model_met_the_bar() -> None:
    text = _headline()
    assert "there is no break-even to compute because no cloud model met the bar" in text
    assert "n/a — no cloud model meets the bar" not in text


def test_headline_names_the_refused_task_and_its_gap_to_the_bar() -> None:
    text = _headline(
        router_saving_scope="gates, no route.yaml constraints",
        refusals=[Refusal("pii_redaction", "f2", 0.92, 0.95)],
    )
    assert "The router (gates, no route.yaml constraints) saves 76.9%" in text
    assert (
        "Keeping pii_redaction local is not possible at its bar: best local f2 92.0% vs 95.0%, "
        "so the router refuses it with HTTP 503." in text
    )


def test_verdict_for_a_free_baseline_has_no_break_even_volume() -> None:
    v = task_verdict(
        "classification",
        is_local_only=False,
        local_meets_bar=True,
        gap=0.0,
        breaks_even_within_volume=True,
        break_even_volume=None,
        v_t=15_000.0,
        hybrid_eligible=False,
        free_baseline="the TF-IDF baseline",
    )
    assert v.label == VERDICT_LOCAL
    assert "immediately" not in v.detail
    assert "break-even at" not in v.detail
    assert v.detail.startswith("the TF-IDF baseline (no LLM) meets the bar at about $0 per task")


def test_hybrid_verdict_explains_why_local_alone_did_not_apply() -> None:
    v = task_verdict(
        "entity_matching",
        is_local_only=False,
        local_meets_bar=True,
        gap=0.0,
        breaks_even_within_volume=False,
        break_even_volume=5_064_689.0,
        v_t=7_500.0,
        hybrid_eligible=True,
        local_alone_reason="local alone is cheaper only above 5,064,689 tasks/month",
        hybrid_escalation_rate=0.03,
        hybrid_escalates_to="open-same-family",
    )
    assert v.label == VERDICT_HYBRID
    assert v.detail == (
        "local alone is cheaper only above 5,064,689 tasks/month; with gates and escalation to open-same-family "
        "the router meets the bar with 3% escalation."
    )
