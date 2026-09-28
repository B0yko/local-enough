"""Rogan-Gladen correction (incl. clipping and the degenerate case) and bootstrap determinism."""

from __future__ import annotations

import math

import pytest

from local_enough.bench.rundir import RunDir
from local_enough.judge.stats import corrected_pass_rate, judge_summary, load_judge_scores, rogan_gladen, tpr_tnr


def test_rogan_gladen_basic_correction() -> None:
    # p_obs=0.6, tpr=0.9, tnr=0.9 -> (0.6+0.9-1)/(0.9+0.9-1) = 0.5/0.8 = 0.625
    assert rogan_gladen(0.6, 0.9, 0.9) == pytest.approx(0.625)


def test_rogan_gladen_perfect_judge_is_identity() -> None:
    assert rogan_gladen(0.42, 1.0, 1.0) == pytest.approx(0.42)


def test_rogan_gladen_degenerate_denominator_is_nan() -> None:
    assert math.isnan(rogan_gladen(0.6, 0.5, 0.4))  # tpr+tnr-1 == -0.1 <= 0
    assert math.isnan(rogan_gladen(0.6, 0.5, 0.5))  # exactly 0


def test_rogan_gladen_clips_above_one() -> None:
    assert rogan_gladen(1.0, 0.6, 0.6) == pytest.approx(1.0)
    assert rogan_gladen(5.0, 0.9, 0.9) == 1.0


def test_rogan_gladen_clips_below_zero() -> None:
    assert rogan_gladen(-5.0, 0.9, 0.9) == 0.0


def test_tpr_tnr_basic() -> None:
    labels = [True, True, True, False, False]
    preds = [True, True, False, False, True]
    tpr, tnr = tpr_tnr(labels, preds)
    assert tpr == pytest.approx(2 / 3)
    assert tnr == pytest.approx(1 / 2)


def test_tpr_tnr_empty_group_is_nan() -> None:
    tpr, tnr = tpr_tnr([True, True], [True, False])
    assert tpr == pytest.approx(0.5)
    assert math.isnan(tnr)


def test_corrected_pass_rate_is_deterministic() -> None:
    holdout = [{"label_pass": True, "predicted_pass": True}] * 80
    holdout += [{"label_pass": True, "predicted_pass": False}] * 20
    holdout += [{"label_pass": False, "predicted_pass": False}] * 90
    holdout += [{"label_pass": False, "predicted_pass": True}] * 10
    evaluated = [True] * 60 + [False] * 40

    first = corrected_pass_rate(evaluated, holdout, n=200, seed=7)
    second = corrected_pass_rate(evaluated, holdout, n=200, seed=7)
    assert first == second


def test_corrected_pass_rate_raw_is_plain_mean() -> None:
    evaluated = [True, True, False, False]
    holdout = [{"label_pass": True, "predicted_pass": True}] * 10 + [
        {"label_pass": False, "predicted_pass": False}
    ] * 10
    result = corrected_pass_rate(evaluated, holdout, n=50, seed=1)
    assert result["raw"] == pytest.approx(0.5)
    assert result["corrected"] == pytest.approx(0.5)  # perfect judge -> no correction
    assert result["ci"][0] <= result["corrected"] <= result["ci"][1]


def test_corrected_pass_rate_empty_evaluated_is_nan() -> None:
    holdout = [{"label_pass": True, "predicted_pass": True}]
    result = corrected_pass_rate([], holdout)
    assert math.isnan(result["raw"])
    assert math.isnan(result["ci"][0])


def test_load_judge_scores_keys_by_model_split_item(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = RunDir(tmp_path)
    run.append_records(
        "judge_scores.jsonl.gz",
        [{"model_id": "m1", "split": "test", "item_id": "i1", "passed": True, "fact_recall": 1.0, "key_token": []}],
    )
    scores = load_judge_scores(tmp_path)
    assert scores[("m1", "test", "i1")]["passed"] is True


def test_judge_summary_empty_without_calibration_or_scores(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert judge_summary(tmp_path) == {}
    run = RunDir(tmp_path)
    run.write_json("judge_calibration.json", {"holdout": {"per_item": []}})
    assert judge_summary(tmp_path) == {}  # calibration present but no scores yet


def test_judge_summary_aggregates_per_model_split(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = RunDir(tmp_path)
    holdout_per_item = [{"id": f"h{i}", "label_pass": i % 2 == 0, "predicted_pass": i % 2 == 0} for i in range(20)]
    run.write_json("judge_calibration.json", {"holdout": {"per_item": holdout_per_item}})
    run.append_records(
        "judge_scores.jsonl.gz",
        [
            {
                "model_id": "cloud-a",
                "split": "test",
                "item_id": "i1",
                "passed": True,
                "fact_recall": 1.0,
                "key_token": [{"index": 0, "check": True, "judge": True, "agree": True}],
            },
            {
                "model_id": "cloud-a",
                "split": "test",
                "item_id": "i2",
                "passed": False,
                "fact_recall": 0.5,
                "key_token": [{"index": 0, "check": True, "judge": False, "agree": False}],
            },
        ],
    )
    summary = judge_summary(tmp_path)
    row = summary[("cloud-a", "test")]
    assert row["n"] == 2
    assert row["raw_pass_rate"] == pytest.approx(0.5)
    assert row["mean_fact_recall"] == pytest.approx(0.75)
    assert row["key_token_agreement"] == pytest.approx(0.5)
