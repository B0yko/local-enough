"""Rogan-Gladen bias correction, the judged-pass-rate bootstrap, and run-dir readers for the report.

Kept separate from ``local_enough.stats`` (bootstrap CI, kappa, balanced accuracy) because these functions
are specific to the judge's two-sided calibration: correcting an observed pass rate for judge error measured
on a held-out set.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from local_enough.bench.rundir import RunDir

_ALPHA = 0.05


def tpr_tnr(labels: Sequence[bool], predictions: Sequence[bool]) -> tuple[float, float]:
    """True positive / true negative rate of ``predictions`` against ``labels``; ``nan`` for an empty group."""
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must be the same length")
    pos = [p for label, p in zip(labels, predictions, strict=True) if label]
    neg = [p for label, p in zip(labels, predictions, strict=True) if not label]
    tpr = sum(1 for p in pos if p) / len(pos) if pos else float("nan")
    tnr = sum(1 for p in neg if not p) / len(neg) if neg else float("nan")
    return tpr, tnr


def rogan_gladen(p_obs: float, tpr: float, tnr: float) -> float:
    """theta = (p_obs + tnr - 1) / (tpr + tnr - 1), clipped to [0, 1]; ``nan`` when ``tpr + tnr - 1 <= 0``."""
    denom = tpr + tnr - 1
    if math.isnan(denom) or denom <= 0:
        return float("nan")
    theta = (p_obs + tnr - 1) / denom
    return min(1.0, max(0.0, theta))


def corrected_pass_rate(
    evaluated_passed: Sequence[bool],
    holdout_items: Sequence[dict[str, Any]],
    n: int = 1000,
    seed: int = 7,
) -> dict[str, Any]:
    """Raw pass rate, its Rogan-Gladen correction, and a bootstrap CI resampling both sides.

    ``holdout_items`` are ``{"label_pass": bool, "predicted_pass": bool}`` records (the judge
    calibration holdout's ``per_item``). Each of the ``n`` resamples draws a fresh holdout sample
    (recomputing TPR/TNR) and a fresh evaluated sample (recomputing ``p_obs``) before correcting, so the
    interval reflects uncertainty in both the judge's measured error and the evaluated pass rate.
    """
    evaluated = list(evaluated_passed)
    holdout_labels = [bool(it["label_pass"]) for it in holdout_items]
    holdout_preds = [bool(it["predicted_pass"]) for it in holdout_items]

    raw = sum(evaluated) / len(evaluated) if evaluated else float("nan")
    tpr, tnr = tpr_tnr(holdout_labels, holdout_preds)
    corrected = rogan_gladen(raw, tpr, tnr)

    if not evaluated or not holdout_items:
        return {"raw": raw, "corrected": corrected, "ci": [float("nan"), float("nan")]}

    rng = random.Random(seed)
    k_eval = len(evaluated)
    k_hold = len(holdout_items)
    estimates: list[float] = []
    for _ in range(n):
        p_obs = sum(evaluated[rng.randrange(k_eval)] for _ in range(k_eval)) / k_eval
        hold_idx = [rng.randrange(k_hold) for _ in range(k_hold)]
        sample_labels = [holdout_labels[i] for i in hold_idx]
        sample_preds = [holdout_preds[i] for i in hold_idx]
        s_tpr, s_tnr = tpr_tnr(sample_labels, sample_preds)
        estimates.append(rogan_gladen(p_obs, s_tpr, s_tnr))

    finite = sorted(e for e in estimates if not math.isnan(e))
    if not finite:
        return {"raw": raw, "corrected": corrected, "ci": [float("nan"), float("nan")]}
    lo_idx = min(max(int((_ALPHA / 2) * len(finite)), 0), len(finite) - 1)
    hi_idx = min(max(int((1 - _ALPHA / 2) * len(finite)) - 1, 0), len(finite) - 1)
    return {"raw": raw, "corrected": corrected, "ci": [finite[lo_idx], finite[hi_idx]]}


def load_judge_scores(run: Path | RunDir) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Every ``judge_scores.jsonl.gz`` record, keyed by ``(model_id, split, item_id)``."""
    run_dir = run if isinstance(run, RunDir) else RunDir(run)
    return {(r["model_id"], r["split"], str(r["item_id"])): r for r in run_dir.iter_records("judge_scores.jsonl.gz")}


def judge_summary(run: Path | RunDir) -> dict[tuple[str, str], dict[str, Any]]:
    """Per ``(model_id, split)``: raw/bias-corrected pass rate + CI, mean fact recall, key-token agreement.

    Empty when the run has no judge calibration or no judge scores, so a report can show summarisation as
    "not judged" instead of failing.
    """
    run_dir = run if isinstance(run, RunDir) else RunDir(run)
    calibration = run_dir.read_json("judge_calibration.json")
    scores = list(run_dir.iter_records("judge_scores.jsonl.gz"))
    if calibration is None or not scores:
        return {}
    holdout_items = calibration["holdout"]["per_item"]

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in scores:
        groups.setdefault((r["model_id"], r["split"]), []).append(r)

    summary: dict[tuple[str, str], dict[str, Any]] = {}
    for key, records in groups.items():
        passed = [bool(r["passed"]) for r in records]
        rate = corrected_pass_rate(passed, holdout_items)
        recalls = [r["fact_recall"] for r in records if isinstance(r["fact_recall"], int | float)]
        recalls = [r for r in recalls if not math.isnan(r)]
        mean_recall = sum(recalls) / len(recalls) if recalls else float("nan")
        agreements = [kt["agree"] for r in records for kt in r.get("key_token", []) if kt.get("agree") is not None]
        key_token_agreement = sum(agreements) / len(agreements) if agreements else float("nan")
        summary[key] = {
            "raw_pass_rate": rate["raw"],
            "corrected_pass_rate": rate["corrected"],
            "ci": rate["ci"],
            "mean_fact_recall": mean_recall,
            "key_token_agreement": key_token_agreement,
            "n": len(records),
        }
    return summary
