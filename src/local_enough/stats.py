"""Deterministic statistics: bootstrap confidence intervals, percentiles, kappa and balanced accuracy.

Every function here is a pure, seeded computation over plain sequences -- no dependency on a run
directory or task module -- so the same helpers serve bench metrics, the judge and the report.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Sequence


def bootstrap_ci[T](
    values_or_items: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    n: int = 1000,
    seed: int = 7,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """A percentile bootstrap CI for ``statistic`` over ``values_or_items``.

    ``values_or_items`` can be plain numbers or per-item records; ``statistic`` computes whatever
    aggregate is being resampled (accuracy, macro-F1, a judge's bias-corrected rate, ...) from one
    resample. Uses :class:`random.Random` (not numpy's global state) so runs never interfere with
    each other's seeding. Returns ``(nan, nan)`` for an empty input.
    """
    items = list(values_or_items)
    k = len(items)
    if k == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    estimates = sorted(statistic([items[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    lo_idx = min(max(int((alpha / 2) * n), 0), n - 1)
    hi_idx = min(max(int((1 - alpha / 2) * n) - 1, 0), n - 1)
    return (estimates[lo_idx], estimates[hi_idx])


def p50_p95(latencies_s: Sequence[float]) -> tuple[float, float]:
    """Linear-interpolated p50 and p95 (numpy-style), or ``(nan, nan)`` for an empty input."""
    if not latencies_s:
        return (float("nan"), float("nan"))
    return (_percentile(latencies_s, 50.0), _percentile(latencies_s, 95.0))


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    k = len(ordered)
    if k == 1:
        return ordered[0]
    rank = (q / 100.0) * (k - 1)
    lo = int(rank)
    hi = min(lo + 1, k - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def weighted_percentile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    """A percentile (``q`` in ``[0, 100]``) weighted by ``weights``.

    Sorts ``(value, weight)`` pairs by value and walks the cumulative weight; each point's
    cumulative weight is its bucket's upper edge, and the result linearly interpolates between the
    two points straddling ``q / 100`` of the total weight. This is a standard, simple weighted
    quantile method; it does not numerically match ``numpy.percentile`` on unit weights (that
    function uses a different, unweighted convention), and is not intended to.
    """
    if not values:
        return float("nan")
    pairs = sorted(zip(values, weights, strict=True))
    ordered_values = [v for v, _ in pairs]
    ordered_weights = [w for _, w in pairs]
    cum = list(itertools.accumulate(ordered_weights))
    total = cum[-1]
    if total <= 0:
        return float("nan")
    target = (q / 100.0) * total
    for i, (value, cum_weight) in enumerate(zip(ordered_values, cum, strict=True)):
        if target <= cum_weight:
            if i == 0:
                return value
            prev_cum = cum[i - 1]
            prev_value = ordered_values[i - 1]
            span = cum_weight - prev_cum
            frac = (target - prev_cum) / span if span > 0 else 0.0
            return prev_value + (value - prev_value) * frac
    return ordered_values[-1]


def cohen_kappa(a: Sequence[object], b: Sequence[object]) -> float:
    """Cohen's kappa between two label sequences of equal length. ``nan`` if either is empty."""
    if not a or len(a) != len(b):
        return float("nan")
    n = len(a)
    labels = sorted({*a, *b}, key=repr)
    a_counts = {label: 0 for label in labels}
    b_counts = {label: 0 for label in labels}
    agree = 0
    for x, y in zip(a, b, strict=True):
        a_counts[x] += 1
        b_counts[y] += 1
        if x == y:
            agree += 1
    po = agree / n
    pe = sum(a_counts[label] * b_counts[label] for label in labels) / (n * n)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def balanced_accuracy(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    """Macro-averaged per-class recall. ``nan`` if ``y_true`` is empty."""
    if not y_true or len(y_true) != len(y_pred):
        return float("nan")
    by_label: dict[object, list[int]] = {}
    for true, pred in zip(y_true, y_pred, strict=True):
        correct, total = by_label.setdefault(true, [0, 0])
        by_label[true][1] = total + 1
        if pred == true:
            by_label[true][0] = correct + 1
    recalls = [correct / total for correct, total in by_label.values() if total > 0]
    return sum(recalls) / len(recalls) if recalls else float("nan")
