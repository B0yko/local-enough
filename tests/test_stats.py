from __future__ import annotations

import math

import numpy as np
import pytest

from local_enough.stats import (
    balanced_accuracy,
    bootstrap_ci,
    cohen_kappa,
    p50_p95,
    weighted_percentile,
)


def test_bootstrap_ci_is_deterministic_for_a_fixed_seed() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    lo1, hi1 = bootstrap_ci(values, lambda s: sum(s) / len(s), n=500, seed=7)
    lo2, hi2 = bootstrap_ci(values, lambda s: sum(s) / len(s), n=500, seed=7)
    assert (lo1, hi1) == (lo2, hi2)


def test_bootstrap_ci_different_seed_can_differ() -> None:
    values = list(range(1, 30))
    lo7, hi7 = bootstrap_ci(values, lambda s: sum(s) / len(s), n=200, seed=7)
    lo8, hi8 = bootstrap_ci(values, lambda s: sum(s) / len(s), n=200, seed=8)
    assert (lo7, hi7) != (lo8, hi8)


def test_bootstrap_ci_brackets_the_mean_for_varied_data() -> None:
    values = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # mean ~0.714
    lo, hi = bootstrap_ci(values, lambda s: sum(s) / len(s), n=2000, seed=7)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi


def test_bootstrap_ci_constant_input_is_a_point() -> None:
    values = [3.0] * 10
    lo, hi = bootstrap_ci(values, lambda s: sum(s) / len(s))
    assert lo == hi == 3.0


def test_bootstrap_ci_empty_is_nan() -> None:
    lo, hi = bootstrap_ci([], lambda s: 0.0)
    assert math.isnan(lo) and math.isnan(hi)


@pytest.mark.parametrize("data", [[1, 2, 3, 4, 5], [10.0], [5.0, 1.0, 3.0, 9.0, 2.0, 7.0]])
def test_p50_p95_matches_numpy_linear_interpolation(data: list[float]) -> None:
    p50, p95 = p50_p95(data)
    assert p50 == pytest.approx(float(np.percentile(data, 50)))
    assert p95 == pytest.approx(float(np.percentile(data, 95)))


def test_p50_p95_empty_is_nan() -> None:
    p50, p95 = p50_p95([])
    assert math.isnan(p50) and math.isnan(p95)


def test_weighted_percentile_unit_weights_hand_computed() -> None:
    values = [5.0, 1.0, 3.0, 9.0, 2.0, 7.0]
    weights = [1.0] * len(values)
    assert weighted_percentile(values, weights, 50.0) == pytest.approx(3.0)
    assert weighted_percentile(values, weights, 95.0) == pytest.approx(8.4)


def test_weighted_percentile_uneven_weights_hand_computed() -> None:
    values = [10.0, 20.0, 30.0]
    weights = [1.0, 1.0, 2.0]
    assert weighted_percentile(values, weights, 50.0) == pytest.approx(20.0)
    assert weighted_percentile(values, weights, 75.0) == pytest.approx(25.0)


def test_weighted_percentile_is_monotonic_in_q() -> None:
    values = [5.0, 1.0, 3.0, 9.0, 2.0, 7.0]
    weights = [3.0, 1.0, 2.0, 1.0, 4.0, 1.0]
    quantiles = [weighted_percentile(values, weights, q) for q in (5, 25, 50, 75, 95)]
    assert quantiles == sorted(quantiles)


def test_weighted_percentile_skews_toward_heavier_weight() -> None:
    values = [1.0, 100.0]
    weights = [99.0, 1.0]
    # almost all the weight is on 1.0, so even p95 should stay close to 1.0
    assert weighted_percentile(values, weights, 95.0) < 2.0


def test_weighted_percentile_empty_is_nan() -> None:
    assert math.isnan(weighted_percentile([], [], 50.0))


def test_cohen_kappa_perfect_agreement_is_one() -> None:
    labels = ["a", "b", "a", "b", "c"]
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_cohen_kappa_chance_agreement_is_low() -> None:
    # classic textbook-style confusion counts
    a = ["yes"] * 40 + ["no"] * 60
    b = ["yes"] * 20 + ["no"] * 20 + ["yes"] * 20 + ["no"] * 40
    kappa = cohen_kappa(a, b)
    assert -1.0 <= kappa <= 1.0
    assert kappa < 1.0


def test_cohen_kappa_empty_is_nan() -> None:
    assert math.isnan(cohen_kappa([], []))


def test_balanced_accuracy_perfect_is_one() -> None:
    y_true = ["a", "b", "a", "b", "c", "c"]
    assert balanced_accuracy(y_true, y_true) == pytest.approx(1.0)


def test_balanced_accuracy_handles_class_imbalance() -> None:
    # 90 "no" (all correct) and 10 "yes" (all wrong): plain accuracy would be 90%, balanced is 50%.
    y_true = ["no"] * 90 + ["yes"] * 10
    y_pred = ["no"] * 90 + ["no"] * 10
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_balanced_accuracy_empty_is_nan() -> None:
    assert math.isnan(balanced_accuracy([], []))
