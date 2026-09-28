"""Chart geometry: the Pareto frontier and deterministic SVG/PNG rendering."""

from __future__ import annotations

import math

from local_enough.report.charts import (
    BASELINE_FLOOR_USD_PER_1K,
    ChartPoint,
    _pareto_frontier,
    frontier_chart,
    inline_svg,
)


def _point(model_id: str, kind: str, primary: float, usd_per_1k: float) -> ChartPoint:
    return ChartPoint(
        model_id=model_id, kind=kind, primary=primary, ci_lo=math.nan, ci_hi=math.nan, usd_per_1k=usd_per_1k
    )


def test_pareto_frontier_drops_dominated_points() -> None:
    points = [
        _point("cheap-bad", "local", 0.5, 1.0),
        _point("mid-better", "cloud", 0.8, 5.0),
        _point("expensive-best", "cloud", 0.95, 50.0),
        _point("dominated", "cloud", 0.7, 10.0),  # worse quality, higher cost than mid-better: never on the front
    ]
    frontier = _pareto_frontier(points)
    assert [p.model_id for p in frontier] == ["cheap-bad", "mid-better", "expensive-best"]


def test_pareto_frontier_ignores_nan_points() -> None:
    points = [_point("a", "local", math.nan, 1.0), _point("b", "cloud", 0.9, 5.0)]
    assert [p.model_id for p in _pareto_frontier(points)] == ["b"]


def test_baseline_floor_is_a_small_positive_constant() -> None:
    assert 0 < BASELINE_FLOOR_USD_PER_1K < 1e-2


def test_frontier_chart_renders_deterministic_inline_svg_with_no_metadata_block() -> None:
    points = [
        _point("local-a", "local", 0.9, 2.0),
        _point("cloud-a", "cloud", 0.95, 20.0),
        _point("baseline-a", "baseline", 0.6, BASELINE_FLOOR_USD_PER_1K),
    ]
    fig1 = frontier_chart("classification", points, quality_bar=0.9)
    fig2 = frontier_chart("classification", points, quality_bar=0.9)
    svg1, svg2 = inline_svg(fig1), inline_svg(fig2)
    assert svg1 == svg2
    assert svg1.startswith("<svg")
    assert "<metadata>" not in svg1
    assert "matplotlib.org" not in svg1


def test_frontier_chart_handles_empty_points() -> None:
    fig = frontier_chart("classification", [], quality_bar=None)
    svg = inline_svg(fig)
    assert svg.startswith("<svg")
