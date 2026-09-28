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


def _crowded_points() -> list[ChartPoint]:
    return [
        ChartPoint("open-large", "cloud", 0.984, 0.975, 0.99, 0.11),
        ChartPoint("open-same-family", "cloud", 0.977, 0.972, 0.983, 0.08),
        ChartPoint("small-closed", "cloud", 0.986, 0.98, 0.99, 0.26),
        ChartPoint("frontier", "cloud", 0.996, 0.993, 0.998, 4.06),
        ChartPoint("local-qwen3-4b", "local", 0.930, 0.923, 0.938, 9.22, 0.0087),
        ChartPoint("local-qwen2.5-1.5b", "local", 0.894, 0.886, 0.902, 9.22, 0.0063),
    ]


def test_crowded_point_labels_do_not_overlap_each_other_or_leave_the_axes() -> None:
    fig = frontier_chart("extraction", _crowded_points(), quality_bar=0.944)
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    ax = fig.axes[0]
    axes_box = ax.get_window_extent(renderer)
    boxes = [t.get_window_extent(renderer) for t in ax.texts]
    assert len(boxes) == len(_crowded_points())
    for i, a in enumerate(boxes):
        assert axes_box.x0 <= a.x0 and a.x1 <= axes_box.x1 and axes_box.y0 <= a.y0 and a.y1 <= axes_box.y1
        for b in boxes[i + 1 :]:
            assert not a.overlaps(b), "two labels overlap"


def test_a_single_dominating_baseline_still_draws_a_pareto_line() -> None:
    points = [
        _point("tfidf-baseline", "baseline", 0.88, BASELINE_FLOOR_USD_PER_1K),
        _point("frontier", "cloud", 0.85, 2.0),
    ]
    fig = frontier_chart("classification", points, quality_bar=0.83)
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.get_lines()]
    assert "Pareto frontier" in labels
    frontier_line = next(line for line in ax.get_lines() if line.get_label() == "Pareto frontier")
    ys = list(frontier_line.get_ydata())
    assert ys and all(math.isclose(y, 0.88) for y in ys)
    # The staircase runs on to the right edge of the plot, so the line spans the whole chart.
    assert frontier_line.get_xdata()[-1] >= ax.get_xlim()[1] * 0.99
