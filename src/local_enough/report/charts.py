"""The accuracy-vs-cost frontier chart: one per task, deterministic SVG and PNG bytes.

Uses the non-interactive ``Agg`` backend only (no display, no event loop). ``svg.hashsalt`` is
fixed so matplotlib's internal element ids (clip paths, markers) hash the same way on every render
of the same figure, and fonts/sizes are pinned so text layout does not depend on what happens to be
installed on the host. Charts are built twice from the same run directory in
``tests/test_report_build.py`` and compared byte-for-byte.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox

BASELINE_FLOOR_USD_PER_1K = 1e-4
BASELINE_LABEL = "baseline (≈$0)"

# Applied once at import time (not per-figure) so every chart in the process renders with the same
# deterministic hash salt, fonts and sizes, independent of what else runs in this interpreter.
matplotlib.rcParams.update(
    {
        "svg.hashsalt": "local-enough",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 100.0,
        "savefig.dpi": 150.0,
        "path.simplify": False,
    }
)

ChartKind = Literal["local", "cloud", "baseline"]

_COLORS: dict[ChartKind, str] = {"local": "#1b7f3a", "cloud": "#1f5fa8", "baseline": "#8a8a8a"}
_MARKERS: dict[ChartKind, str] = {"local": "o", "cloud": "s", "baseline": "D"}


@dataclass(frozen=True)
class ChartPoint:
    """One model's plotted position on a task's accuracy-vs-cost frontier chart.

    ``usd_per_1k`` is the full, fully-loaded cost (dedicated scenario for local, measured for
    cloud); a baseline's is pre-floored by the caller to :data:`BASELINE_FLOOR_USD_PER_1K` so it is
    visible on the log-scale axis. ``usd_per_1k_energy`` is the local energy-only cost (hollow
    marker, joined to the full-cost point by a thin line); ``None`` for cloud and baselines.
    """

    model_id: str
    kind: ChartKind
    primary: float
    ci_lo: float
    ci_hi: float
    usd_per_1k: float
    usd_per_1k_energy: float | None = None


def _pareto_frontier(points: list[ChartPoint]) -> list[ChartPoint]:
    """The maximal Pareto front over (minimise cost, maximise primary metric)."""
    usable = [p for p in points if not math.isnan(p.primary) and not math.isnan(p.usd_per_1k)]
    ordered = sorted(usable, key=lambda p: (p.usd_per_1k, -p.primary, p.model_id))
    frontier: list[ChartPoint] = []
    best = -math.inf
    for p in ordered:
        if p.primary > best:
            frontier.append(p)
            best = p.primary
    return frontier


_LABEL_FONT_SIZE = 7
_LEADER_FROM_OFFSET_INDEX = 2
LegendLoc = Literal["lower right", "lower left", "upper left", "upper right"]
_LEGEND_LOCS: tuple[LegendLoc, ...] = ("lower right", "lower left", "upper left", "upper right")
# Label positions to try, most preferred first: (dx px, dy px, horizontal anchor, vertical anchor).
_LABEL_OFFSETS: tuple[tuple[float, float, str, str], ...] = (
    (9, 4, "left", "bottom"),
    (9, -4, "left", "top"),
    (-14, 6, "right", "bottom"),
    (-14, -6, "right", "top"),
    (0, 11, "center", "bottom"),
    (0, -11, "center", "top"),
    (9, 17, "left", "bottom"),
    (9, -17, "left", "top"),
    (-9, 17, "right", "bottom"),
    (-9, -17, "right", "top"),
    (0, 23, "center", "bottom"),
    (0, -23, "center", "top"),
    (9, 30, "left", "bottom"),
    (9, -30, "left", "top"),
    (-9, 30, "right", "bottom"),
    (-9, -30, "right", "top"),
)


def _overlap(a: Bbox, b: Bbox) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def _y_limits(points: list[ChartPoint], quality_bar: float | None) -> tuple[float, float]:
    """Full 0..1 range, or a zoomed lower bound when every point (and the bar) sits well above zero."""
    lows = [v for p in points for v in (p.ci_lo, p.primary) if not math.isnan(v)]
    if quality_bar is not None and not math.isnan(quality_bar):
        lows.append(quality_bar)
    top = 1.06
    if not lows:
        return -0.02, top
    zoomed = math.floor((min(lows) - 0.05) / 0.05) * 0.05
    return (round(zoomed, 2), top) if zoomed > 0.15 else (-0.02, top)


def _x_limits(points: list[ChartPoint]) -> tuple[float, float] | None:
    xs = [
        v for p in points for v in (p.usd_per_1k, p.usd_per_1k_energy) if v is not None and not math.isnan(v) and v > 0
    ]
    return (min(xs) / 2.2, max(xs) * 2.6) if xs else None


def _obstacles(ax: Axes, points: list[ChartPoint], frontier: list[ChartPoint], quality_bar: float | None) -> list[Bbox]:
    """Display-space boxes that a label should stay clear of: markers, CI whiskers, cost links, lines."""
    boxes: list[Bbox] = []

    def to_px(x: float, y: float) -> tuple[float, float]:
        px, py = ax.transData.transform((x, y))
        return float(px), float(py)

    for p in points:
        x, y = to_px(p.usd_per_1k, p.primary)
        boxes.append(Bbox([[x - 6, y - 6], [x + 6, y + 6]]))
        if not math.isnan(p.ci_lo) and not math.isnan(p.ci_hi):
            _, lo = to_px(p.usd_per_1k, p.ci_lo)
            _, hi = to_px(p.usd_per_1k, p.ci_hi)
            boxes.append(Bbox([[x - 5, min(lo, hi)], [x + 5, max(lo, hi)]]))
        if p.usd_per_1k_energy is not None and not math.isnan(p.usd_per_1k_energy):
            ex, _ = to_px(p.usd_per_1k_energy, p.primary)
            boxes.append(Bbox([[ex - 6, y - 6], [ex + 6, y + 6]]))
            boxes.append(Bbox([[min(ex, x), y - 1.5], [max(ex, x), y + 1.5]]))
    if quality_bar is not None and not math.isnan(quality_bar):
        (x0, y), (x1, _) = to_px(ax.get_xlim()[0], quality_bar), to_px(ax.get_xlim()[1], quality_bar)
        boxes.append(Bbox([[x0, y - 1.5], [x1, y + 1.5]]))
    del frontier  # the frontier line is thin and crossing it reads fine; only markers and links block labels
    return boxes


def _place_labels(
    fig: Figure, ax: Axes, points: list[ChartPoint], obstacles: list[Bbox], extra_blocked: list[Bbox]
) -> None:
    """Annotate every point, choosing per point the first offset whose text box collides with nothing.

    Deterministic: points are visited in a fixed order, offsets are tried in a fixed order, and text is measured
    with the bundled DejaVu Sans, so the same run always yields the same picture.
    """
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    font = FontProperties(size=_LABEL_FONT_SIZE)
    axes_box = ax.get_window_extent(renderer)
    placed: list[Bbox] = []
    blocked = [*obstacles, *extra_blocked]
    order = sorted(points, key=lambda p: (p.usd_per_1k, -p.primary, p.model_id))
    for p in order:
        text = BASELINE_LABEL if p.kind == "baseline" else p.model_id
        width, height, _ = renderer.get_text_width_height_descent(text, font, ismath=False)
        px, py = (float(v) for v in ax.transData.transform((p.usd_per_1k, p.primary)))
        best: tuple[float, int] | None = None
        for index, (dx, dy, ha, va) in enumerate(_LABEL_OFFSETS):
            x0 = px + dx - (0 if ha == "left" else width if ha == "right" else width / 2)
            y0 = py + dy - (0 if va == "bottom" else height)
            box = Bbox([[x0, y0], [x0 + width, y0 + height]])
            inside = _overlap(box, axes_box)
            outside_area = box.width * box.height - inside
            score = (
                outside_area * 10 + sum(_overlap(box, b) for b in blocked) + sum(_overlap(box, b) for b in placed) * 4
            )
            if best is None or score < best[0]:
                best = (score, index)
            if score == 0:
                break
        assert best is not None
        index = best[1]
        dx, dy, ha, va = _LABEL_OFFSETS[index]
        x0 = px + dx - (0 if ha == "left" else width if ha == "right" else width / 2)
        y0 = py + dy - (0 if va == "bottom" else height)
        placed.append(Bbox([[x0, y0], [x0 + width, y0 + height]]))
        scale = 72.0 / fig.dpi
        ax.annotate(
            text,
            (p.usd_per_1k, p.primary),
            textcoords="offset points",
            xytext=(dx * scale, dy * scale),
            ha=ha,
            va=va,
            fontsize=_LABEL_FONT_SIZE,
            # A label that had to move away from its marker gets a thin leader line so it stays unambiguous.
            arrowprops={"arrowstyle": "-", "linewidth": 0.5, "color": "#888888", "shrinkA": 0, "shrinkB": 3}
            if index >= _LEADER_FROM_OFFSET_INDEX
            else None,
        )


def frontier_chart(
    task: str, points: list[ChartPoint], *, quality_bar: float | None, split_label: str = "test"
) -> Figure:
    """One accuracy-vs-cost frontier chart: x = USD/1,000 tasks (log), y = primary metric with 95% CI.

    The Pareto frontier is drawn as a staircase (the best metric reachable at or below each cost) so that a single
    dominating point, such as a free baseline, still shows as a line; labels are placed so they do not collide.
    """
    fig = Figure(figsize=(6.4, 4.2))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    xlim = _x_limits(points)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(*_y_limits(points, quality_bar))

    frontier = _pareto_frontier(points)
    if frontier:
        x_right = ax.get_xlim()[1]
        ax.step(
            [p.usd_per_1k for p in frontier] + [x_right],
            [p.primary for p in frontier] + [frontier[-1].primary],
            where="post",
            color="#c0392b",
            linewidth=1.2,
            zorder=1,
            label="Pareto frontier",
        )

    seen_kinds: set[ChartKind] = set()
    plotted: list[ChartPoint] = []
    for p in sorted(points, key=lambda p: p.model_id):
        if math.isnan(p.primary) or math.isnan(p.usd_per_1k):
            continue
        plotted.append(p)
        color, marker = _COLORS[p.kind], _MARKERS[p.kind]
        label = p.kind if p.kind not in seen_kinds else None
        seen_kinds.add(p.kind)
        yerr = None
        if not math.isnan(p.ci_lo) and not math.isnan(p.ci_hi):
            yerr = [[max(0.0, p.primary - p.ci_lo)], [max(0.0, p.ci_hi - p.primary)]]
        ax.errorbar(
            [p.usd_per_1k],
            [p.primary],
            yerr=yerr,
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            markersize=7,
            zorder=3,
            label=label,
        )
        if p.kind == "local" and p.usd_per_1k_energy is not None and not math.isnan(p.usd_per_1k_energy):
            ax.plot([p.usd_per_1k_energy, p.usd_per_1k], [p.primary, p.primary], color=color, linewidth=0.8, zorder=2)
            ax.plot(
                [p.usd_per_1k_energy],
                [p.primary],
                marker=marker,
                markerfacecolor="none",
                markeredgecolor=color,
                markersize=7,
                zorder=3,
            )

    if quality_bar is not None and not math.isnan(quality_bar):
        ax.axhline(quality_bar, color="#666666", linewidth=1.0, linestyle="--", label="quality bar")

    ax.set_xlabel("USD per 1,000 tasks (log scale)")
    ax.set_ylabel("primary metric")
    ax.set_title(f"{task}: accuracy vs. cost ({split_label} split)")
    fig.tight_layout()

    obstacles = _obstacles(ax, plotted, frontier, quality_bar)
    legend_box: list[Bbox] = []
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles, strict=True))
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
        best_loc, best_cost = _LEGEND_LOCS[0], math.inf
        for loc in _LEGEND_LOCS:
            legend = ax.legend(by_label.values(), by_label.keys(), loc=loc, frameon=False)
            box = legend.get_window_extent(renderer)
            cost = sum(_overlap(box, b) for b in obstacles)
            legend.remove()
            if cost < best_cost:
                best_loc, best_cost = loc, cost
        legend = ax.legend(by_label.values(), by_label.keys(), loc=best_loc, frameon=False)
        legend_box = [legend.get_window_extent(renderer)]
    _place_labels(fig, ax, plotted, obstacles, legend_box)
    return fig


_METADATA_BLOCK_RE = re.compile(r"\s*<metadata>.*?</metadata>", re.DOTALL)


def _strip_rdf_metadata(svg_text: str) -> str:
    """Drop matplotlib's RDF ``<metadata>`` block (a ``matplotlib.org`` credit, no rendering effect).

    Keeps the chart fully self-contained with no reference to an external site; the ``xmlns``
    namespace URIs on the root ``<svg>`` element are unrelated (required XML identifiers, not
    fetched resources) and are left alone.
    """
    return _METADATA_BLOCK_RE.sub("", svg_text)


def save_svg(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_svg(fig), encoding="utf-8")


def save_png(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", metadata={"Software": None})


def _render_svg(fig: Figure) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    return _strip_rdf_metadata(buf.getvalue())


def inline_svg(fig: Figure) -> str:
    """Render ``fig`` to an SVG string starting at ``<svg``, suitable for inline embedding in HTML."""
    text = _render_svg(fig)
    return text[text.index("<svg") :]
