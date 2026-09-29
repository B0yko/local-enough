"""The accuracy-vs-cost frontier chart: one per task, deterministic SVG and PNG bytes, light and dark themes.

Uses the non-interactive ``Agg`` backend only (no display, no event loop). ``svg.hashsalt`` is
fixed so matplotlib's internal element ids (clip paths, markers) hash the same way on every render
of the same figure, and fonts/sizes are pinned so text layout does not depend on what happens to be
installed on the host. Charts are built twice from the same run directory in
``tests/test_report_build.py`` and compared byte-for-byte.

Every colour comes from a :class:`ChartTheme`; the same drawing code renders the light theme (report
and GitHub light mode) and the dark theme (GitHub dark mode, matching the repository banner).
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
from matplotlib.transforms import Bbox, blended_transform_factory

BASELINE_FLOOR_USD_PER_1K = 1e-4
BASELINE_TICK_LABEL = "≈$0"

_SANS = "DejaVu Sans"
_MONO = "DejaVu Sans Mono"

# Applied once at import time (not per-figure) so every chart in the process renders with the same
# deterministic hash salt, fonts and sizes, independent of what else runs in this interpreter.
matplotlib.rcParams.update(
    {
        "svg.hashsalt": "local-enough",
        "font.family": _SANS,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 100.0,
        "savefig.dpi": 150.0,
        "path.simplify": False,
    }
)

ChartKind = Literal["local", "cloud", "baseline"]
ThemeName = Literal["light", "dark"]


@dataclass(frozen=True)
class ChartTheme:
    """The colours one chart theme uses; every artist reads its colour from here."""

    name: ThemeName
    background: str
    text: str
    muted: str
    grid: str
    axis: str
    local: str
    cloud: str
    baseline: str
    frontier: str
    refused: str
    refused_alpha: float
    ci_alpha: float

    def kind_color(self, kind: ChartKind) -> str:
        return {"local": self.local, "cloud": self.cloud, "baseline": self.baseline}[kind]


LIGHT = ChartTheme(
    name="light",
    background="#ffffff",
    text="#111827",
    muted="#6b7280",
    grid="#eceef1",
    axis="#c9ced6",
    local="#16a34a",
    cloud="#2563eb",
    baseline="#6b7280",
    frontier="#374151",
    refused="#dc2626",
    refused_alpha=0.055,
    ci_alpha=0.42,
)

DARK = ChartTheme(
    name="dark",
    background="#0b0f0e",
    text="#e8ece9",
    muted="#8b948f",
    grid="#1a201e",
    axis="#2c3431",
    local="#4ade80",
    cloud="#60a5fa",
    baseline="#8b948f",
    frontier="#c7cdca",
    refused="#f87171",
    refused_alpha=0.065,
    ci_alpha=0.5,
)

THEMES: dict[ThemeName, ChartTheme] = {"light": LIGHT, "dark": DARK}

_MARKERS: dict[ChartKind, str] = {"local": "o", "cloud": "s", "baseline": "D"}
_MARKER_SIZE: dict[ChartKind, float] = {"local": 8.0, "cloud": 7.2, "baseline": 7.2}
_KIND_ORDER: tuple[ChartKind, ...] = ("local", "cloud", "baseline")


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


_LABEL_FONT_SIZE = 7.5
_LEADER_FROM_OFFSET_INDEX = 2
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

# Fixed page geometry (figure fractions) so every chart lines up the same way in a grid of charts.
_FIG_SIZE = (7.2, 4.5)
_AXES_RECT = {"left": 0.085, "right": 0.872, "bottom": 0.115, "top": 0.8}


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


def _usd_tick(value: float, _pos: float | None = None) -> str:
    """``$0.001``, ``$0.1``, ``$1``, ``$10``: plain dollars on the log axis, no scientific notation."""
    if math.isclose(value, BASELINE_FLOOR_USD_PER_1K, rel_tol=1e-9):
        return BASELINE_TICK_LABEL
    if value >= 1:
        return f"${value:,.0f}"
    decimals = max(0, -math.floor(math.log10(value) + 1e-9))
    return f"${value:.{decimals}f}"


def _style_axes(ax: Axes, theme: ChartTheme, has_baseline: bool) -> None:
    ax.set_facecolor(theme.background)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.axis)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", color=theme.grid, linewidth=0.8)
    ax.tick_params(axis="both", which="both", length=0, pad=6, colors=theme.muted, labelfontfamily=_MONO)

    # y: a handful of round ticks, none above 1.0 (the headroom above it is for labels only).
    y0, y1 = ax.get_ylim()
    y_ticks = [t for t in MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(y0, y1) if y0 <= t <= 1.0 + 1e-9]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.2f}"))

    # x: one tick per decade in plain dollars; a baseline's floor is labelled as roughly free.
    x0, x1 = ax.get_xlim()
    first, last = math.ceil(math.log10(x0) - 1e-9), math.floor(math.log10(x1) + 1e-9)
    x_ticks = [10.0**e for e in range(first, last + 1)]
    if has_baseline and x0 <= BASELINE_FLOOR_USD_PER_1K <= x1:
        x_ticks = [BASELINE_FLOOR_USD_PER_1K] + [t for t in x_ticks if t > BASELINE_FLOOR_USD_PER_1K * 3]
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(_usd_tick))


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
        boxes.append(Bbox([[x0, y - 3.5], [x1, y + 3.5]]))
    # The staircase: a horizontal run from each frontier point to the next, then the riser up to it.
    x_right = ax.get_xlim()[1]
    for i, p in enumerate(frontier):
        x, y = to_px(p.usd_per_1k, p.primary)
        nxt = frontier[i + 1] if i + 1 < len(frontier) else None
        x_end, y_end = to_px(nxt.usd_per_1k, nxt.primary) if nxt is not None else (to_px(x_right, p.primary)[0], y)
        boxes.append(Bbox([[x, y - 2], [x_end, y + 2]]))
        if nxt is not None:
            boxes.append(Bbox([[x_end - 2, y], [x_end + 2, y_end]]))
    return boxes


def _place_labels(
    fig: Figure,
    ax: Axes,
    points: list[ChartPoint],
    obstacles: list[Bbox],
    theme: ChartTheme,
    quality_bar: float | None,
) -> None:
    """Annotate every point, choosing per point the first offset whose text box collides with nothing.

    Deterministic: points are visited in a fixed order, offsets are tried in a fixed order, and text is measured
    with the bundled DejaVu Sans, so the same run always yields the same picture. Models under the quality bar get a
    quieter label so the ones that qualify read first.
    """
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    font = FontProperties(family=_SANS, size=_LABEL_FONT_SIZE)
    axes_box = ax.get_window_extent(renderer)
    placed: list[Bbox] = []
    order = sorted(points, key=lambda p: (p.usd_per_1k, -p.primary, p.model_id))
    has_bar = quality_bar is not None and not math.isnan(quality_bar)
    for p in order:
        text = p.model_id
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
                outside_area * 10 + sum(_overlap(box, b) for b in obstacles) + sum(_overlap(box, b) for b in placed) * 4
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
        below = has_bar and quality_bar is not None and p.primary < quality_bar
        ax.annotate(
            text,
            (p.usd_per_1k, p.primary),
            textcoords="offset points",
            xytext=(dx * scale, dy * scale),
            ha=ha,
            va=va,
            fontsize=_LABEL_FONT_SIZE,
            fontfamily=_SANS,
            color=theme.muted if below else theme.text,
            zorder=5,
            # A label that had to move away from its marker gets a thin leader line so it stays unambiguous.
            arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": theme.muted, "shrinkA": 0, "shrinkB": 4}
            if index >= _LEADER_FROM_OFFSET_INDEX
            else None,
        )


def _legend_handles(kinds: set[ChartKind], has_energy: bool, has_bar: bool, theme: ChartTheme) -> list[Line2D | Patch]:
    handles: list[Line2D | Patch] = []

    def marker(kind: ChartKind, label: str, *, hollow: bool = False) -> Line2D:
        color = theme.kind_color(kind)
        return Line2D(
            [],
            [],
            linestyle="none",
            marker=_MARKERS[kind],
            markersize=_MARKER_SIZE[kind] - 1,
            markerfacecolor=theme.background if hollow else color,
            markeredgecolor=color if hollow else theme.background,
            markeredgewidth=1.3 if hollow else 0.8,
            label=label,
        )

    if "local" in kinds:
        handles.append(marker("local", "local, full cost"))
        if has_energy:
            handles.append(marker("local", "local, energy only", hollow=True))
    if "cloud" in kinds:
        handles.append(marker("cloud", "cloud"))
    if "baseline" in kinds:
        handles.append(marker("baseline", "baseline"))
    handles.append(Line2D([], [], color=theme.frontier, linewidth=1.6, label="Pareto frontier"))
    if has_bar:
        handles.append(
            Patch(facecolor=theme.refused, alpha=min(1.0, theme.refused_alpha * 4), linewidth=0, label="below the bar")
        )
    return handles


METRIC_LABELS = {
    "accuracy": "accuracy",
    "f1": "match F1",
    "field_accuracy": "field accuracy",
    "f2": "span F2",
    "pass_rate": "judged pass rate",
    "primary": "primary metric",
}


def frontier_chart(
    task: str,
    points: list[ChartPoint],
    *,
    quality_bar: float | None,
    split_label: str = "test",
    theme: ChartTheme | ThemeName = "light",
    metric: str = "primary",
) -> Figure:
    """One accuracy-vs-cost frontier chart: x = USD/1,000 tasks (log), y = primary metric with 95% CI.

    The Pareto frontier is drawn as a staircase (the best metric reachable at or below each cost) so that a single
    dominating point, such as a free baseline, still shows as a line; labels are placed so they do not collide.
    ``theme`` picks the palette (``"light"`` or ``"dark"``, or a :class:`ChartTheme`); geometry is identical.
    """
    th = THEMES[theme] if isinstance(theme, str) else theme
    fig = Figure(figsize=_FIG_SIZE)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(th.background)
    fig.subplots_adjust(**_AXES_RECT)
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    xlim = _x_limits(points)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(*_y_limits(points, quality_bar))
    plotted = [
        p for p in sorted(points, key=lambda p: p.model_id) if not (math.isnan(p.primary) or math.isnan(p.usd_per_1k))
    ]
    kinds = {p.kind for p in plotted}
    _style_axes(ax, th, "baseline" in kinds)

    has_bar = quality_bar is not None and not math.isnan(quality_bar)
    if has_bar and quality_bar is not None:
        ax.axhspan(ax.get_ylim()[0], quality_bar, color=th.refused, alpha=th.refused_alpha, linewidth=0, zorder=0)
        ax.axhline(quality_bar, color=th.text, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.7, zorder=1)
        bar_text = f"{quality_bar:.3f}".rstrip("0").rstrip(".")
        fig.text(
            1.0,
            quality_bar,
            f" bar {bar_text} ",
            transform=blended_transform_factory(ax.transAxes, ax.transData),
            ha="left",
            va="center",
            fontsize=7.5,
            fontfamily=_MONO,
            color=th.text,
            bbox={
                "boxstyle": "round,pad=0.35,rounding_size=0.6",
                "facecolor": th.background,
                "edgecolor": th.axis,
                "linewidth": 0.8,
            },
        )

    frontier = _pareto_frontier(points)
    if frontier:
        x_right = ax.get_xlim()[1]
        ax.step(
            [p.usd_per_1k for p in frontier] + [x_right],
            [p.primary for p in frontier] + [frontier[-1].primary],
            where="post",
            color=th.frontier,
            linewidth=1.6,
            alpha=0.75,
            solid_joinstyle="miter",
            zorder=1.5,
            label="Pareto frontier",
        )

    has_energy = False
    for kind in _KIND_ORDER:
        for p in (q for q in plotted if q.kind == kind):
            color, marker = th.kind_color(p.kind), _MARKERS[p.kind]
            if not math.isnan(p.ci_lo) and not math.isnan(p.ci_hi):
                ax.vlines(
                    p.usd_per_1k,
                    min(p.ci_lo, p.primary),
                    max(p.ci_hi, p.primary),
                    color=color,
                    linewidth=2.2,
                    alpha=th.ci_alpha,
                    capstyle="round",
                    zorder=2.5,
                )
            if p.kind == "local" and p.usd_per_1k_energy is not None and not math.isnan(p.usd_per_1k_energy):
                has_energy = True
                ax.plot(
                    [p.usd_per_1k_energy, p.usd_per_1k],
                    [p.primary, p.primary],
                    color=color,
                    linewidth=1.1,
                    linestyle=(0, (1, 2.2)),
                    dash_capstyle="round",
                    alpha=0.8,
                    zorder=2,
                )
                ax.plot(
                    [p.usd_per_1k_energy],
                    [p.primary],
                    linestyle="none",
                    marker=marker,
                    markerfacecolor=th.background,
                    markeredgecolor=color,
                    markeredgewidth=1.4,
                    markersize=_MARKER_SIZE[p.kind] - 1,
                    zorder=3,
                )
            ax.plot(
                [p.usd_per_1k],
                [p.primary],
                linestyle="none",
                marker=marker,
                markerfacecolor=color,
                markeredgecolor=th.background,
                markeredgewidth=1.2,
                markersize=_MARKER_SIZE[p.kind],
                zorder=4,
            )

    ax.set_xlabel("USD per 1,000 tasks  ·  log scale", color=th.muted, labelpad=8)
    metric_label = METRIC_LABELS.get(metric, metric.replace("_", " "))
    ax.set_ylabel(f"{metric_label}  ·  95% CI", color=th.muted, labelpad=8)

    left = _AXES_RECT["left"]
    fig.text(left, 0.945, task, ha="left", va="baseline", fontsize=13, fontweight="bold", color=th.text)
    fig.text(
        left,
        0.895,
        f"{metric_label} vs. cost on the {split_label} split  ·  cheaper to the left, better to the top",
        ha="left",
        va="baseline",
        fontsize=8.5,
        color=th.muted,
    )
    if plotted:
        fig.legend(
            handles=_legend_handles(kinds, has_energy, has_bar, th),
            loc="lower left",
            bbox_to_anchor=(left - 0.008, _AXES_RECT["top"] + 0.022),
            ncol=6,
            frameon=False,
            handlelength=1.6,
            handletextpad=0.5,
            columnspacing=1.4,
            borderaxespad=0,
            borderpad=0,
            labelcolor=th.muted,
        )

    _place_labels(fig, ax, plotted, _obstacles(ax, plotted, frontier, quality_bar), th, quality_bar)
    return fig


_METADATA_BLOCK_RE = re.compile(r"\s*<metadata>.*?</metadata>", re.DOTALL)


def _strip_rdf_metadata(svg_text: str) -> str:
    """Drop matplotlib's RDF ``<metadata>`` block (creator string and links, no rendering effect).

    The chart files then depend only on the data and the theme, and the SVG makes no reference to
    an external site; the ``xmlns`` namespace URIs on the root ``<svg>`` element are unrelated
    (required XML identifiers, not fetched resources) and are left alone.
    """
    return _METADATA_BLOCK_RE.sub("", svg_text)


def save_svg(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_svg(fig), encoding="utf-8")


def save_png(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", metadata={"Software": None}, facecolor=fig.get_facecolor())


def _render_svg(fig: Figure) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None}, facecolor=fig.get_facecolor())
    return _strip_rdf_metadata(buf.getvalue())


def inline_svg(fig: Figure) -> str:
    """Render ``fig`` to an SVG string starting at ``<svg``, suitable for inline embedding in HTML."""
    text = _render_svg(fig)
    return text[text.index("<svg") :]
