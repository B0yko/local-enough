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

from matplotlib.figure import Figure

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


def frontier_chart(
    task: str, points: list[ChartPoint], *, quality_bar: float | None, split_label: str = "test"
) -> Figure:
    """One accuracy-vs-cost frontier chart: x = USD/1,000 tasks (log), y = primary metric with 95% CI."""
    fig = Figure(figsize=(6.4, 4.2))
    ax = fig.add_subplot(111)

    frontier = _pareto_frontier(points)
    if len(frontier) >= 2:
        ax.plot(
            [p.usd_per_1k for p in frontier],
            [p.primary for p in frontier],
            color="#c0392b",
            linewidth=1.2,
            zorder=1,
            label="Pareto frontier",
        )

    seen_kinds: set[ChartKind] = set()
    for p in sorted(points, key=lambda p: p.model_id):
        if math.isnan(p.primary) or math.isnan(p.usd_per_1k):
            continue
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
        annotation = BASELINE_LABEL if p.kind == "baseline" else p.model_id
        ax.annotate(annotation, (p.usd_per_1k, p.primary), textcoords="offset points", xytext=(6, 4), fontsize=7)

    if quality_bar is not None and not math.isnan(quality_bar):
        ax.axhline(quality_bar, color="#666666", linewidth=1.0, linestyle="--", label="quality bar")

    ax.set_xscale("log")
    ax.set_xlabel("USD per 1,000 tasks (log scale)")
    ax.set_ylabel("primary metric")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{task}: accuracy vs. cost ({split_label} split)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles, strict=True))
        ax.legend(by_label.values(), by_label.keys(), loc="lower right", frameon=False)
    fig.tight_layout()
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
