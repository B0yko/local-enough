"""Writes the three report files from a run directory: ``index.html``, ``report.md``, an ADR.

Both files render the same :mod:`local_enough.report.markdown` blocks (report.md as Markdown
directly, index.html through :mod:`local_enough.report.htmlconv`), so they never say different
things about the same run. The ADR comes from ``templates/adr.md.j2``. Everything here is a pure
function of the run directory: no network, no wall-clock timestamps, so two builds from the same
run produce byte-identical files (``tests/test_report_build.py``).
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from local_enough.bench.rundir import RunDir
from local_enough.config import RouteConfig
from local_enough.report import charts
from local_enough.report.htmlconv import markdown_to_html
from local_enough.report.markdown import render_blocks
from local_enough.report.quality import VERDICT_RULE_TEXT
from local_enough.report.tables import ReportData, build_report_data
from local_enough.tasks.base import TaskSpec

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

_HTML_CSS = """
:root { color-scheme: light dark; }
body {
  font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  max-width: 960px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5;
}
h1, h2, h3, h4 { line-height: 1.25; }
table { border-collapse: collapse; width: 100%; margin: 0.75rem 0; font-size: 0.92rem; }
th, td { border: 1px solid #8884; padding: 0.35rem 0.55rem; text-align: left; }
th { background: #8882; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
pre { background: #8881; padding: 0.75rem; overflow-x: auto; }
section { margin-bottom: 2rem; }
svg { max-width: 100%; height: auto; }
"""

# (block name, section heading) in report order; "results" additionally gets a chart per task.
_SECTIONS: tuple[tuple[str, str], ...] = (
    ("setup", "Setup"),
    ("results", "Results"),
    ("local_perf", "Local performance"),
    ("break_even", "Break-even"),
    ("sensitivity", "Sensitivity"),
    ("verdicts", "Verdicts"),
    ("router", "Router (mixed workload)"),
    ("live_check", "Router live check"),
    ("judge", "Judge calibration"),
    ("data", "Data"),
    ("downloads", "Downloads"),
    ("spend", "Spend"),
    ("plan", "Router plan"),
)


def _adr_env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False, trim_blocks=True, lstrip_blocks=True
    )


def _render_adr(data: ReportData) -> str:
    template = _adr_env().get_template("adr.md.j2")
    return template.render(
        run_label=data.run_label,
        date=data.date,
        headline=data.headline,
        verdicts=data.verdicts,
        verdict_rule=VERDICT_RULE_TEXT,
        hardware_name=data.setup.get("hardware_name"),
        electricity_usd_per_kwh=data.setup.get("electricity_usd_per_kwh"),
        total_api_spend_usd=data.setup.get("total_api_spend_usd"),
        constrained_tasks=sorted(data.route_cfg.constraints.data_must_stay_local),
    )


def _report_md(data: ReportData, blocks: dict[str, str], img_dir_name: str) -> str:
    lines = [f"# local-enough report — {data.run_label} ({data.date})", "", blocks["headline"].strip(), ""]
    for name, title in _SECTIONS:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(blocks[name].strip())
        lines.append("")
        if name == "results":
            for task in data.tasks:
                lines.append(f"![{task} frontier chart]({img_dir_name}/{task}.png)")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _index_html(data: ReportData, blocks: dict[str, str], svg_by_task: dict[str, str]) -> str:
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>local-enough report — {data.run_label}</title>",
        f"<style>{_HTML_CSS}</style></head><body>",
        f"<h1>local-enough report — {data.run_label} ({data.date})</h1>",
        markdown_to_html(blocks["headline"]),
    ]
    for name, title in _SECTIONS:
        parts.append(f"<section><h2>{title}</h2>")
        parts.append(markdown_to_html(blocks[name]))
        if name == "results":
            for task in data.tasks:
                parts.append(f"<h4>{task} frontier chart</h4>")
                parts.append(svg_by_task[task])
        parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def build_report(
    run_dir: str | Path | RunDir,
    out_dir: str | Path,
    *,
    specs: dict[str, TaskSpec] | None = None,
    route_cfg: RouteConfig | None = None,
) -> list[Path]:
    """Write ``index.html``, ``report.md``, ``ADR-local-vs-cloud.md`` and ``img/<task>.png`` into ``out_dir``."""
    data = build_report_data(run_dir, specs=specs, route_cfg=route_cfg)
    blocks = render_blocks(data)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "img"

    svg_by_task: dict[str, str] = {}
    written: list[Path] = []
    for task in data.tasks:
        block = data.results[task]
        fig = charts.frontier_chart(task, block["chart_points"], quality_bar=block["bar"], split_label="test")
        svg_by_task[task] = charts.inline_svg(fig)
        png_path = img_dir / f"{task}.png"
        charts.save_png(fig, png_path)
        written.append(png_path)

    md_path = out / "report.md"
    md_path.write_text(_report_md(data, blocks, "img"), encoding="utf-8")
    written.append(md_path)

    html_path = out / "index.html"
    html_path.write_text(_index_html(data, blocks, svg_by_task), encoding="utf-8")
    written.append(html_path)

    adr_path = out / "ADR-local-vs-cloud.md"
    adr_path.write_text(_render_adr(data), encoding="utf-8")
    written.append(adr_path)

    return sorted(written)


def readme_blocks(
    run_dir: str | Path | RunDir, *, specs: dict[str, TaskSpec] | None = None, route_cfg: RouteConfig | None = None
) -> dict[str, str]:
    """Markdown for each of :data:`local_enough.report.markdown.BLOCK_NAMES`, from a run directory."""
    return render_blocks(build_report_data(run_dir, specs=specs, route_cfg=route_cfg))
