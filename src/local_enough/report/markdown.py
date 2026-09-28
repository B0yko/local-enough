"""Renders a :class:`~local_enough.report.tables.ReportData` to the 14 named Markdown blocks.

Each ``render_<name>`` function is pure text: no file I/O, no charts (those are composed on top in
``report.build``). ``BLOCK_NAMES`` and :func:`render_blocks` are what ``readme_blocks`` and
``scripts/check_readme.py`` use; ``report.build`` reuses the same functions for ``report.md`` and
(via a Markdown-to-HTML pass) ``index.html``, so the three outputs never drift from each other.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from local_enough.report import format as fmt
from local_enough.report.quality import VERDICT_RULE_TEXT
from local_enough.report.tables import PRIMARY_METRIC_NAME, ReportData

BLOCK_NAMES: tuple[str, ...] = (
    "headline",
    "setup",
    "results",
    "local_perf",
    "break_even",
    "sensitivity",
    "verdicts",
    "router",
    "live_check",
    "judge",
    "data",
    "downloads",
    "spend",
    "plan",
)


def _esc(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if not rows:
        return "*(no data)*\n"
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(_esc(c) for c in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def render_headline(data: ReportData) -> str:
    return data.headline + "\n"


def render_setup(data: ReportData) -> str:
    s = data.setup
    lines = [
        _table(
            ["field", "value"],
            [
                ["hardware", s.get("hardware_name") or fmt.NA],
                ["machine", s.get("machine") or fmt.NA],
                ["macOS", f"{s.get('macos_version') or fmt.NA} ({s.get('macos_build') or fmt.NA})"],
                ["mlx", s.get("mlx_version") or fmt.NA],
                ["mlx-lm", s.get("mlx_lm_version") or fmt.NA],
                ["judge model", s.get("judge_model") or "not calibrated"],
                ["run date", s.get("run_date") or fmt.NA],
                [
                    "hardware price",
                    f"{fmt.fmt_usd(s.get('hardware_price', {}).get('usd'), decimals=0)} "
                    f"({s.get('hardware_price', {}).get('label') or fmt.NA}), "
                    f"source: {s.get('hardware_price', {}).get('source_url') or fmt.NA}, "
                    f"{s.get('hardware_price', {}).get('date') or fmt.NA}",
                ],
                [
                    "electricity price (assumption)",
                    f"{fmt.fmt_number(s.get('electricity_usd_per_kwh'), decimals=2)} USD/kWh",
                ],
                ["power", s.get("power") or fmt.NOT_MEASURED],
                ["measurement windows", s.get("measurement_windows") or fmt.NOT_MEASURED],
                ["total API spend", fmt.fmt_usd(s.get("total_api_spend_usd"))],
            ],
        )
    ]
    lines.append("\n**Local models**\n")
    lines.append(
        _table(
            ["id", "repo", "revision", "size on disk"],
            [
                [m["id"], m.get("repo") or fmt.NA, m.get("revision") or fmt.NA, fmt.fmt_gb(m.get("size_bytes"))]
                for m in s.get("local_models", [])
            ],
        )
    )
    lines.append(f"\n**Cloud models** (price snapshot: {s.get('price_snapshot_date') or fmt.NA})\n")
    lines.append(
        _table(
            ["id", "role", "model"],
            [[m["id"], m.get("role") or fmt.NA, m.get("model") or fmt.NA] for m in s.get("cloud_models", [])],
        )
    )
    lines.append("\n**Split sizes**\n")
    lines.append(
        _table(
            ["task", "calib", "test"],
            [
                [task, fmt.fmt_int(sizes.get("calib")), fmt.fmt_int(sizes.get("test"))]
                for task, sizes in s.get("split_sizes", {}).items()
            ],
        )
    )
    return "\n".join(lines) + "\n"


def _cost_cell(row: dict[str, Any]) -> str:
    """USD per 1,000 tasks; a local model shows its fully loaded cost with the energy-only cost beside it."""
    cost = fmt.fmt_usd_per_1k(row["usd_per_task"])
    energy = row.get("energy_usd_per_task")
    if row["kind"] == "local" and energy is not None:
        return f"{cost} (energy {fmt.fmt_usd_per_1k(energy)})"
    return cost


def render_results(data: ReportData) -> str:
    out = []
    for task in data.tasks:
        block = data.results[task]
        metric_name = block["primary_metric"]
        out.append(f"#### {task}\n")
        out.append(
            f"Quality bar (calib): {fmt.fmt_pct(block['bar'])}. Split: test. Scenario: local dedicated at V_t.\n"
        )
        rows = [
            [
                r["model_id"],
                r["kind"],
                f"{fmt.fmt_pct(r['primary'])} {fmt.fmt_ci_pct(r['ci'][0], r['ci'][1])}",
                fmt.fmt_pct(r["invalid_output_rate"]),
                fmt.fmt_seconds(r["p50_s"]),
                fmt.fmt_seconds(r["p95_s"]),
                _cost_cell(r),
                fmt.fmt_bool(r["meets_bar_calib"]),
                fmt.fmt_bool(r["holds_test"]),
            ]
            for r in block["rows"]
        ]
        out.append(
            _table(
                [
                    "model",
                    "kind",
                    f"{metric_name} (95% CI)",
                    "invalid %",
                    "p50",
                    "p95",
                    "$/1k tasks",
                    "meets bar (calib)",
                    "holds (test)",
                ],
                rows,
            )
        )
        if block["calib_pass_test_fail"]:
            out.append(f"\nCalib-pass/test-fail: {', '.join(block['calib_pass_test_fail'])}.\n")
        out.append(f"\nReproduce: `local-enough report --run {data.run_label}`.\n")
    return "\n".join(out)


def _peak_memory_cell(m: dict[str, Any]) -> str:
    peak = m["peak_memory_bytes"]
    if peak is None:
        return fmt.NOT_MEASURED
    method = m["peak_memory_method"]
    return f"{fmt.fmt_gb(peak)} ({method})" if method else fmt.fmt_gb(peak)


def _combined_memory_line(data: ReportData, *, machine_gb: float | None) -> str:
    combined = data.local_perf.get("combined") or {}
    ours = combined.get("ours_bytes")
    if ours is None:
        return f"Both local models + router: {fmt.NOT_MEASURED}."
    method = str(combined.get("method") or "").split(" (", 1)[0]
    parts = [f"{model_id} {fmt.fmt_gib(b)}" for model_id, b in sorted((combined.get("models_bytes") or {}).items())]
    if combined.get("router_bytes") is not None:
        parts.append(f"router {fmt.fmt_gib(combined['router_bytes'])}")
    detail = "; ".join(x for x in [method, ", ".join(parts)] if x)
    line = f"Both local models + router: {fmt.fmt_gib(ours)}{f' ({detail})' if detail else ''}"
    used = combined.get("system_used_bytes")
    if used is not None:
        of = f" of {fmt.fmt_number(machine_gb, decimals=0)} GiB" if machine_gb else ""
        before = combined.get("system_used_before_bytes")
        before_text = f", {fmt.fmt_gib(before)} before loading them" if before is not None else ""
        line += f"; system memory in use at the time: {fmt.fmt_gib(used)}{of}{before_text} (includes other workloads)"
    return line + "."


def render_local_perf(data: ReportData) -> str:
    if not data.local_perf["models"]:
        return "*(no local models in this run)*\n"
    out = []
    for model_id, m in sorted(data.local_perf["models"].items()):
        out.append(f"#### {model_id}\n")
        out.append(
            _table(
                ["task", "tasks/hour (c=1, Pass A)", "tasks/hour (c=4, Pass B)"],
                [
                    [t["task"], fmt.fmt_number(t["tph_c1"], decimals=0), fmt.fmt_number(t["tph_c4"], decimals=0)]
                    for t in m["tasks"]
                ],
            )
        )
        out.append(
            _table(
                ["metric", "value"],
                [
                    [
                        "first-5-min throughput (soak)",
                        fmt.fmt_number(m["first5_tph"], decimals=0, missing=fmt.NOT_MEASURED),
                    ],
                    [
                        "last-5-min throughput (soak)",
                        fmt.fmt_number(m["last5_tph"], decimals=0, missing=fmt.NOT_MEASURED),
                    ],
                    [
                        "throttle factor (last5/first5)",
                        fmt.fmt_number(m["throttle_factor"], decimals=3, missing=fmt.NOT_MEASURED),
                    ],
                    ["peak memory", _peak_memory_cell(m)],
                    ["incremental watts", fmt.fmt_watts(m["incremental_watts"]) + f" ({m['watts_label']})"],
                    ["idle watts", fmt.fmt_watts(m["idle_watts"]) + f" ({m['watts_label']})"],
                ],
            )
        )
    out.append("\n" + _combined_memory_line(data, machine_gb=data.setup.get("memory_gb")) + "\n")
    out.append("\nReproduce: `local-enough bench --local-only --split test --concurrency 4`, `local-enough soak`.\n")
    return "\n".join(out)


def render_break_even(data: ReportData) -> str:
    rows_map = data.break_even["rows"]
    rows = [
        [
            task,
            fmt.fmt_int(r["v_t"]),
            r["best_local_model"] or fmt.NA,
            r["cheapest_cloud_model"] or "none meets the bar",
            fmt.fmt_usd_per_1k(r["cheapest_cloud_usd_per_task"]),
            fmt.fmt_usd_per_1k(r["energy_usd_per_task"]),
            fmt.fmt_usd(r["fixed_usd_per_month"], decimals=2),
            fmt.fmt_int(r["break_even_volume"]),
            fmt.fmt_int(r["capacity_per_month"]),
            fmt.fmt_int(r["machines_needed"]),
            r["verdict"],
        ]
        for task, r in rows_map.items()
    ]
    body = _table(
        [
            "task",
            "V_t (tasks/mo)",
            "best local model",
            "cheapest cloud (meets bar)",
            "cloud $/1k",
            "local energy $/1k",
            "local fixed $/mo",
            "break-even (tasks/mo)",
            "capacity (tasks/mo)",
            "machines needed",
            "verdict",
        ],
        rows,
    )
    reproduce = f"Reproduce: `local-enough report --run {data.run_label}`.\n"
    return f"Scenario: dedicated (one machine per task at V_t).\n\n{body}\n{reproduce}"


def render_sensitivity(data: ReportData) -> str:
    task = data.break_even["headline_task"]
    cells = data.break_even["sensitivity"]
    if not cells:
        note = data.break_even.get("sensitivity_note") or f"not measured for the headline task, {task}"
        return f"*(sensitivity grid {note})*\n"
    rows = [
        [
            fmt.fmt_number(c.lifetime_years, decimals=0),
            f"{c.price_multiplier:.2f}x",
            fmt.fmt_usd(c.fixed_usd_per_month, decimals=2),
            fmt.fmt_int(c.break_even.volume),
            c.break_even.verdict,
        ]
        for c in cells
    ]
    body = _table(["lifetime (years)", "price multiplier", "fixed $/mo", "break-even (tasks/mo)", "verdict"], rows)
    return f"Headline task: **{task}**. Scenario: dedicated.\n\n{body}"


def render_verdicts(data: ReportData) -> str:
    rows = [[v.task, v.label, v.detail] for v in data.verdicts]
    return f"{VERDICT_RULE_TEXT}\n\n{_table(['task', 'verdict', 'detail'], rows)}"


ROUTER_ROW_LABELS = {
    "all-frontier": "All traffic to frontier",
    "cheapest-single-cloud": "Cheapest single cloud model meeting every bar",
    "router (no gates, no constraints)": "Router, primary only (no gates), no constraints",
    "router (gates, no constraints)": "Router with gates, no constraints",
    "router (gates, route.yaml constraints)": "Router with gates + data_must_stay_local",
}


def _router_row_name(row: dict[str, Any]) -> str:
    label = str(row.get("label"))
    name = ROUTER_ROW_LABELS.get(label, label)
    per_task = row.get("per_task") or {}
    refused = sum(1 for v in per_task.values() if v.get("unservable"))
    if refused:
        name += f" ({len(per_task) - refused}/{len(per_task)} tasks served)"
    if row.get("model_id"):
        return f"{name} (`{row['model_id']}`)"
    if label == "cheapest-single-cloud":
        return f"{name}: none"
    return name


def _pct_value(value: Any) -> str:
    return f"{float(value):.1f}%" if isinstance(value, int | float) and math.isfinite(value) else fmt.NA


def _signed_pct_value(value: Any) -> str:
    return f"{float(value):+.1f}%" if isinstance(value, int | float) and math.isfinite(value) else fmt.NA


def _usd_1k(value: Any) -> str:
    return f"${float(value):,.3f}" if isinstance(value, int | float) and math.isfinite(value) else fmt.NA


def _secs(value: Any) -> str:
    return f"{float(value):.2f}" if isinstance(value, int | float) and math.isfinite(value) else fmt.NA


def render_router(data: ReportData) -> str:
    if not data.router.available:
        return f"{data.router.plan_text}\n"
    rows_data = data.router.mixed_rows
    if not rows_data:
        return "*(router simulation returned no rows)*\n"
    headers = [
        "configuration",
        "USD / 1k mixed tasks",
        "tasks meeting bar",
        "served locally",
        "escalated",
        "p50 / p95 s",
        "saving vs all-frontier",
        "saving vs cheapest cloud",
    ]
    rows = []
    for row in rows_data:
        per_task = row.get("per_task") or {}
        met = sum(1 for v in per_task.values() if v.get("meets_bar"))
        rows.append(
            [
                _router_row_name(row),
                _usd_1k(row.get("usd_per_1k")),
                f"{met}/{len(per_task)}" if per_task else fmt.NA,
                _pct_value(row.get("served_locally_pct")),
                _pct_value(row.get("escalation_pct")),
                f"{_secs(row.get('p50_s'))} / {_secs(row.get('p95_s'))}",
                _signed_pct_value(row.get("saving_vs_all_frontier_pct")),
                _signed_pct_value(row.get("saving_vs_cheapest_cloud_pct")),
            ]
        )
    tasks = sorted({t for row in rows_data for t in (row.get("per_task") or {})})
    per_rows = []
    for row in rows_data:
        per_task = row.get("per_task") or {}
        cells = [_router_row_name(row)]
        for task in tasks:
            v = per_task.get(task)
            if not v:
                cells.append(fmt.NA)
                continue
            if v.get("unservable"):
                cells.append("unservable (503)")
                continue
            metric, bar = v.get("metric"), v.get("bar")
            mark = "meets" if v.get("meets_bar") else "below"
            cells.append(f"{fmt.fmt_number(metric, decimals=3)} vs {fmt.fmt_number(bar, decimals=3)} ({mark})")
        per_rows.append(cells)
    return (
        f"Scenario: shared machine (mixed workload, `workload_mix` weights, full test split). "
        f"Source: `{data.router.source_cmd}`.\n\n{_table(headers, rows)}\n"
        f"Per-task primary metric against its calib bar:\n\n{_table(['configuration', *tasks], per_rows)}"
    )


def render_live_check(data: ReportData) -> str:
    lc: dict[str, Any] = data.live_check
    if not lc:
        return "*(not measured: no `live_check.json` in this run)*\n"
    rows = [
        ["n", fmt.fmt_int(lc.get("n"))],
        ["seed", fmt.fmt_int(lc.get("seed"))],
        ["created", lc.get("created_utc") or fmt.NA],
        ["router overhead p50", fmt.fmt_ms(lc.get("overhead_p50_ms"))],
        ["direct call p50", fmt.fmt_ms(lc.get("direct_p50_ms"))],
        ["router call p50", fmt.fmt_ms(lc.get("router_p50_ms"))],
        ["decision match rate", fmt.fmt_pct(lc.get("decision_match_rate"))],
        ["local-only requests that reached cloud", fmt.fmt_int(lc.get("local_only_cloud_calls"))],
    ]
    body = _table(["metric", "value"], rows)
    mismatches = lc.get("mismatches") or []
    note = f"\n{len(mismatches)} mismatch(es) between the live run and the simulation.\n" if mismatches else ""
    return body + note


def render_judge(data: ReportData) -> str:
    j = data.judge
    if not j:
        return "*(not judged: no `judge_calibration.json` in this run)*\n"
    summary = _table(
        ["field", "value"],
        [
            ["judge model", j.get("judge_model") or fmt.NA],
            ["n (judge-calib, selection)", fmt.fmt_int(j.get("judge_calib_n"))],
            ["n (judge-holdout, reported rates)", fmt.fmt_int(j.get("holdout_n"))],
            ["TPR", fmt.fmt_pct(j.get("tpr"))],
            ["TNR", fmt.fmt_pct(j.get("tnr"))],
            ["balanced accuracy", fmt.fmt_pct(j.get("balanced_accuracy"))],
            ["Cohen's kappa", fmt.fmt_number(j.get("kappa"), decimals=3)],
            ["target (>= 0.90 balanced accuracy) met?", fmt.fmt_bool(j.get("meets_target"))],
        ],
    )
    rows = [
        [
            r["model_id"],
            fmt.fmt_int(r["n"]),
            fmt.fmt_pct(r["raw_pass_rate"]),
            fmt.fmt_pct(r["corrected_pass_rate"]),
            fmt.fmt_ci_pct(r["ci"][0], r["ci"][1]),
            fmt.fmt_pct(r["key_token_agreement"]),
        ]
        for r in j.get("rows", [])
    ]
    per_model = _table(
        ["model", "n", "raw pass rate", "bias-corrected pass rate", "95% CI", "key-token agreement"], rows
    )
    return f"{summary}\n**Per-model summarisation pass rates (test split)**\n\n{per_model}"


def render_data(data: ReportData) -> str:
    rows = [
        [
            r["task"],
            r["source"] or fmt.NA,
            r["licence"] or fmt.NA,
            r["metric"] or fmt.NA,
            fmt.fmt_int(r["distinct_templates"]),
        ]
        for r in data.data_table
    ]
    return _table(["task", "source", "licence", "primary metric", "distinct templates"], rows)


def render_downloads(data: ReportData) -> str:
    d = data.downloads
    if not d.get("rows"):
        return "*(not measured: no `downloads.json` in this run)*\n"
    rows = [
        [
            r["model_id"],
            r["repo"] or fmt.NA,
            r["revision_sha"] or fmt.NA,
            fmt.fmt_gb(r["bytes"]),
            r["licence"] or fmt.NA,
        ]
        for r in d["rows"]
    ]
    body = _table(["model", "repo", "revision", "size", "licence"], rows)
    return f"{body}\nTotal downloaded: {fmt.fmt_gb(d.get('total_bytes'))}.\n"


def render_spend(data: ReportData) -> str:
    sp = data.spend
    not_recorded = "not recorded in this run"
    rows = [["total API spend", fmt.fmt_usd(sp.get("total_usd"))]]
    rows.extend([f"spend: {command}", fmt.fmt_usd(usd)] for command, usd in (sp.get("by_command") or {}).items())
    rows.extend(
        [
            ["budget cap", fmt.fmt_usd(sp.get("budget_usd"), missing=not_recorded)],
            ["budget warning level", fmt.fmt_usd(sp.get("budget_warn_usd"), missing=not_recorded)],
        ]
    )
    return _table(["field", "value"], rows)


def render_plan(data: ReportData) -> str:
    return f"```text\n{data.router.plan_text}\n```\n"


_RENDERERS = {
    "headline": render_headline,
    "setup": render_setup,
    "results": render_results,
    "local_perf": render_local_perf,
    "break_even": render_break_even,
    "sensitivity": render_sensitivity,
    "verdicts": render_verdicts,
    "router": render_router,
    "live_check": render_live_check,
    "judge": render_judge,
    "data": render_data,
    "downloads": render_downloads,
    "spend": render_spend,
    "plan": render_plan,
}


def render_blocks(data: ReportData) -> dict[str, str]:
    """Every named block, rendered to Markdown. What ``readme_blocks`` and ``check_readme.py`` use."""
    return {name: renderer(data) for name, renderer in _RENDERERS.items()}


__all__ = ["BLOCK_NAMES", "PRIMARY_METRIC_NAME", "render_blocks"]
