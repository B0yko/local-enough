"""Assembles every report table from a run directory into one :class:`ReportData`.

This is the single place that reads run-directory files and shared helpers (``candidates``,
``costing``, ``costmodel``, ``judge.stats``) and turns them into plain, JSON-friendly rows that
``report.markdown`` renders to Markdown/HTML blocks and ``report.charts`` turns into figures.
Missing optional files (soak, power, memory, judge, live-check, downloads) degrade to "not
measured"/"not judged" fields rather than raising, so a partial run always renders.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_enough import costing, costmodel, evaluate, paths
from local_enough.bench.rundir import RunDir
from local_enough.bench.runner import DEFAULT_ROUTE_CONFIG
from local_enough.candidates import Candidate, candidate_table
from local_enough.config import HardwareConfig, RouteConfig
from local_enough.costing import LocalCost
from local_enough.costmodel import BreakEven, SensitivityCell
from local_enough.judge import stats as judge_stats
from local_enough.report import format as fmt
from local_enough.report import quality
from local_enough.report.charts import BASELINE_FLOOR_USD_PER_1K, ChartKind, ChartPoint
from local_enough.report.quality import TaskVerdict
from local_enough.tasks import registry
from local_enough.tasks.base import TaskSpec

PRIMARY_METRIC_NAME: dict[str, str] = {
    "classification": "accuracy",
    "extraction": "field_accuracy",
    "pii_redaction": "f2",
    "summarisation": "pass_rate",
    "entity_matching": "f1",
}


@dataclass(frozen=True)
class RouterFacts:
    """Router-derived facts for the report; degrades cleanly when ``local_enough.route`` is missing."""

    available: bool
    plan_text: str = "router table unavailable: the `local_enough.route` package is not present in this build."
    mixed_rows: list[dict[str, Any]] = field(default_factory=list)
    hybrid_by_task: dict[str, bool] = field(default_factory=dict)
    saving_pct: float | None = None
    saving_vs: str | None = None
    saving_scope: str | None = None
    """Which router row the headline saving comes from, when it is not the deployed (gates + constraints) row."""
    refused_tasks: list[str] = field(default_factory=list)
    """Tasks the deployed router refuses with HTTP 503 (must stay local, no local candidate meets the bar)."""
    hybrid_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_cmd: str = ""


@dataclass(frozen=True)
class ReportData:
    run: RunDir
    run_label: str
    date: str
    specs: dict[str, TaskSpec]
    tasks: list[str]
    route_cfg: RouteConfig
    setup: dict[str, Any]
    results: dict[str, Any]
    local_perf: dict[str, Any]
    break_even: dict[str, Any]
    verdicts: list[TaskVerdict]
    router: RouterFacts
    live_check: dict[str, Any]
    judge: dict[str, Any]
    data_table: list[dict[str, Any]]
    downloads: dict[str, Any]
    spend: dict[str, Any]
    headline: str


def run_label(run: RunDir) -> str:
    """``"reference"`` for the bundled reference run, else the run directory's own name (never a full path)."""
    try:
        if run.path.resolve() == paths.reference_run_dir().resolve():
            return "reference"
    except OSError:
        pass
    return run.path.name or "run"


LEDGER_COMMAND_ORDER: tuple[str, ...] = ("bench", "judge calibrate", "judge score", "route", "route replay")


def _ledger_by_command(run: RunDir) -> dict[str, float]:
    """``actual_usd`` in the run's own ``cost_ledger.jsonl`` copy, summed per ledger ``command``.

    Known commands come first in pipeline order, any others after them alphabetically.
    """
    records: list[dict[str, Any]] = []
    if run.exists("cost_ledger.jsonl.gz"):  # the committed reference run stores its ledger copy compressed
        records = list(run.iter_records("cost_ledger.jsonl.gz"))
    elif run.exists("cost_ledger.jsonl"):
        with run.file("cost_ledger.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not records:
        return {}
    totals: dict[str, float] = {}
    for record in records:
        command = str(record.get("command") or "unknown")
        totals[command] = totals.get(command, 0.0) + float(record.get("actual_usd") or 0.0)
    known = [c for c in LEDGER_COMMAND_ORDER if c in totals]
    return {c: totals[c] for c in [*known, *sorted(set(totals) - set(known))]}


def _default_route_cfg(run: RunDir) -> RouteConfig:
    data = run.read_json("route.json")
    return RouteConfig.model_validate(data) if data else DEFAULT_ROUTE_CONFIG


MACHINE_FAMILY: dict[str, str] = {
    "Mac13,1": "Mac Studio",
    "Mac13,2": "Mac Studio",
    "Mac14,13": "Mac Studio",
    "Mac14,14": "Mac Studio",
    "Mac16,9": "Mac Studio",
}
DESKTOP_FAMILIES = frozenset({"Mac Studio", "Mac mini", "iMac", "Mac Pro"})


def machine_family(machine_model: str | None) -> str | None:
    """The product name for a hardware identifier we know (``Mac16,9`` -> ``Mac Studio``), else ``None``."""
    return MACHINE_FAMILY.get(machine_model or "")


def _machine_text(env: dict[str, Any]) -> str | None:
    """``"Mac Studio (Mac16,9), Apple M4 Max, 128 GB"`` from ``env.json``; parts that are missing are left out."""
    model, chip, memory = env.get("machine_model"), env.get("chip"), env.get("memory_gb")
    family = machine_family(model)
    if family and model:
        head: str | None = f"{family} ({model})"
    else:
        head = model or None
    parts = [head, chip, fmt.fmt_gb_whole(memory, missing="") or None]
    text = ", ".join(str(p) for p in parts if p)
    return text or None


def _power_text(run: RunDir, cfg: dict[str, Any], env: dict[str, Any]) -> str:
    """How the watts behind the energy cost were obtained: measured, or configured (and why not measured)."""
    hw_power = (cfg.get("hardware") or {}).get("power") or {}
    idle, incremental = hw_power.get("idle_watts"), hw_power.get("incremental_watts")
    telemetry = run.read_json("power.json", {}) or {}
    modes = {str(v.get("mode")) for v in telemetry.values() if isinstance(v, dict)}
    reasons = sorted({str(v["reason"]) for v in telemetry.values() if isinstance(v, dict) and v.get("reason")})
    watts = f"{fmt.fmt_number(idle, decimals=0)} W idle, {fmt.fmt_number(incremental, decimals=0)} W incremental"
    if hw_power.get("mode") == "configured" or not modes or modes == {"unavailable"}:
        if not hw_power:
            return fmt.NOT_MEASURED
        if modes == {"unavailable"} and machine_family(env.get("machine_model")) in DESKTOP_FAMILIES:
            return f"configured: {watts}, from Apple's published figures; no battery telemetry on a desktop"
        why = f"; measured telemetry unavailable ({'; '.join(reasons)})" if reasons else ""
        return f"configured: {watts}{why}"
    return f"measured (battery telemetry): {watts}"


def _measurement_windows_text(run: RunDir) -> str:
    """Count of load-sample windows and how many were flagged contaminated by other workloads."""
    samples = run.read_json("load_samples.json", {}) or {}
    windows = samples.get("windows") if isinstance(samples, dict) else None
    if not isinstance(windows, list) or not windows:
        return fmt.NOT_MEASURED
    flagged = sum(1 for w in windows if isinstance(w, dict) and w.get("contaminated"))
    return f"{len(windows)} load-sample windows, {flagged} flagged contaminated"


def _build_setup(run: RunDir, specs: dict[str, TaskSpec], tasks: list[str]) -> dict[str, Any]:
    env = run.read_json("env.json", {}) or {}
    cfg = run.read_json("config.json", {}) or {}
    price_snapshot = run.read_json("price_snapshot.json", {}) or {}
    downloads = run.read_json("downloads.json", {}) or {}
    calibration = run.read_json("judge_calibration.json")

    models_cfg: list[dict[str, Any]] = cfg.get("models", [])
    env_models: dict[str, str] = env.get("models", {}) or {}

    local_rows = []
    for m in models_cfg:
        if m.get("kind") != "local":
            continue
        download = downloads.get(m["id"], {}) or {}
        display = env_models.get(m["id"])
        launch = m.get("launch") or {}
        repo = download.get("repo") or launch.get("repo")
        revision = download.get("revision_sha") or launch.get("revision")
        if display and "@" in display and not download.get("revision_sha"):
            repo, _, revision = display.partition("@")
        local_rows.append({"id": m["id"], "repo": repo, "revision": revision, "size_bytes": download.get("bytes")})

    call_stats: dict[str, dict[str, float]] = {}
    for rec in run.predictions("A"):
        st = call_stats.setdefault(str(rec["model_id"]), {"calls": 0, "retries": 0, "failed": 0, "usd": 0.0})
        st["calls"] += 1
        st["retries"] += int(rec.get("retries") or 0)
        st["failed"] += 1 if str(rec.get("error") or "").startswith("http") else 0
        st["usd"] += float(rec.get("cost_usd") or 0.0)
    cloud_rows = [
        {"id": m["id"], "role": m.get("role"), "model": m.get("model"), **call_stats.get(m["id"], {})}
        for m in models_cfg
        if m.get("kind") == "cloud"
    ]

    split_sizes: dict[str, dict[str, int | None]] = {}
    for name in tasks:
        spec = specs[name]
        sizes: dict[str, int | None] = {}
        for split in ("calib", "test"):
            try:
                sizes[split] = len(registry.load_items(spec, split))
            except FileNotFoundError:
                sizes[split] = None
        split_sizes[name] = sizes

    spend_by_command = _ledger_by_command(run)
    return {
        "machine_model": env.get("machine_model"),
        "chip": env.get("chip"),
        "memory_gb": env.get("memory_gb"),
        "machine": _machine_text(env),
        "power": _power_text(run, cfg, env),
        "measurement_windows": _measurement_windows_text(run),
        "macos_version": env.get("macos_version"),
        "macos_build": env.get("macos_build"),
        "mlx_version": env.get("mlx_version"),
        "mlx_lm_version": env.get("mlx_lm_version"),
        "local_models": local_rows,
        "cloud_models": cloud_rows,
        "judge_model": calibration.get("selected") if calibration else None,
        "price_snapshot_date": (price_snapshot.get("taken_at_utc") or "")[:10] or None,
        "split_sizes": split_sizes,
        "hardware_name": (cfg.get("hardware") or {}).get("name"),
        "hardware_price": env.get("hardware_price") or {},
        "electricity_usd_per_kwh": (cfg.get("hardware") or {}).get("electricity_usd_per_kwh"),
        "total_api_spend_usd": math.fsum(spend_by_command.values()),
        "spend_by_command": spend_by_command,
        "run_date": (env.get("start_utc") or "")[:10] or None,
    }


def _row_sort_key(c: Candidate) -> tuple[int, str]:
    order = {"local": 0, "cloud": 1, "baseline": 2}
    return (order.get(c.kind, 3), c.model_id)


def _build_results(
    tasks: list[str],
    specs: dict[str, TaskSpec],
    calib: dict[str, list[Candidate]],
    test: dict[str, list[Candidate]],
    bar_by_task: dict[str, float],
    local_costs_map: dict[tuple[str, str], LocalCost],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in tasks:
        calib_c, test_c = calib.get(task, []), test.get(task, [])
        bar = bar_by_task[task]
        calib_by_id = {c.model_id: c for c in calib_c}
        rows: list[dict[str, Any]] = []
        drift: list[str] = []
        chart_points: list[ChartPoint] = []
        for c in sorted(test_c, key=_row_sort_key):
            calib_cand = calib_by_id.get(c.model_id)
            calib_primary = calib_cand.primary if calib_cand else math.nan
            meets_calib = quality.meets_bar(calib_primary, bar)
            meets_test = quality.meets_bar(c.primary, bar)
            if meets_calib and not meets_test:
                drift.append(c.model_id)
            energy_per_task: float | None = None
            if c.kind == "local":
                lc_row = local_costs_map.get((c.model_id, task))
                if lc_row is not None and math.isfinite(lc_row.energy_usd_per_task):
                    energy_per_task = lc_row.energy_usd_per_task
            rows.append(
                {
                    "model_id": c.model_id,
                    "kind": c.kind,
                    "energy_usd_per_task": energy_per_task,
                    "role": c.role,
                    "primary": c.primary,
                    "ci": c.primary_ci,
                    "invalid_output_rate": c.invalid_output_rate,
                    "p50_s": c.p50_s,
                    "p95_s": c.p95_s,
                    "usd_per_task": c.usd_per_task,
                    "meets_bar_calib": meets_calib,
                    "holds_test": meets_test,
                    "judged": c.judged,
                }
            )
            kind: ChartKind = "local" if c.kind == "local" else "baseline" if c.kind == "baseline" else "cloud"
            x_energy: float | None = None
            if kind == "baseline":
                x = BASELINE_FLOOR_USD_PER_1K
            else:
                x = c.usd_per_task * 1000.0 if not math.isnan(c.usd_per_task) else math.nan
                if kind == "local":
                    lc = local_costs_map.get((c.model_id, task))
                    if lc is not None and math.isfinite(lc.energy_usd_per_task):
                        x_energy = lc.energy_usd_per_task * 1000.0
            chart_points.append(
                ChartPoint(
                    model_id=c.model_id,
                    kind=kind,
                    primary=c.primary,
                    ci_lo=c.primary_ci[0],
                    ci_hi=c.primary_ci[1],
                    usd_per_1k=x,
                    usd_per_1k_energy=x_energy,
                )
            )
        out[task] = {
            "kind": specs[task].kind,
            "primary_metric": PRIMARY_METRIC_NAME.get(specs[task].kind, "primary"),
            "bar": bar,
            "rows": rows,
            "calib_pass_test_fail": drift,
            "chart_points": chart_points,
        }
    return out


def _build_local_perf(run: RunDir, local_costs_map: dict[tuple[str, str], LocalCost]) -> dict[str, Any]:
    cfg = run.read_json("config.json", {}) or {}
    local_ids = [m["id"] for m in cfg.get("models", []) if m.get("kind") == "local"]
    soak = run.read_json("soak.json", {}) or {}
    memory = run.read_json("memory.json", {}) or {}

    per_model: dict[str, Any] = {}
    for model_id in local_ids:
        rows = sorted((lc for (mid, _), lc in local_costs_map.items() if mid == model_id), key=lambda lc: lc.task)
        soak_entry = soak.get(model_id) or {}
        mem_entry = memory.get(model_id) or {}
        first = rows[0] if rows else None
        per_model[model_id] = {
            "tasks": [
                {
                    "task": lc.task,
                    "tph_c1": lc.tasks_per_hour_c1,
                    "tph_c4": lc.tasks_per_hour_c4,
                    "sustained_tph": lc.sustained_tasks_per_hour,
                    "capacity_per_month": lc.capacity_per_month,
                }
                for lc in rows
            ],
            "applied_throttle": first.throttle_factor if first else None,
            "throttle_source": first.throttle_source if first else None,
            "first5_tph": soak_entry.get("first5_tph"),
            "last5_tph": soak_entry.get("last5_tph"),
            "throttle_factor": soak_entry.get("throttle_factor"),
            "peak_memory_bytes": mem_entry.get("peak_bytes", mem_entry.get("bytes")),
            "peak_memory_method": mem_entry.get("method"),
            "disk_bytes": mem_entry.get("disk_bytes"),
            "incremental_watts": first.incremental_watts if first else None,
            "idle_watts": first.idle_watts if first else None,
            "watts_label": first.watts_label if first else "not measured",
        }
    combined_raw = memory.get("combined")
    combined: dict[str, Any] = combined_raw if isinstance(combined_raw, dict) else {}
    ours = combined.get("ours_bytes", combined.get("bytes"))
    return {
        "models": per_model,
        "combined_memory_bytes": ours,
        "combined": {
            "ours_bytes": ours,
            "models_bytes": combined.get("models_bytes") or {},
            "router_bytes": combined.get("router_bytes"),
            "system_used_bytes": combined.get("system_used_bytes"),
            "system_used_before_bytes": combined.get("system_used_before_bytes"),
            "method": combined.get("method"),
        },
    }


def _cheapest_cloud(
    calib_c: list[Candidate], test_c: list[Candidate], bar: float
) -> tuple[Candidate | None, float | None]:
    """Cheapest cloud candidate meeting the bar on calib; its cost measured on the other split."""
    eligible = [c for c in calib_c if c.kind == "cloud" and quality.meets_bar(c.primary, bar)]
    if not eligible:
        return None, None
    test_by_id = {c.model_id: c for c in test_c}

    def _cost(c: Candidate) -> float:
        t = test_by_id.get(c.model_id)
        return t.usd_per_task if t is not None and not math.isnan(t.usd_per_task) else math.inf

    best = min(eligible, key=lambda c: (_cost(c), c.model_id))
    cost = _cost(best)
    return best, (cost if math.isfinite(cost) else None)


def _build_break_even(
    tasks: list[str],
    calib: dict[str, list[Candidate]],
    test: dict[str, list[Candidate]],
    route_cfg: RouteConfig,
    bar_by_task: dict[str, float],
    local_costs_map: dict[tuple[str, str], LocalCost],
) -> tuple[dict[str, dict[str, Any]], dict[str, BreakEven]]:
    rows: dict[str, dict[str, Any]] = {}
    by_task: dict[str, BreakEven] = {}
    for task in tasks:
        calib_c, test_c = calib.get(task, []), test.get(task, [])
        bar = bar_by_task[task]
        v_t = route_cfg.monthly_volume(task)

        local_llm_calib = [c for c in quality.local_llm_candidates(calib_c) if not math.isnan(c.primary)]
        best_local = max(local_llm_calib, key=lambda c: c.primary, default=None)
        local_meets = quality.meets_bar(best_local.primary, bar) if best_local else False

        cheapest_cloud, cloud_cost = _cheapest_cloud(calib_c, test_c, bar)
        cloud_meets = cheapest_cloud is not None

        lc = local_costs_map.get((best_local.model_id, task)) if best_local else None
        fixed = lc.fixed_usd_per_month if lc else 0.0
        energy = lc.energy_usd_per_task if lc else math.inf
        capacity = lc.capacity_per_month if lc else 0.0

        be = costmodel.break_even(
            fixed_usd_per_month=fixed,
            cloud_usd_per_task=cloud_cost if cloud_meets else None,
            energy_usd_per_task=energy,
            capacity_per_month=capacity,
            monthly_volume=v_t,
            local_meets_bar=local_meets,
            cloud_meets_bar=cloud_meets,
        )
        by_task[task] = be
        rows[task] = {
            "v_t": v_t,
            "best_local_model": best_local.model_id if best_local else None,
            "cheapest_cloud_model": cheapest_cloud.model_id if cheapest_cloud else None,
            "cheapest_cloud_usd_per_task": cloud_cost if cloud_meets else None,
            "energy_usd_per_task": lc.energy_usd_per_task if lc else None,
            "fixed_usd_per_month": lc.fixed_usd_per_month if lc else None,
            "break_even_volume": be.volume,
            "capacity_per_month": capacity if lc else None,
            "machines_needed": be.machines_needed,
            "verdict": be.verdict,
            "measured": lc is not None,
        }
    return rows, by_task


def _build_sensitivity(
    headline_task: str,
    be_row: dict[str, Any],
    met_bar_tasks: list[str],
    local_costs_map: dict[tuple[str, str], LocalCost],
    hw: HardwareConfig | None,
) -> list[SensitivityCell]:
    if hw is None or not be_row:
        return []
    model_id = be_row.get("best_local_model")
    cloud_cost = be_row.get("cheapest_cloud_usd_per_task")
    if not model_id or cloud_cost is None:
        return []
    lc = local_costs_map.get((str(model_id), headline_task))
    if lc is None:
        return []
    return costmodel.sensitivity_grid(
        purchase_price_usd=hw.purchase_price_usd,
        allocation=hw.allocation,
        idle_watts=lc.idle_watts,
        electricity_usd_per_kwh=hw.electricity_usd_per_kwh,
        ops_usd_per_month=hw.ops_usd_per_month,
        cloud_usd_per_task=float(cloud_cost),
        energy_usd_per_task=lc.energy_usd_per_task,
        capacity_per_month=lc.capacity_per_month,
        monthly_volume=float(be_row.get("v_t") or 0.0),
        local_meets_bar=headline_task in met_bar_tasks,
        cloud_meets_bar=True,
    )


def _sensitivity_note(
    headline_task: str,
    be_row: dict[str, Any],
    local_costs_map: dict[tuple[str, str], LocalCost],
    hw: HardwareConfig | None,
) -> str:
    """Why the sensitivity grid is empty, for the report to say instead of "not measured"."""
    if hw is None:
        return "not measured: this run has no hardware configuration, so the fixed cost cannot be varied"
    if not be_row:
        return f"not measured: no break-even row for the headline task, {headline_task}"
    if be_row.get("cheapest_cloud_usd_per_task") is None or not be_row.get("cheapest_cloud_model"):
        return f"not applicable: no cloud model meets the {headline_task} bar, so there is no break-even volume to vary"
    model_id = be_row.get("best_local_model")
    if not model_id or local_costs_map.get((str(model_id), headline_task)) is None:
        return f"not measured: no local throughput was measured for the headline task, {headline_task}"
    return f"not measured for the headline task, {headline_task}"


def _as_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if dataclasses.is_dataclass(row) and not isinstance(row, type):
        return dataclasses.asdict(row)
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        result: dict[str, Any] = dict(to_dict())
        return result
    return {"value": row}


ROUTER_DEPLOYED_LABEL = "router (gates, route.yaml constraints)"
ROUTER_UNCONSTRAINED_LABEL = "router (gates, no constraints)"


def _refused_tasks(row: dict[str, Any] | None) -> list[str]:
    per_task = (row or {}).get("per_task") or {}
    return sorted(t for t, v in per_task.items() if v.get("unservable"))


def _extract_saving(rows: list[dict[str, Any]]) -> tuple[float | None, str | None, str | None, list[str]]:
    """Headline saving of the deployed router (gates + route.yaml constraints), in percent.

    It is measured against the cheapest single cloud model meeting every bar, or against all-frontier when no such
    model exists. When the deployed router refuses (503) any task, its cost covers fewer tasks than the baselines it
    is compared with, so the saving comes from the gates-without-constraints row instead. Returns
    ``(saving_pct, saving_vs, scope, refused_tasks)``; ``scope`` names the substitute row, ``None`` for the deployed
    one, and ``refused_tasks`` are the tasks the deployed router refuses.
    """
    deployed = next((r for r in rows if r.get("label") == ROUTER_DEPLOYED_LABEL), None)
    if deployed is None:
        return None, None, None, []
    refused = _refused_tasks(deployed)
    router, scope = deployed, None
    if refused:
        unconstrained = next((r for r in rows if r.get("label") == ROUTER_UNCONSTRAINED_LABEL), None)
        if unconstrained is not None:
            router, scope = unconstrained, "gates, no route.yaml constraints"
    vs_cheapest = router.get("saving_vs_cheapest_cloud_pct")
    if isinstance(vs_cheapest, int | float) and math.isfinite(vs_cheapest):
        return float(vs_cheapest), "the cheapest single cloud model meeting every bar", scope, refused
    vs_frontier = router.get("saving_vs_all_frontier_pct")
    if isinstance(vs_frontier, int | float) and math.isfinite(vs_frontier):
        vs = "all traffic to the frontier model (no single cloud model meets every bar)"
        return float(vs_frontier), vs, scope, refused
    return None, None, scope, refused


def _gather_router_facts(
    run: RunDir,
    specs: dict[str, TaskSpec],
    route_cfg: RouteConfig,
    calib: dict[str, list[Candidate]],
    bar_by_task: dict[str, float],
    label: str,
) -> RouterFacts:
    from local_enough.route import planner as route_planner
    from local_enough.route import simulate as route_simulate

    plan = route_planner.build_plan(run, specs, route_cfg)
    # `hybrid` follows the plan the router actually serves: a local primary (local LLM or baseline) whose gated chain,
    # escalating to cloud on failed gates, meets the bar on calib with at most 20% escalation.
    calib_sim = route_simulate.simulate(run, plan, specs, route_cfg, split="calib")
    hybrid_by_task: dict[str, bool] = {}
    hybrid_details: dict[str, dict[str, Any]] = {}
    for task, res in calib_sim.tasks.items():
        tp = plan.tasks[task]
        local_primary = tp.primary is not None and tp.primary.kind in ("local", "baseline")
        cloud_fallbacks = [r.model_id for r in tp.fallbacks if r.kind == "cloud"]
        hybrid_by_task[task] = bool(
            local_primary and cloud_fallbacks and res.meets_bar and 0 < res.escalation_rate <= 0.20
        )
        hybrid_details[task] = {
            "escalation_rate": res.escalation_rate,
            "escalates_to": cloud_fallbacks[0] if cloud_fallbacks else None,
        }
    mixed_rows = [row.to_dict() for row in route_simulate.mixed_table(run, plan, specs, route_cfg)]
    saving_pct, saving_vs, saving_scope, refused = _extract_saving(mixed_rows)
    return RouterFacts(
        available=True,
        plan_text=route_planner.print_plan(plan),
        mixed_rows=mixed_rows,
        hybrid_by_task=hybrid_by_task,
        saving_pct=saving_pct,
        saving_vs=saving_vs,
        saving_scope=saving_scope,
        refused_tasks=refused,
        hybrid_details=hybrid_details,
        source_cmd=f"local-enough route --simulate --run {label}",
    )


def _local_alone_reason(be: BreakEven | None, local_meets: bool, gap: float) -> str | None:
    """Why local on its own did not win the task, for the ``hybrid`` verdict's detail."""
    if not local_meets:
        gap_text = "" if math.isnan(gap) else f" (gap to bar: -{fmt.fmt_pct(gap)})"
        return f"local alone is below the bar{gap_text}"
    if be is None:
        return None
    volume = fmt.fmt_number(be.volume) if be.volume is not None else None
    if be.verdict == costmodel.VERDICT_BREAK_EVEN and volume:
        return f"local alone is cheaper only above {volume} tasks/month"
    if be.verdict == costmodel.VERDICT_INFEASIBLE:
        where = f"at {volume} tasks/month, " if volume else ""
        return f"local alone would break even {where}more than this machine can serve"
    if be.verdict == costmodel.VERDICT_NEVER:
        return "local alone is never cheaper than the cheapest cloud model meeting the bar"
    if be.verdict == costmodel.VERDICT_NO_CLOUD_BAR:
        return "no cloud model meets the bar on its own, so local alone has nothing to break even against"
    return None


def _build_verdicts(
    tasks: list[str],
    calib: dict[str, list[Candidate]],
    route_cfg: RouteConfig,
    bar_by_task: dict[str, float],
    break_even_by_task: dict[str, BreakEven],
    router: RouterFacts,
) -> list[TaskVerdict]:
    out: list[TaskVerdict] = []
    for task in tasks:
        calib_c = calib.get(task, [])
        bar = bar_by_task[task]
        broad = [c for c in quality.local_candidates(calib_c) if not math.isnan(c.primary)]
        best_broad = max(broad, key=lambda c: c.primary, default=None)
        local_meets = quality.meets_bar(best_broad.primary, bar) if best_broad else False
        gap = quality.gap_to_bar(best_broad.primary, bar) if best_broad else math.nan
        v_t = route_cfg.monthly_volume(task)
        be = break_even_by_task.get(task)

        free_baseline: str | None = None
        if best_broad is not None and best_broad.kind == "baseline" and local_meets:
            breaks_even_within, be_volume = True, None
            free_baseline = quality.baseline_name(best_broad.model_id)
        else:
            reached_break_even = be is not None and be.verdict == costmodel.VERDICT_BREAK_EVEN and be.volume is not None
            breaks_even_within = bool(
                reached_break_even and be is not None and be.volume is not None and v_t >= be.volume
            )
            be_volume = be.volume if be is not None else None

        hybrid = router.hybrid_details.get(task) or {}
        out.append(
            quality.task_verdict(
                task,
                is_local_only=route_cfg.is_local_only(task),
                local_meets_bar=local_meets,
                gap=gap,
                breaks_even_within_volume=breaks_even_within,
                break_even_volume=be_volume,
                v_t=v_t,
                hybrid_eligible=router.hybrid_by_task.get(task, False),
                free_baseline=free_baseline,
                local_alone_reason=_local_alone_reason(be, local_meets, gap),
                hybrid_escalation_rate=hybrid.get("escalation_rate"),
                hybrid_escalates_to=hybrid.get("escalates_to"),
            )
        )
    return out


def _build_judge(run: RunDir) -> dict[str, Any]:
    calibration = run.read_json("judge_calibration.json")
    if not calibration:
        return {}
    summary = judge_stats.judge_summary(run)
    rows = [
        {
            "model_id": model_id,
            "n": s["n"],
            "raw_pass_rate": s["raw_pass_rate"],
            "corrected_pass_rate": s["corrected_pass_rate"],
            "ci": tuple(s["ci"]),
            "key_token_agreement": s["key_token_agreement"],
        }
        for (model_id, split), s in sorted(summary.items())
        if split == "test"
    ]
    selected = calibration.get("selected")
    holdout = calibration.get("holdout") or {}
    calib_stats = ((calibration.get("candidates") or {}).get(selected) or {}).get("calib") or {}
    return {
        "judge_model": selected,
        "target_balanced_accuracy": calibration.get("target_balanced_accuracy"),
        "meets_target": calibration.get("meets_target"),
        "judge_calib_n": calibration.get("judge_calib_n"),
        "holdout_n": holdout.get("n"),
        "tpr": holdout.get("tpr"),
        "tnr": holdout.get("tnr"),
        "balanced_accuracy": holdout.get("balanced_accuracy"),
        "kappa": holdout.get("kappa"),
        "calib_balanced_accuracy": calib_stats.get("balanced_accuracy"),
        "rows": rows,
    }


def _build_data_table(specs: dict[str, TaskSpec], tasks: list[str]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        spec = specs[task]
        card = spec.card
        rows.append(
            {
                "task": task,
                "source": card.source if card else None,
                "licence": card.licence if card else None,
                "metric": card.metric if card else PRIMARY_METRIC_NAME.get(spec.kind),
                "distinct_templates": card.distinct_templates if card else None,
            }
        )
    return rows


def _build_downloads(run: RunDir) -> dict[str, Any]:
    downloads = run.read_json("downloads.json", {}) or {}
    rows = [
        {
            "model_id": mid,
            "repo": d.get("repo"),
            "revision_sha": d.get("revision_sha"),
            "bytes": d.get("bytes"),
            "licence": d.get("licence"),
        }
        for mid, d in sorted(downloads.items())
    ]
    total_bytes = sum(d.get("bytes") or 0 for d in downloads.values())
    return {"rows": rows, "total_bytes": total_bytes}


def _build_spend(cfg: dict[str, Any], total_usd: float, by_command: dict[str, float]) -> dict[str, Any]:
    return {
        "total_usd": total_usd,
        "by_command": by_command,
        "budget_usd": cfg.get("budget_usd"),
        "budget_warn_usd": cfg.get("budget_warn_usd"),
    }


def build_report_data(
    run_dir: str | Path | RunDir, *, specs: dict[str, TaskSpec] | None = None, route_cfg: RouteConfig | None = None
) -> ReportData:
    """Read ``run_dir`` and every shared helper it needs into one :class:`ReportData`."""
    run = run_dir if isinstance(run_dir, RunDir) else RunDir(run_dir)
    resolved_specs = specs or evaluate.task_specs_for_run(run)
    resolved_route = route_cfg or _default_route_cfg(run)
    tasks = sorted(resolved_specs)
    label = run_label(run)

    calib = candidate_table(run, resolved_specs, resolved_route, "calib")
    test = candidate_table(run, resolved_specs, resolved_route, "test")
    bar_by_task = {t: quality.bar_value(resolved_route, t, calib.get(t, [])) for t in tasks}
    local_costs_map = costing.local_costs(run, resolved_route)
    hw = costing.hardware_from_run(run)

    setup = _build_setup(run, resolved_specs, tasks)
    results = _build_results(tasks, resolved_specs, calib, test, bar_by_task, local_costs_map)
    local_perf = _build_local_perf(run, local_costs_map)
    break_even_rows, break_even_by_task = _build_break_even(
        tasks, calib, test, resolved_route, bar_by_task, local_costs_map
    )

    met_bar_tasks = [
        t
        for t in tasks
        if quality.meets_bar(quality.best_score(quality.local_candidates(calib.get(t, []))), bar_by_task[t])
    ]
    headline_task_name, _ = quality.headline_task(met_bar_tasks, resolved_route.workload_mix)
    sensitivity = _build_sensitivity(
        headline_task_name, break_even_rows.get(headline_task_name, {}), met_bar_tasks, local_costs_map, hw
    )

    router = _gather_router_facts(run, resolved_specs, resolved_route, calib, bar_by_task, label)
    verdicts = _build_verdicts(tasks, calib, resolved_route, bar_by_task, break_even_by_task, router)
    live_check = run.read_json("live_check.json", {}) or {}
    judge_table = _build_judge(run)
    data_table = _build_data_table(resolved_specs, tasks)
    downloads = _build_downloads(run)
    cfg_dict = run.read_json("config.json", {}) or {}
    spend = _build_spend(cfg_dict, setup["total_api_spend_usd"], setup["spend_by_command"])
    v_t_by_task = {t: resolved_route.monthly_volume(t) for t in tasks}
    local_winners: dict[str, quality.LocalWinner] = {}
    for t in met_bar_tasks:
        scored = [c for c in quality.local_candidates(calib.get(t, [])) if not math.isnan(c.primary)]
        if scored:
            top = max(scored, key=lambda c: (c.primary, c.model_id))
            local_winners[t] = quality.LocalWinner(top.model_id, top.kind)
    refusals: list[quality.Refusal] = []
    for t in router.refused_tasks:
        local_scores = [c.primary for c in quality.local_candidates(calib.get(t, [])) if not math.isnan(c.primary)]
        refusals.append(
            quality.Refusal(
                t,
                PRIMARY_METRIC_NAME.get(resolved_specs[t].kind, "primary"),
                max(local_scores, default=math.nan),
                bar_by_task[t],
            )
        )
    headline_str = quality.headline_text(
        met_bar_tasks=met_bar_tasks,
        total_tasks=len(tasks),
        workload_mix=resolved_route.workload_mix,
        break_even_by_task=break_even_by_task,
        v_t_by_task=v_t_by_task,
        router_available=router.available,
        router_saving_pct=router.saving_pct,
        router_saving_vs=router.saving_vs,
        router_saving_scope=router.saving_scope,
        local_winners=local_winners,
        refusals=refusals,
    )

    return ReportData(
        run=run,
        run_label=label,
        date=setup.get("run_date") or "unknown",
        specs=resolved_specs,
        tasks=tasks,
        route_cfg=resolved_route,
        setup=setup,
        results=results,
        local_perf=local_perf,
        break_even={
            "rows": break_even_rows,
            "sensitivity": sensitivity,
            "sensitivity_note": _sensitivity_note(
                headline_task_name, break_even_rows.get(headline_task_name, {}), local_costs_map, hw
            ),
            "headline_task": headline_task_name,
            "headline_task_met_bar": headline_task_name in met_bar_tasks,
        },
        verdicts=verdicts,
        router=router,
        live_check=live_check,
        judge=judge_table,
        data_table=data_table,
        downloads=downloads,
        spend=spend,
        headline=headline_str,
    )
