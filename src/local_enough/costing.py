"""Cost inputs derived from a run directory: local cost per (model, task) and measured cloud cost per task.

Local numbers use the dedicated-machine scenario (docs/cost-model.md): the machine serves task t at its monthly
volume V_t. The router's shared-machine costing lives in ``route.simulate`` and reuses ``LocalCost`` inputs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from local_enough import costmodel
from local_enough.bench.rundir import RunDir
from local_enough.config import HardwareConfig, RouteConfig

WattsLabel = Literal["measured", "configured"]


@dataclass(frozen=True)
class LocalCost:
    model_id: str
    task: str
    tasks_per_hour_c1: float | None
    tasks_per_hour_c4: float | None
    throttle_factor: float
    throttle_source: str
    sustained_tasks_per_hour: float | None
    throughput_source: str
    incremental_watts: float
    idle_watts: float
    watts_label: WattsLabel
    fixed_usd_per_month: float
    energy_usd_per_task: float
    monthly_volume: float
    usd_per_task: float
    capacity_per_month: float


def run_models(run: RunDir) -> dict[str, dict[str, Any]]:
    """Model entries from the run's sanitised config, keyed by id."""
    cfg = run.read_json("config.json", {}) or {}
    return {m["id"]: m for m in cfg.get("models", [])}


def provider_of(model: dict[str, Any]) -> str:
    """Provider class used for fallback diversity: vendor prefix for cloud, 'local' or 'baseline' otherwise."""
    kind = model.get("kind")
    if kind == "cloud":
        name = str(model.get("model") or model.get("id"))
        return name.split("/", 1)[0] if "/" in name else name
    return "local" if kind == "local" else "baseline"


def hardware_from_run(run: RunDir) -> HardwareConfig | None:
    cfg = run.read_json("config.json", {}) or {}
    hw = cfg.get("hardware")
    return HardwareConfig.model_validate(hw) if hw else None


def power_for(run: RunDir, model_id: str, hw: HardwareConfig) -> tuple[float, float, WattsLabel]:
    """(incremental W, idle W, label): power-probe results when mode is measured and present, else configured."""
    probe = (run.read_json("power.json", {}) or {}).get(model_id) or {}
    if hw.power.mode == "measured" and "incremental_watts" in probe and probe.get("mode") == "measured":
        return float(probe["incremental_watts"]), float(probe["idle_watts"]), "measured"
    return hw.power.incremental_watts, hw.power.idle_watts, "configured"


def _tph(latency: dict[str, Any], model_id: str, task: str, split: str, pass_: str) -> float | None:
    node = latency.get(model_id, {}).get(task, {}).get(split, {}).get(pass_)
    if not node:
        return None
    value = node.get("tasks_per_hour")
    return float(value) if value else None


def local_costs(
    run: RunDir, route_cfg: RouteConfig, hw: HardwareConfig | None = None
) -> dict[tuple[str, str], LocalCost]:
    """Dedicated-scenario cost for every local model and task measured in the run."""
    hw = hw or hardware_from_run(run)
    if hw is None:
        return {}
    latency = run.read_json("latency.json", {}) or {}
    soak = run.read_json("soak.json", {}) or {}
    out: dict[tuple[str, str], LocalCost] = {}
    for model_id, model in run_models(run).items():
        if model.get("kind") != "local":
            continue
        soak_entry = soak.get(model_id) or {}
        throttle = soak_entry.get("throttle_factor")
        if throttle:
            # A last window faster than the first is noise, not negative throttling: never scale throughput up.
            throttle_factor = min(1.0, float(throttle))
            throttle_source = "soak (last 5 / first 5 full minutes)" + (", capped at 1.0" if throttle > 1 else "")
        else:
            throttle_factor, throttle_source = 1.0, "no soak (assumed 1.0)"
        incremental, idle, label = power_for(run, model_id, hw)
        fixed = costmodel.fixed_usd_per_month(
            purchase_price_usd=hw.purchase_price_usd,
            allocation=hw.allocation,
            lifetime_years=hw.lifetime_years,
            idle_watts=idle,
            electricity_usd_per_kwh=hw.electricity_usd_per_kwh,
            ops_usd_per_month=hw.ops_usd_per_month,
        )
        for task in latency.get(model_id, {}):
            c1 = _tph(latency, model_id, task, "test", "A") or _tph(latency, model_id, task, "calib", "A")
            c4 = _tph(latency, model_id, task, "test", "B")
            if c4:
                base, source = c4, "pass B (c=4) x throttle"
            elif c1:
                base, source = c1, "pass A (c=1, no pass B) x throttle"
            else:
                base, source = 0.0, "not measured"
            sustained = costmodel.sustained_tasks_per_hour(base, throttle_factor) if base else None
            energy = (
                costmodel.energy_usd_per_task(
                    incremental_watts=incremental,
                    sustained_tasks_per_hour=sustained,
                    electricity_usd_per_kwh=hw.electricity_usd_per_kwh,
                )
                if sustained
                else math.inf
            )
            volume = route_cfg.monthly_volume(task)
            out[(model_id, task)] = LocalCost(
                model_id=model_id,
                task=task,
                tasks_per_hour_c1=c1,
                tasks_per_hour_c4=c4,
                throttle_factor=throttle_factor,
                throttle_source=throttle_source,
                sustained_tasks_per_hour=sustained,
                throughput_source=source,
                incremental_watts=incremental,
                idle_watts=idle,
                watts_label=label,
                fixed_usd_per_month=fixed,
                energy_usd_per_task=energy,
                monthly_volume=volume,
                usd_per_task=costmodel.local_usd_per_task(
                    fixed_usd_per_month=fixed, monthly_volume=volume, energy_usd_per_task=energy
                ),
                capacity_per_month=costmodel.capacity_per_month(sustained or 0.0, hw.busy_hours_per_day),
            )
    return out


def cloud_usd_per_task(run: RunDir, split: str = "calib") -> dict[tuple[str, str], float]:
    """Mean recorded cost per Pass A call of each cloud model on each task (failed calls included)."""
    models = run_models(run)
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rec in run.predictions("A"):
        if rec.get("split") != split or models.get(rec["model_id"], {}).get("kind") != "cloud":
            continue
        totals[(rec["model_id"], rec["task"])].append(float(rec.get("cost_usd") or 0.0))
    return {key: sum(values) / len(values) for key, values in totals.items() if values}


def usd_per_task(
    run: RunDir, route_cfg: RouteConfig, split: str = "calib", hw: HardwareConfig | None = None
) -> dict[tuple[str, str], float]:
    """USD per task for every candidate: local dedicated at V_t, cloud measured on ``split``, baselines 0."""
    out: dict[tuple[str, str], float] = {k: v.usd_per_task for k, v in local_costs(run, route_cfg, hw).items()}
    out.update(cloud_usd_per_task(run, split))
    models = run_models(run)
    for rec in run.predictions("A"):
        if models.get(rec["model_id"], {}).get("kind") == "baseline":
            out[(rec["model_id"], rec["task"])] = 0.0
    return out
