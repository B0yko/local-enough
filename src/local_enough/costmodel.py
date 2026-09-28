"""Pure cost-model functions: amortised hardware, energy, break-even and the shared-machine scenario.

Every function here is deterministic and side-effect free; see ``docs/cost-model.md`` for the
formulas in prose and a worked example. Two scenarios are modelled, always labelled:

- **Dedicated**: a machine dedicated to one task at its own monthly volume ``V_t``. Used for the
  per-task tables, frontier charts and break-even verdicts (:func:`break_even`).
- **Shared**: one machine's fixed cost is shared across every task it serves locally, used by the
  router's mixed-workload costing (:func:`shared_machine`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DAYS_PER_MONTH = 30
HOURS_PER_DAY = 24
JOULES_PER_KWH = 3.6e6

BreakEvenVerdict = str
VERDICT_BREAK_EVEN: BreakEvenVerdict = "break-even"
VERDICT_NEVER: BreakEvenVerdict = "never"
VERDICT_INFEASIBLE: BreakEvenVerdict = "infeasible on this hardware"
VERDICT_LOCAL_BELOW_BAR: BreakEvenVerdict = "n/a — local below bar"
VERDICT_NO_CLOUD_BAR: BreakEvenVerdict = "n/a — no cloud model meets the bar"


def fixed_usd_per_month(
    *,
    purchase_price_usd: float,
    allocation: float,
    lifetime_years: float,
    idle_watts: float,
    electricity_usd_per_kwh: float,
    ops_usd_per_month: float = 0.0,
) -> float:
    """Amortised purchase price plus idle-power draw plus ops, scaled by the allocated share."""
    amortised = purchase_price_usd * allocation / (lifetime_years * 12)
    idle_energy = idle_watts * HOURS_PER_DAY * DAYS_PER_MONTH / 1000 * electricity_usd_per_kwh * allocation
    return amortised + idle_energy + ops_usd_per_month


def sustained_tasks_per_hour(pass_b_tasks_per_hour: float, throttle_factor: float) -> float:
    """Pass B (concurrency 4) throughput scaled by the soak throttle factor (last5 / first5)."""
    return pass_b_tasks_per_hour * throttle_factor


def energy_usd_per_task(
    *, incremental_watts: float, sustained_tasks_per_hour: float, electricity_usd_per_kwh: float
) -> float:
    """USD of electricity per task, from incremental watts while serving and sustained throughput."""
    if sustained_tasks_per_hour <= 0:
        return math.inf
    tasks_per_second = sustained_tasks_per_hour / 3600
    joules_per_task = incremental_watts / tasks_per_second
    kwh_per_task = joules_per_task / JOULES_PER_KWH
    return kwh_per_task * electricity_usd_per_kwh


def local_usd_per_task(*, fixed_usd_per_month: float, monthly_volume: float, energy_usd_per_task: float) -> float:
    """Dedicated-scenario USD per task: amortised fixed cost spread over V_t, plus energy."""
    if monthly_volume <= 0:
        return math.inf
    return fixed_usd_per_month / monthly_volume + energy_usd_per_task


def capacity_per_month(
    sustained_tasks_per_hour: float, busy_hours_per_day: float, days_per_month: float = DAYS_PER_MONTH
) -> float:
    """How many tasks per month this machine can sustain at the given daily duty cycle."""
    return sustained_tasks_per_hour * busy_hours_per_day * days_per_month


def machines_needed(monthly_volume: float, capacity_per_month: float) -> int | None:
    """``ceil(V_t / capacity)``; ``None`` when capacity is zero or unknown."""
    if capacity_per_month <= 0:
        return None
    return math.ceil(monthly_volume / capacity_per_month)


@dataclass(frozen=True)
class BreakEven:
    """The break-even monthly volume, its verdict, and machines needed to serve ``monthly_volume``."""

    volume: float | None
    verdict: BreakEvenVerdict
    machines_needed: int | None


def break_even(
    *,
    fixed_usd_per_month: float,
    cloud_usd_per_task: float | None,
    energy_usd_per_task: float,
    capacity_per_month: float,
    monthly_volume: float,
    local_meets_bar: bool,
    cloud_meets_bar: bool,
) -> BreakEven:
    """The dedicated-scenario break-even volume and verdict for one task.

    ``cloud_usd_per_task`` is the cheapest cloud model that meets the bar (``None``/ignored when
    ``cloud_meets_bar`` is False). ``monthly_volume`` is the task's actual expected volume, used
    only to report ``machines_needed`` alongside the verdict.
    """
    needed = machines_needed(monthly_volume, capacity_per_month)

    if not cloud_meets_bar or cloud_usd_per_task is None:
        return BreakEven(None, VERDICT_NO_CLOUD_BAR, needed)
    if not local_meets_bar:
        return BreakEven(None, VERDICT_LOCAL_BELOW_BAR, needed)

    denominator = cloud_usd_per_task - energy_usd_per_task
    if denominator <= 0:
        return BreakEven(None, VERDICT_NEVER, needed)

    volume = fixed_usd_per_month / denominator
    if volume > capacity_per_month:
        return BreakEven(volume, VERDICT_INFEASIBLE, needed)
    return BreakEven(volume, VERDICT_BREAK_EVEN, needed)


@dataclass(frozen=True)
class SharedTask:
    """One task's inputs to the shared-machine scenario."""

    monthly_volume: float
    capacity_per_month: float
    energy_usd_per_task: float


@dataclass(frozen=True)
class SharedMachineResult:
    utilisation: float
    machines: int
    shared_fixed_usd_per_task: float
    local_usd_per_task: dict[str, float]


def shared_machine(tasks: dict[str, SharedTask], fixed_usd_per_month: float) -> SharedMachineResult:
    """Cost of serving several tasks locally from one shared machine (or a fleet of identical ones).

    Utilisation is ``sum(V_t / capacity_t)`` over the locally served tasks; above 1 it charges
    ``ceil`` of that sum in machines. The fleet's total fixed cost is then spread across all
    locally served volume, and each task's energy cost is added on top.
    """
    if not tasks:
        return SharedMachineResult(utilisation=0.0, machines=0, shared_fixed_usd_per_task=0.0, local_usd_per_task={})

    utilisation = sum(t.monthly_volume / t.capacity_per_month for t in tasks.values() if t.capacity_per_month > 0)
    machines = max(1, math.ceil(utilisation)) if utilisation > 0 else 1
    total_fixed = fixed_usd_per_month * machines
    total_volume = sum(t.monthly_volume for t in tasks.values())
    shared_fixed = total_fixed / total_volume if total_volume > 0 else 0.0

    per_task = {name: shared_fixed + t.energy_usd_per_task for name, t in tasks.items()}
    return SharedMachineResult(
        utilisation=utilisation, machines=machines, shared_fixed_usd_per_task=shared_fixed, local_usd_per_task=per_task
    )


@dataclass(frozen=True)
class SensitivityCell:
    lifetime_years: float
    price_multiplier: float
    fixed_usd_per_month: float
    break_even: BreakEven


def sensitivity_grid(
    *,
    purchase_price_usd: float,
    allocation: float,
    idle_watts: float,
    electricity_usd_per_kwh: float,
    ops_usd_per_month: float,
    cloud_usd_per_task: float | None,
    energy_usd_per_task: float,
    capacity_per_month: float,
    monthly_volume: float,
    local_meets_bar: bool,
    cloud_meets_bar: bool,
    lifetimes_years: tuple[float, ...] = (2.0, 3.0, 4.0),
    price_multipliers: tuple[float, ...] = (0.75, 1.0, 1.25),
) -> list[SensitivityCell]:
    """Break-even at each (lifetime, purchase-price multiplier) combination, for the headline task."""
    cells: list[SensitivityCell] = []
    for lifetime in lifetimes_years:
        for multiplier in price_multipliers:
            fixed = fixed_usd_per_month(
                purchase_price_usd=purchase_price_usd * multiplier,
                allocation=allocation,
                lifetime_years=lifetime,
                idle_watts=idle_watts,
                electricity_usd_per_kwh=electricity_usd_per_kwh,
                ops_usd_per_month=ops_usd_per_month,
            )
            cell_break_even = break_even(
                fixed_usd_per_month=fixed,
                cloud_usd_per_task=cloud_usd_per_task,
                energy_usd_per_task=energy_usd_per_task,
                capacity_per_month=capacity_per_month,
                monthly_volume=monthly_volume,
                local_meets_bar=local_meets_bar,
                cloud_meets_bar=cloud_meets_bar,
            )
            cells.append(
                SensitivityCell(
                    lifetime_years=lifetime,
                    price_multiplier=multiplier,
                    fixed_usd_per_month=fixed,
                    break_even=cell_break_even,
                )
            )
    return cells
