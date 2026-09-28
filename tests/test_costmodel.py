from __future__ import annotations

import math
import random

import pytest

from local_enough.costmodel import (
    VERDICT_BREAK_EVEN,
    VERDICT_INFEASIBLE,
    VERDICT_LOCAL_BELOW_BAR,
    VERDICT_NEVER,
    VERDICT_NO_CLOUD_BAR,
    SharedTask,
    break_even,
    capacity_per_month,
    energy_usd_per_task,
    fixed_usd_per_month,
    local_usd_per_task,
    machines_needed,
    sensitivity_grid,
    shared_machine,
    sustained_tasks_per_hour,
)


def test_fixed_usd_per_month_worked_example() -> None:
    fixed = fixed_usd_per_month(
        purchase_price_usd=1200, allocation=1.0, lifetime_years=3, idle_watts=5, electricity_usd_per_kwh=0.30
    )
    assert fixed == pytest.approx(34.413333, rel=1e-6)


def test_fixed_usd_per_month_scales_with_allocation() -> None:
    full = fixed_usd_per_month(
        purchase_price_usd=1200, allocation=1.0, lifetime_years=3, idle_watts=5, electricity_usd_per_kwh=0.30
    )
    half = fixed_usd_per_month(
        purchase_price_usd=1200, allocation=0.5, lifetime_years=3, idle_watts=5, electricity_usd_per_kwh=0.30
    )
    assert half == pytest.approx(full / 2, rel=1e-9)


def test_sustained_tasks_per_hour_applies_throttle() -> None:
    assert sustained_tasks_per_hour(500, 0.9) == pytest.approx(450.0)


def test_energy_usd_per_task_worked_example() -> None:
    result = energy_usd_per_task(incremental_watts=20, sustained_tasks_per_hour=450.0, electricity_usd_per_kwh=0.30)
    assert result == pytest.approx(1.3333333e-05, rel=1e-6)


def test_energy_usd_per_task_zero_throughput_is_infinite() -> None:
    result = energy_usd_per_task(incremental_watts=20, sustained_tasks_per_hour=0, electricity_usd_per_kwh=0.30)
    assert result == math.inf


def test_local_usd_per_task_worked_example() -> None:
    result = local_usd_per_task(fixed_usd_per_month=34.413333, monthly_volume=15000, energy_usd_per_task=1.3333e-05)
    assert result * 1000 == pytest.approx(2.3076, rel=1e-3)


def test_local_usd_per_task_zero_volume_is_infinite() -> None:
    assert local_usd_per_task(fixed_usd_per_month=10, monthly_volume=0, energy_usd_per_task=0.0) == math.inf


def test_capacity_per_month() -> None:
    assert capacity_per_month(450.0, busy_hours_per_day=8) == pytest.approx(108000.0)


def test_machines_needed_rounds_up() -> None:
    assert machines_needed(15000, 108000) == 1
    assert machines_needed(120000, 108000) == 2
    assert machines_needed(100, 0) is None


def test_break_even_worked_example() -> None:
    result = break_even(
        fixed_usd_per_month=34.413333,
        cloud_usd_per_task=0.003,
        energy_usd_per_task=1.3333e-05,
        capacity_per_month=108000.0,
        monthly_volume=15000.0,
        local_meets_bar=True,
        cloud_meets_bar=True,
    )
    assert result.verdict == VERDICT_BREAK_EVEN
    assert result.volume == pytest.approx(11522.3, rel=1e-3)
    assert result.machines_needed == 1


def test_break_even_never_when_energy_exceeds_cloud_cost() -> None:
    result = break_even(
        fixed_usd_per_month=34.41,
        cloud_usd_per_task=0.001,
        energy_usd_per_task=0.002,
        capacity_per_month=108000.0,
        monthly_volume=15000.0,
        local_meets_bar=True,
        cloud_meets_bar=True,
    )
    assert result.verdict == VERDICT_NEVER
    assert result.volume is None


def test_break_even_infeasible_when_volume_exceeds_capacity() -> None:
    result = break_even(
        fixed_usd_per_month=100000.0,
        cloud_usd_per_task=0.003,
        energy_usd_per_task=0.0,
        capacity_per_month=1000.0,
        monthly_volume=500.0,
        local_meets_bar=True,
        cloud_meets_bar=True,
    )
    assert result.verdict == VERDICT_INFEASIBLE
    assert result.volume is not None and result.volume > 1000.0


def test_break_even_local_below_bar() -> None:
    result = break_even(
        fixed_usd_per_month=34.41,
        cloud_usd_per_task=0.003,
        energy_usd_per_task=1e-5,
        capacity_per_month=108000.0,
        monthly_volume=15000.0,
        local_meets_bar=False,
        cloud_meets_bar=True,
    )
    assert result.verdict == VERDICT_LOCAL_BELOW_BAR
    assert result.volume is None
    assert result.machines_needed == 1


def test_break_even_no_cloud_meets_bar() -> None:
    result = break_even(
        fixed_usd_per_month=34.41,
        cloud_usd_per_task=None,
        energy_usd_per_task=1e-5,
        capacity_per_month=108000.0,
        monthly_volume=15000.0,
        local_meets_bar=True,
        cloud_meets_bar=False,
    )
    assert result.verdict == VERDICT_NO_CLOUD_BAR
    assert result.volume is None


def test_shared_machine_single_task_matches_dedicated_when_utilisation_low() -> None:
    result = shared_machine({"classification": SharedTask(15000.0, 108000.0, 1.3333e-05)}, 34.413333)
    assert result.machines == 1
    assert result.local_usd_per_task["classification"] * 1000 == pytest.approx(2.3076, rel=1e-3)


def test_shared_machine_charges_extra_machines_above_capacity() -> None:
    tasks = {
        "a": SharedTask(monthly_volume=80000.0, capacity_per_month=100000.0, energy_usd_per_task=0.0),
        "b": SharedTask(monthly_volume=50000.0, capacity_per_month=100000.0, energy_usd_per_task=0.0),
    }
    result = shared_machine(tasks, fixed_usd_per_month=100.0)
    assert result.utilisation == pytest.approx(1.3)
    assert result.machines == 2
    assert result.shared_fixed_usd_per_task == pytest.approx(200.0 / 130000.0)


def test_shared_machine_empty_is_zero() -> None:
    result = shared_machine({}, 100.0)
    assert result.machines == 0
    assert result.local_usd_per_task == {}


def test_sensitivity_grid_has_nine_cells_for_default_axes() -> None:
    cells = sensitivity_grid(
        purchase_price_usd=1200,
        allocation=1.0,
        idle_watts=5,
        electricity_usd_per_kwh=0.30,
        ops_usd_per_month=0.0,
        cloud_usd_per_task=0.003,
        energy_usd_per_task=1.3333e-05,
        capacity_per_month=108000.0,
        monthly_volume=15000.0,
        local_meets_bar=True,
        cloud_meets_bar=True,
    )
    assert len(cells) == 9
    lifetimes = {c.lifetime_years for c in cells}
    multipliers = {c.price_multiplier for c in cells}
    assert lifetimes == {2.0, 3.0, 4.0}
    assert multipliers == {0.75, 1.0, 1.25}
    # a longer lifetime and a lower price both lower the fixed cost
    cheapest = min(cells, key=lambda c: c.fixed_usd_per_month)
    assert cheapest.lifetime_years == 4.0
    assert cheapest.price_multiplier == 0.75


def test_identical_extra_machines_never_create_a_break_even_seeded_grid() -> None:
    """If one machine cannot break even within its capacity, N identical machines cannot either:
    fixed cost and capacity both scale by N, so the break-even/capacity ratio is unchanged."""
    rng = random.Random(20260928)
    for _ in range(200):
        fixed1 = rng.uniform(1.0, 500.0)
        cloud = rng.uniform(0.0005, 5.0)
        cap1 = rng.uniform(100.0, 200000.0)
        volume = rng.uniform(10.0, 300000.0)
        force_never = rng.random() < 0.3
        energy = cloud + rng.uniform(0.0, 1.0) if force_never else rng.uniform(0.0, cloud)

        be1 = break_even(
            fixed_usd_per_month=fixed1,
            cloud_usd_per_task=cloud,
            energy_usd_per_task=energy,
            capacity_per_month=cap1,
            monthly_volume=volume,
            local_meets_bar=True,
            cloud_meets_bar=True,
        )
        if be1.verdict not in (VERDICT_NEVER, VERDICT_INFEASIBLE):
            continue
        for n in (2, 3, 5, 10):
            be_n = break_even(
                fixed_usd_per_month=fixed1 * n,
                cloud_usd_per_task=cloud,
                energy_usd_per_task=energy,
                capacity_per_month=cap1 * n,
                monthly_volume=volume,
                local_meets_bar=True,
                cloud_meets_bar=True,
            )
            assert be_n.verdict == be1.verdict, (fixed1, cloud, energy, cap1, volume, n, be1, be_n)
