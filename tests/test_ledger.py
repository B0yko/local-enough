from __future__ import annotations

import asyncio
import random
from pathlib import Path

import pytest

from local_enough.bench.ledger import BudgetExceeded, Ledger, estimate_tokens


def test_estimate_tokens_sums_per_message() -> None:
    messages = [{"role": "user", "content": "abcdefghi"}, {"role": "system", "content": "abc"}]
    # ceil(9/3)+8=11, ceil(3/3)+8=9 -> 20
    assert estimate_tokens(messages) == 20


@pytest.mark.asyncio
async def test_reserve_and_settle_writes_ledger_and_run_copy(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ledger = Ledger("proj", cap_usd=1.0, warn_usd=0.5, run_dir=run_dir)
    async with ledger.reserve(0.10, {"model_id": "m1", "command": "bench", "task": "t", "item_id": "i1"}) as r:
        r.settle(0.05, "usage.cost", {"prompt_tokens": 10, "completion_tokens": 5})

    assert ledger.total_spent() == pytest.approx(0.05)
    assert ledger.outstanding_usd == pytest.approx(0.0)
    home_lines = ledger.path.read_text(encoding="utf-8").strip().splitlines()
    run_lines = (run_dir / "cost_ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(home_lines) == 1
    assert len(run_lines) == 1
    assert '"actual_usd": 0.05' in home_lines[0]
    assert '"model_id": "m1"' in home_lines[0]


@pytest.mark.asyncio
async def test_budget_exceeded_raised_before_entering_block(tmp_path: Path) -> None:
    ledger = Ledger("proj", cap_usd=0.05, warn_usd=None)
    entered = False
    with pytest.raises(BudgetExceeded):
        async with ledger.reserve(0.10, {"model_id": "m1"}):
            entered = True
    assert entered is False
    assert ledger.total_spent() == 0.0
    assert ledger.outstanding_usd == 0.0


@pytest.mark.asyncio
async def test_env_budget_var_lowers_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_ENOUGH_BUDGET_USD", "0.02")
    ledger = Ledger("proj", cap_usd=1.0, warn_usd=None)
    assert ledger.cap_usd == pytest.approx(0.02)
    with pytest.raises(BudgetExceeded):
        async with ledger.reserve(0.05, {"model_id": "m1"}):
            pass


@pytest.mark.asyncio
async def test_unsettled_reservation_releases_budget_with_warning() -> None:
    ledger = Ledger("proj", cap_usd=0.10, warn_usd=None)
    with pytest.warns(UserWarning):
        async with ledger.reserve(0.10, {"model_id": "m1"}):
            pass  # never settled
    assert ledger.total_spent() == 0.0
    assert ledger.outstanding_usd == 0.0
    async with ledger.reserve(0.10, {"model_id": "m1"}) as r:
        r.settle(0.02, "usage.cost")
    assert ledger.total_spent() == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_double_settle_raises() -> None:
    ledger = Ledger("proj", cap_usd=1.0, warn_usd=None)
    async with ledger.reserve(0.10, {"model_id": "m1"}) as r:
        r.settle(0.01, "usage.cost")
        with pytest.raises(RuntimeError):
            r.settle(0.01, "usage.cost")


@pytest.mark.asyncio
async def test_50_concurrent_calls_never_overshoot_cap_and_stop_cleanly(tmp_path: Path) -> None:
    cap = 1.0
    ledger = Ledger("proj", cap_usd=cap, warn_usd=0.8, run_dir=tmp_path / "run")
    rng = random.Random(11)
    peak = 0.0
    peak_lock = asyncio.Lock()
    refused = 0

    async def one_call(i: int) -> None:
        nonlocal peak, refused
        estimate = 0.05
        try:
            async with ledger.reserve(
                estimate, {"model_id": f"m{i}", "command": "bench", "task": "t", "item_id": str(i)}
            ) as r:
                async with peak_lock:
                    current = ledger.total_spent() + ledger.outstanding_usd
                    peak = max(peak, current)
                await asyncio.sleep(rng.uniform(0, 0.01))
                actual = estimate * rng.uniform(0.5, 1.0)
                r.settle(actual, "usage.cost", {"prompt_tokens": 100, "completion_tokens": 16})
        except BudgetExceeded:
            refused += 1

    await asyncio.gather(*(one_call(i) for i in range(50)))

    assert peak <= cap + 1e-9
    assert ledger.total_spent() <= cap + 1e-9
    assert ledger.outstanding_usd == pytest.approx(0.0)
    assert refused > 0  # cap of 1.0 with 50 x ~0.05 reservations must refuse some calls
    settled_lines = ledger.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(settled_lines) == 50 - refused
