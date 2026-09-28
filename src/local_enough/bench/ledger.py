"""A cost ledger with a hard spend cap, safe under concurrent reservations.

Every paid call reserves an upper-bound estimate before it is sent, then settles to the actual
cost afterwards. The reservation keeps concurrent calls from ever overshooting the cap: a call
that would push committed-plus-outstanding spend past the cap is refused before it is sent.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import math
import os
import warnings
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from local_enough.paths import ledger_path
from local_enough.providers.openai_compat import Message

_ENV_BUDGET_VAR = "LOCAL_ENOUGH_BUDGET_USD"


class BudgetExceeded(RuntimeError):
    """Raised before a call is sent if it would push spend past the cap."""

    def __init__(self, cap_usd: float, would_be_usd: float) -> None:
        super().__init__(f"reserving this call would reach ${would_be_usd:.4f}, over the ${cap_usd:.4f} cap")
        self.cap_usd = cap_usd
        self.would_be_usd = would_be_usd


def estimate_tokens(messages: Sequence[Message]) -> int:
    """A conservative token estimate: ``sum(ceil(len(content) / 3.0) + 8)`` over the messages."""
    return sum(math.ceil(len(m.get("content", "")) / 3.0) + 8 for m in messages)


def _read_settled_total(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0.0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += float(record.get("actual_usd") or 0.0)
    return total


@dataclass
class Reservation:
    """A held budget slot; call ``settle`` once the call finishes, with any actual cost."""

    estimate_usd: float
    meta: dict[str, Any]
    _ledger: Ledger
    _settled: bool = field(default=False, repr=False)

    def settle(self, actual_usd: float, cost_source: str, usage_meta: dict[str, Any] | None = None) -> None:
        """Record the actual cost and release the reservation. Safe to call at most once."""
        if self._settled:
            raise RuntimeError("Reservation.settle() called more than once")
        self._settled = True
        self._ledger._settle(self, actual_usd, cost_source, usage_meta or {})


class Ledger:
    """Append-only JSONL cost ledger, capped at ``min(cap_usd, $LOCAL_ENOUGH_BUDGET_USD)``."""

    def __init__(
        self,
        project: str,
        cap_usd: float,
        warn_usd: float | None,
        run_dir: Path | None = None,
    ) -> None:
        self.project = project
        env_cap = os.environ.get(_ENV_BUDGET_VAR)
        self.cap_usd = min(cap_usd, float(env_cap)) if env_cap else cap_usd
        self.warn_usd = warn_usd
        self.run_dir = run_dir

        self.path = ledger_path(project)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)

        self._lock = asyncio.Lock()
        self._committed_usd = _read_settled_total(self.path)
        self._outstanding_usd = 0.0
        self._warned = self.warn_usd is not None and self._committed_usd >= self.warn_usd

    def total_spent(self) -> float:
        return self._committed_usd

    @property
    def outstanding_usd(self) -> float:
        """Sum of reservations not yet settled. Exposed mainly so tests can assert the cap holds."""
        return self._outstanding_usd

    @asynccontextmanager
    async def reserve(self, estimate_usd: float, meta: dict[str, Any]) -> AsyncIterator[Reservation]:
        """Reserve budget for one call. Raises :class:`BudgetExceeded` before the call is sent."""
        async with self._lock:
            would_be = self._committed_usd + self._outstanding_usd + estimate_usd
            if would_be > self.cap_usd:
                raise BudgetExceeded(self.cap_usd, would_be)
            self._outstanding_usd += estimate_usd

        reservation = Reservation(estimate_usd=estimate_usd, meta=dict(meta), _ledger=self)
        try:
            yield reservation
        finally:
            if not reservation._settled:
                async with self._lock:
                    self._outstanding_usd -= estimate_usd
                warnings.warn(
                    f"ledger reservation for {meta.get('model_id', '?')!r} was never settled; releasing it",
                    stacklevel=2,
                )

    def _settle(
        self, reservation: Reservation, actual_usd: float, cost_source: str, usage_meta: dict[str, Any]
    ) -> None:
        # Synchronous and awaits nothing: under asyncio's cooperative scheduling this whole method
        # runs without yielding control, so it needs no lock to stay race-free against other tasks.
        self._outstanding_usd -= reservation.estimate_usd
        self._committed_usd += actual_usd
        crossed_warn = self.warn_usd is not None and not self._warned and self._committed_usd >= self.warn_usd
        if crossed_warn:
            self._warned = True
        record = {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reserved_usd": reservation.estimate_usd,
            "actual_usd": actual_usd,
            "cost_source": cost_source,
            **reservation.meta,
            **usage_meta,
        }
        self._write_line(record)
        if crossed_warn:
            warnings.warn(
                f"ledger for {self.project!r} crossed the ${self.warn_usd:.2f} warning level "
                f"(now ${self._committed_usd:.4f})",
                stacklevel=2,
            )

    def _write_line(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True) + "\n"
        for target in (self.path, self.run_dir / "cost_ledger.jsonl" if self.run_dir else None):
            if target is None:
                continue
            with target.open("a", encoding="utf-8") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    fh.write(line)
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
