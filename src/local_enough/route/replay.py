"""Router live check: replay mixed test items through a *running* router with the official
``openai`` client and compare its decisions with an offline :func:`local_enough.route.simulate.simulate` run.

Sends real HTTP requests to ``url`` (a router the caller already started, e.g. with ``local-enough route``),
so this module never spins up a server itself. Nothing here writes request or response text to
``live_check.json``: only counts, rates and latencies.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import openai

from local_enough.bench.rundir import RunDir
from local_enough.config import RouteConfig
from local_enough.providers.openai_compat import ChatClient, Endpoint
from local_enough.route import simulate as simulate_module
from local_enough.route.planner import Plan
from local_enough.stats import p50_p95
from local_enough.tasks import registry
from local_enough.tasks.base import Item, TaskSpec


@dataclass(frozen=True)
class Mismatch:
    task: str
    item_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task, "item_id": self.item_id, "reason": self.reason}


@dataclass(frozen=True)
class LiveCheck:
    n: int
    seed: int
    created_utc: str
    overhead_p50_ms: float
    direct_p50_ms: float | None
    router_p50_ms: float
    decision_match_rate: float
    mismatches: list[Mismatch]
    local_only_cloud_calls: int
    per_task: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "seed": self.seed,
            "created_utc": self.created_utc,
            "overhead_p50_ms": self.overhead_p50_ms,
            "direct_p50_ms": self.direct_p50_ms,
            "router_p50_ms": self.router_p50_ms,
            "decision_match_rate": self.decision_match_rate,
            "mismatches": [m.to_dict() for m in self.mismatches],
            "local_only_cloud_calls": self.local_only_cloud_calls,
            "per_task": self.per_task,
        }


def _sample_items(specs: dict[str, TaskSpec], route_cfg: RouteConfig, n: int, seed: int) -> list[tuple[str, Item]]:
    """``n`` test items across tasks in ``workload_mix`` proportions, seeded and deterministic."""
    rng = random.Random(seed)
    tasks = [t for t in route_cfg.workload_mix if t in specs]
    if not tasks:
        return []
    weights = [route_cfg.workload_mix[t] for t in tasks]
    items_by_task = {t: registry.load_items(specs[t], "test") for t in tasks}
    orders = {t: rng.sample(range(len(items)), len(items)) for t, items in items_by_task.items() if items}
    cursor = dict.fromkeys(tasks, 0)

    sample: list[tuple[str, Item]] = []
    for _ in range(n):
        task = rng.choices(tasks, weights=weights, k=1)[0]
        order = orders.get(task)
        if not order:
            continue
        items = items_by_task[task]
        idx = order[cursor[task] % len(order)]
        cursor[task] += 1
        sample.append((task, items[idx]))
    return sample


def _display_of(run_dir: RunDir, plan: Plan, task_name: str, model_id: str | None) -> str | None:
    """The display model a given internal ``model_id`` answers as, matching what the router itself returns."""
    if model_id is None:
        return None
    tp = plan.tasks.get(task_name)
    if tp is not None and tp.candidate_kind(model_id) == "baseline":
        return model_id
    for rec in run_dir.predictions("A"):
        if rec.get("model_id") == model_id and rec.get("model"):
            return str(rec["model"])
    return model_id


def _fetch_stats(url: str) -> dict[str, Any]:
    with httpx.Client(timeout=10.0) as client:
        return dict(client.get(f"{url.rstrip('/')}/stats").json())


def _local_only_cloud_calls(plan: Plan, before: dict[str, Any], after: dict[str, Any]) -> int:
    total = 0
    for task_name in plan.local_only_tasks:
        b = (before.get("tasks", {}).get(task_name) or {}).get("calls_by_class", {}).get("cloud", 0)
        a = (after.get("tasks", {}).get(task_name) or {}).get("calls_by_class", {}).get("cloud", 0)
        total += max(0, a - b)
    return total


async def _direct_latencies_ms(
    calls: list[tuple[str, list[dict[str, str]], int]], endpoints: dict[str, Endpoint]
) -> list[float]:
    """Call each local model directly with the router's own rendered prompt; ``calls`` is (model_id, messages,
    max_tokens)."""
    latencies: list[float] = []
    async with ChatClient() as client:
        for model_id, messages, max_tokens in calls:
            endpoint = endpoints.get(model_id)
            if endpoint is None:
                continue
            result = await client.complete(endpoint, messages, max_tokens, temperature=0.0)
            latencies.append(result.latency_s * 1000)
    return latencies


def run_replay(
    url: str,
    run: RunDir | str | Path,
    specs: dict[str, TaskSpec],
    route_cfg: RouteConfig,
    plan: Plan,
    *,
    n: int = 100,
    seed: int = 7,
    endpoints: dict[str, Endpoint] | None = None,
) -> LiveCheck:
    """Replay ``n`` mixed test items through the router already running at ``url``."""
    run_dir = run if isinstance(run, RunDir) else RunDir(run)
    sample = _sample_items(specs, route_cfg, n, seed)

    sim = simulate_module.simulate(run_dir, plan, specs, route_cfg, split="test", gates=plan.gates_enabled)
    expected_by_key = {(task, r.item_id): r for task, task_result in sim.tasks.items() for r in task_result.items}

    client = openai.OpenAI(base_url=url.rstrip("/") + "/v1", api_key="unused")
    stats_before = _fetch_stats(url)

    router_latencies_ms: list[float] = []
    overheads_ms: list[float] = []
    mismatches: list[Mismatch] = []
    per_task_total: dict[str, int] = {}
    per_task_match: dict[str, int] = {}
    matched = 0
    refusals: dict[str, int] = {}
    direct_calls: list[tuple[str, list[dict[str, str]], int]] = []

    for task_name, item in sample:
        spec = specs[task_name]
        module = registry.get_kind(spec.kind)
        user_input = module.user_input(spec, item)
        item_id = str(item["id"])
        per_task_total[task_name] = per_task_total.get(task_name, 0) + 1

        t0 = time.perf_counter()
        try:
            raw = client.chat.completions.with_raw_response.create(
                model=f"local-enough/{task_name}", messages=[{"role": "user", "content": user_input}]
            )
        except openai.APIStatusError as exc:
            router_latencies_ms.append((time.perf_counter() - t0) * 1000)
            expected = expected_by_key.get((task_name, item_id))
            # A refusal is a decision too: the offline replay serves nothing (unservable or exhausted local-only
            # chain) exactly when the router answers 503.
            if exc.status_code == 503 and expected is not None and expected.served_by is None:
                matched += 1
                per_task_match[task_name] = per_task_match.get(task_name, 0) + 1
                refusals[task_name] = refusals.get(task_name, 0) + 1
                continue
            mismatches.append(Mismatch(task_name, item_id, f"router returned HTTP {exc.status_code}"))
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000
        router_latencies_ms.append(elapsed_ms)
        headers = raw.headers
        answering_model = headers.get("x-local-enough-model")
        escalated = headers.get("x-local-enough-escalated") == "true"
        upstream_ms_raw = headers.get("x-local-enough-upstream-ms")
        if upstream_ms_raw is not None:
            overheads_ms.append(elapsed_ms - float(upstream_ms_raw))

        expected = expected_by_key.get((task_name, item_id))
        expected_display = _display_of(run_dir, plan, task_name, expected.served_by if expected else None)
        if expected is not None and answering_model == expected_display and escalated == expected.escalated:
            matched += 1
            per_task_match[task_name] = per_task_match.get(task_name, 0) + 1
        else:
            reason = (
                f"router answered as {answering_model!r} (escalated={escalated}), offline replay expected "
                f"{expected_display!r} (escalated={expected.escalated if expected else None!r})"
            )
            mismatches.append(Mismatch(task_name, item_id, reason))

        tp = plan.tasks.get(task_name)
        served_model_id = expected.served_by if expected else None
        if (
            endpoints
            and tp is not None
            and served_model_id is not None
            and tp.candidate_kind(served_model_id) == "local"
        ):
            messages = module.render_messages(spec, item)
            direct_calls.append((served_model_id, messages, spec.output_cap()))

    stats_after = _fetch_stats(url)
    direct_latencies_ms = asyncio.run(_direct_latencies_ms(direct_calls, endpoints)) if endpoints else []

    router_p50, _ = p50_p95(router_latencies_ms)
    overhead_p50, _ = p50_p95(overheads_ms) if overheads_ms else (float("nan"), float("nan"))
    direct_p50 = p50_p95(direct_latencies_ms)[0] if direct_latencies_ms else None

    per_task = {
        name: {
            "n": total,
            "decision_match_rate": per_task_match.get(name, 0) / total if total else float("nan"),
            "refused_503": refusals.get(name, 0),
        }
        for name, total in per_task_total.items()
    }

    return LiveCheck(
        n=len(sample),
        seed=seed,
        created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        overhead_p50_ms=overhead_p50,
        direct_p50_ms=direct_p50,
        router_p50_ms=router_p50,
        decision_match_rate=(matched / len(sample)) if sample else float("nan"),
        mismatches=mismatches,
        local_only_cloud_calls=_local_only_cloud_calls(plan, stats_before, stats_after),
        per_task=per_task,
    )


def write_live_check(run: RunDir | str | Path, live_check: LiveCheck) -> None:
    run_dir = run if isinstance(run, RunDir) else RunDir(run)
    run_dir.write_json("live_check.json", live_check.to_dict())
