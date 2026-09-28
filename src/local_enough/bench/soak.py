"""Pass C: sustained throughput of one local model under a fixed-concurrency mixed workload.

Cycles ``test``-split items of every task in ``route.yaml``'s ``workload_mix`` in a seeded,
deterministic interleaving, keeping ``concurrency`` requests in flight until the clock runs out.
The first-5-minute vs last-5-minute throughput ratio (the throttle factor) feeds the cost model's
sustained throughput figure; see ``docs/cost-model.md`` and docs/run-directory.md ("Bench").
"""

from __future__ import annotations

import asyncio
import itertools
import os
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from local_enough.bench.load import LoadSampler
from local_enough.bench.rundir import RunDir
from local_enough.bench.runner import _max_tokens_for
from local_enough.config import Config, LocalModel, RouteConfig
from local_enough.providers.caffeinate import spawn_caffeinate
from local_enough.providers.mlx import MlxServer
from local_enough.providers.openai_compat import ChatClient, Endpoint, endpoint_for
from local_enough.tasks import registry
from local_enough.tasks.base import Item, TaskKindModule, TaskSpec

_PERIOD_LENGTH = 100


def _period(mix: dict[str, float], seed: int, length: int = _PERIOD_LENGTH) -> list[str]:
    """A deterministic, seed-shuffled sequence of task names approximating ``mix`` proportions."""
    names = sorted(mix)
    alloc = {name: round(mix[name] * length) for name in names}
    diff = length - sum(alloc.values())
    i = 0
    guard = 0
    while diff != 0 and guard < 10 * length + 100:
        name = names[i % len(names)]
        if diff > 0:
            alloc[name] += 1
            diff -= 1
        elif alloc[name] > 0:
            alloc[name] -= 1
            diff += 1
        i += 1
        guard += 1
    sequence = [name for name in names for _ in range(alloc[name])]
    random.Random(seed).shuffle(sequence)
    return sequence or list(names)


def _per_minute_tph(completions: list[dict[str, Any]]) -> list[float]:
    if not completions:
        return []
    last_bucket = max(int(c["t"] // 60.0) for c in completions)
    buckets = [0] * (last_bucket + 1)
    for c in completions:
        buckets[int(c["t"] // 60.0)] += 1
    return [count * 60.0 for count in buckets]


Workload = tuple[dict[str, TaskSpec], dict[str, TaskKindModule], dict[str, list[Item]]]


def build_workload(mix: dict[str, float]) -> Workload:
    """Bundled task specs/modules/``test``-split items for every task named in ``mix``."""
    specs_by_name: dict[str, TaskSpec] = {spec.name: spec for spec in registry.bundled_tasks()}
    missing = sorted(name for name in mix if name not in specs_by_name)
    if missing:
        raise ValueError(f"workload_mix references unknown bundled tasks: {missing}")
    specs = {name: specs_by_name[name] for name in mix}
    modules = {name: registry.get_kind(specs[name].kind) for name in mix}
    items = {name: registry.load_items(specs[name], "test") for name in mix}
    return specs, modules, items


class ItemCycler:
    """Round-robins each task's item list independently, wrapping around when exhausted."""

    def __init__(self, items_by_task: dict[str, list[Item]]) -> None:
        self._items = items_by_task
        self._cursors: dict[str, int] = dict.fromkeys(items_by_task, 0)

    def next(self, task_name: str) -> Item:
        items = self._items[task_name]
        idx = self._cursors[task_name] % len(items)
        self._cursors[task_name] += 1
        return items[idx]


def local_endpoint(model_cfg: LocalModel, route_cfg: RouteConfig) -> tuple[MlxServer | None, Endpoint]:
    """Start (and return) an ``MlxServer`` for a ``launch: mlx`` model, or build a ``base_url`` endpoint."""
    if model_cfg.launch is None:
        return None, endpoint_for(model_cfg, timeout_s=model_cfg.timeout_s or route_cfg.timeouts_s.local)
    launch = model_cfg.launch
    server = MlxServer(
        launch.repo, launch.revision, host=launch.host, port=launch.port, startup_timeout_s=launch.startup_timeout_s
    )
    server.start()
    endpoint = endpoint_for(
        model_cfg,
        base_url=server.base_url,
        model_name="default_model",
        display_model=server.display_model,
        timeout_s=model_cfg.timeout_s or route_cfg.timeouts_s.local,
    )
    return server, endpoint


def own_pids(server: MlxServer | None, caffeinate_pid: int | None) -> set[int]:
    pids = {os.getpid()}
    if server is not None and server.pid is not None:
        pids.add(server.pid)
    if caffeinate_pid is not None:
        pids.add(caffeinate_pid)
    return pids


def record_load_window(run_dir: RunDir, name: str, sampler: LoadSampler) -> None:
    if not sampler.samples:
        return
    payload = run_dir.read_json("load_samples.json", {}) or {}
    windows = list(payload.get("windows") or [])
    windows.append(
        {
            "name": name,
            "start_utc": _iso(sampler.samples[0].t),
            "samples": [
                {"t": s.t, "load1": s.load1, "other_cpu_pct": s.other_cpu_pct, "other_mem_gb": s.other_mem_gb}
                for s in sampler.samples
            ],
            "contaminated": sampler.contaminated,
        }
    )
    payload["windows"] = windows
    run_dir.write_json("load_samples.json", payload)


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


async def run_soak(
    cfg: Config,
    model_id: str,
    route_cfg: RouteConfig,
    run: Path,
    *,
    minutes: float,
    concurrency: int = 4,
    seed: int = 7,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Run the soak workload against ``model_id`` for ``minutes`` and write ``soak.json``.

    ``minutes`` accepts floats; a test passes a tiny value with a fast fake server so the whole
    loop finishes in real time proportional to the number of (mocked, near-instant) calls it makes
    before the injected ``clock`` reports the deadline has passed.
    """
    model_cfg = cfg.model(model_id)
    if not isinstance(model_cfg, LocalModel):
        raise ValueError(f"soak only runs local models; {model_id!r} is not one")

    mix = {name: weight for name, weight in route_cfg.workload_mix.items() if weight > 0}
    if not mix:
        raise ValueError("route.yaml workload_mix has no tasks with weight > 0")
    specs_by_name, modules_by_name, items_by_task = build_workload(mix)
    cycler = ItemCycler(items_by_task)

    run_dir = RunDir(run)
    caffeinate_proc = spawn_caffeinate(os.getpid())
    caffeinate_pid = caffeinate_proc.pid if caffeinate_proc is not None else None
    server, endpoint = local_endpoint(model_cfg, route_cfg)

    sampler = LoadSampler(own_pids_fn=lambda: own_pids(server, caffeinate_pid))
    sampler.sample_once()
    await sampler.start()

    completions: list[dict[str, Any]] = []
    try:
        start_t = clock()
        deadline = start_t + minutes * 60.0
        cycle = itertools.cycle(_period(mix, seed))
        cycle_lock = asyncio.Lock()

        async def _next_task_and_item() -> tuple[str, Item]:
            async with cycle_lock:
                task_name = next(cycle)
                return task_name, cycler.next(task_name)

        async def _worker(client: ChatClient, endpoint: Endpoint) -> None:
            while clock() < deadline:
                task_name, item = await _next_task_and_item()
                spec = specs_by_name[task_name]
                module = modules_by_name[task_name]
                messages = module.render_messages(spec, item)
                max_tokens = _max_tokens_for(cfg, spec)
                result = await client.complete(endpoint, messages, max_tokens, temperature=0.0)
                completions.append({"t": clock() - start_t, "task": task_name, "tokens": result.completion_tokens})

        async with ChatClient() as client:
            await asyncio.gather(*(_worker(client, endpoint) for _ in range(max(1, concurrency))))
    finally:
        if server is not None:
            server.stop()
        await sampler.stop()
        record_load_window(run_dir, f"soak:{model_id}", sampler)

    per_minute_tph = _per_minute_tph(completions)
    first_window = per_minute_tph[: min(5, len(per_minute_tph))]
    last_window = per_minute_tph[-min(5, len(per_minute_tph)) :] if per_minute_tph else []
    first5_tph = mean(first_window) if first_window else 0.0
    last5_tph = mean(last_window) if last_window else 0.0
    throttle_factor = (last5_tph / first5_tph) if first5_tph > 0 else float("nan")

    run_dir.merge_json(
        "soak.json",
        {
            model_id: {
                "minutes": minutes,
                "concurrency": concurrency,
                "mix": mix,
                "per_minute_tph": per_minute_tph,
                "first5_tph": first5_tph,
                "last5_tph": last5_tph,
                "throttle_factor": throttle_factor,
                "completions_n": len(completions),
            }
        },
    )
