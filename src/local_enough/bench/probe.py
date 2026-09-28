"""``power-probe``: incremental watts of one local model under the Pass C workload, without sudo.

:func:`local_enough.bench.power.probe` is synchronous and blocking: it sleeps for the idle and load
windows while sampling ``ioreg``. It runs on a worker thread here (``asyncio.to_thread``) so the
rest of the CLI stays async; its ``start_load``/``stop_load`` callbacks start and stop plain
``threading.Thread`` workers that make synchronous HTTP calls against the model, independent of the
asyncio event loop, so there is no cross-thread coordination with it.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import itertools
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from local_enough.bench.load import LoadSampler
from local_enough.bench.power import BatteryReading, PowerTelemetryUnavailable, read_battery
from local_enough.bench.power import probe as power_probe
from local_enough.bench.rundir import RunDir
from local_enough.bench.runner import _max_tokens_for
from local_enough.bench.soak import ItemCycler, _period, build_workload, local_endpoint, own_pids, record_load_window
from local_enough.config import Config, LocalModel, RouteConfig
from local_enough.providers.caffeinate import spawn_caffeinate
from local_enough.providers.openai_compat import ChatClient, Endpoint
from local_enough.tasks.base import Item, TaskKindModule, TaskSpec

_PROBE_CONCURRENCY = 4


async def run_probe(
    cfg: Config,
    model_id: str,
    route_cfg: RouteConfig,
    run: Path,
    *,
    idle_s: float = 60.0,
    load_s: float = 120.0,
    concurrency: int = _PROBE_CONCURRENCY,
    seed: int = 7,
    sampler_fn: Callable[[], BatteryReading] = read_battery,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Warm up ``model_id``, then measure incremental watts and write ``power.json``.

    When battery telemetry is unavailable (a desktop Mac, or missing ``ioreg`` fields),
    ``power.json`` records ``{"mode": "unavailable", "reason": ...}`` for the model and a clear
    message is printed; the cost model then falls back to the machine's configured watts.
    """
    model_cfg = cfg.model(model_id)
    if not isinstance(model_cfg, LocalModel):
        raise ValueError(f"power-probe only runs local models; {model_id!r} is not one")

    mix = {name: weight for name, weight in route_cfg.workload_mix.items() if weight > 0}
    if not mix:
        raise ValueError("route.yaml workload_mix has no tasks with weight > 0")
    specs_by_name, modules_by_name, items_by_task = build_workload(mix)
    cycler = ItemCycler(items_by_task)
    cycle = itertools.cycle(_period(mix, seed))
    cycle_lock = threading.Lock()

    def _next_task_and_item() -> tuple[str, Item]:
        with cycle_lock:
            task_name = next(cycle)
            return task_name, cycler.next(task_name)

    run_dir = RunDir(run)
    caffeinate_proc = spawn_caffeinate(os.getpid())
    caffeinate_pid = caffeinate_proc.pid if caffeinate_proc is not None else None
    server, endpoint = local_endpoint(model_cfg, route_cfg)

    sampler = LoadSampler(own_pids_fn=lambda: own_pids(server, caffeinate_pid))
    sampler.sample_once()
    await sampler.start()

    try:
        async with ChatClient() as client:
            task_name, item = _next_task_and_item()
            spec, module = specs_by_name[task_name], modules_by_name[task_name]
            messages = module.render_messages(spec, item)
            # Warm-up: one request before measurement starts, outside the timed windows.
            await client.complete(endpoint, messages, _max_tokens_for(cfg, spec), temperature=0.0)

        load = _LoadGenerator(cfg, endpoint, specs_by_name, modules_by_name, _next_task_and_item, concurrency)
        try:
            result = await asyncio.to_thread(
                power_probe, idle_s, load_s, load.start, load.stop, sampler_fn, sleep_fn=sleep_fn
            )
        except PowerTelemetryUnavailable as exc:
            print(
                f"power-probe: battery telemetry unavailable for {model_id!r} ({exc}); "
                "the cost model will use the configured watts instead."
            )
            run_dir.merge_json("power.json", {model_id: {"mode": "unavailable", "reason": str(exc)}})
            return
    finally:
        if server is not None:
            server.stop()
        await sampler.stop()
        record_load_window(run_dir, f"power-probe:{model_id}", sampler)

    run_dir.merge_json("power.json", {model_id: dataclasses.asdict(result)})


def _post_sync(client: httpx.Client, endpoint: Endpoint, messages: list[dict[str, str]], max_tokens: int) -> None:
    url = f"{endpoint.base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
    body = {
        "model": endpoint.model_name,
        "messages": messages,
        "max_tokens": max_tokens + endpoint.reasoning_allowance_tokens,
        "temperature": 0.0,
    }
    # Best-effort load generation; a failed call still briefly exercised the server.
    with contextlib.suppress(httpx.HTTPError):
        client.post(url, json=body, headers=headers, timeout=endpoint.timeout_s)


class _LoadGenerator:
    """Plain OS threads making synchronous calls, started/stopped by ``power.probe``'s callbacks.

    Deliberately independent of the asyncio event loop (which is busy elsewhere, running the
    blocking ``power.probe`` call itself on a worker thread) so there is no cross-thread handoff.
    """

    def __init__(
        self,
        cfg: Config,
        endpoint: Endpoint,
        specs_by_name: dict[str, TaskSpec],
        modules_by_name: dict[str, TaskKindModule],
        next_task_and_item: Callable[[], tuple[str, Item]],
        concurrency: int,
    ) -> None:
        self._cfg = cfg
        self._endpoint = endpoint
        self._specs_by_name = specs_by_name
        self._modules_by_name = modules_by_name
        self._next_task_and_item = next_task_and_item
        self._concurrency = max(1, concurrency)
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def _worker(self) -> None:
        with httpx.Client() as client:
            while not self._stop_event.is_set():
                task_name, item = self._next_task_and_item()
                spec = self._specs_by_name[task_name]
                module = self._modules_by_name[task_name]
                messages = module.render_messages(spec, item)
                max_tokens = _max_tokens_for(self._cfg, spec)
                _post_sync(client, self._endpoint, messages, max_tokens)

    def start(self) -> None:
        self._stop_event.clear()
        self._threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(self._concurrency)]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join()
        self._threads = []
