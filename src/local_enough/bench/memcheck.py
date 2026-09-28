"""The "both local models plus router loaded" memory check.

Starts every ``launch: mlx`` model in the config, warms each with one request per task, builds the router in this
process (plan, gates and baselines from the run), sends one request per task through it, then records the
physical footprint of each model server and of the router process, and the system-wide memory in use. Only local
endpoints are wired into the router here, so the check never calls a cloud model.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import ExitStack
from datetime import UTC, datetime
from typing import Any

import httpx

from local_enough.bench import memory
from local_enough.bench.ledger import Ledger
from local_enough.bench.rundir import RunDir
from local_enough.bench.soak import local_endpoint
from local_enough.config import Config, LocalModel, RouteConfig
from local_enough.evaluate import task_specs_for_run
from local_enough.providers.mlx import MlxServer
from local_enough.providers.openai_compat import ChatClient, Endpoint
from local_enough.tasks import registry
from local_enough.tasks.base import TaskSpec


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


async def _warm(endpoints: dict[str, Endpoint], specs: dict[str, TaskSpec]) -> None:
    client = ChatClient()
    for endpoint in endpoints.values():
        for spec in specs.values():
            item = registry.load_items(spec, "calib")[0]
            messages = registry.get_kind(spec.kind).render_messages(spec, item)
            await client.complete(endpoint, messages, spec.output_cap())


async def _through_router(
    run: RunDir, specs: dict[str, TaskSpec], route_cfg: RouteConfig, cfg: Config, endpoints: dict[str, Endpoint]
) -> None:
    from local_enough.route import gates as gates_module
    from local_enough.route.planner import build_plan
    from local_enough.route.server import create_app

    plan = build_plan(run, specs, route_cfg)
    app = create_app(
        plan,
        specs,
        endpoints,
        gates_module.build_gate_context(run, specs),
        Ledger(cfg.project, cfg.budget_usd, cfg.budget_warn_usd),
        route_cfg,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://router", timeout=120.0) as client:
        for spec in specs.values():
            item = registry.load_items(spec, "calib")[0]
            raw = registry.get_kind(spec.kind).user_input(spec, item)
            body = {"model": f"local-enough/{spec.name}", "messages": [{"role": "user", "content": raw}]}
            await client.post("/v1/chat/completions", json=body)


def run_memory_check(cfg: Config, route_cfg: RouteConfig, run: RunDir) -> dict[str, Any]:
    """Measure and merge ``memory.json["combined"]``; returns the recorded entry."""
    specs = task_specs_for_run(run)
    idle_used = memory.system_used_bytes()
    local_models = [m for m in cfg.models if isinstance(m, LocalModel) and m.launch is not None]
    servers: dict[str, MlxServer] = {}
    endpoints: dict[str, Endpoint] = {}
    with ExitStack() as stack:
        for model_cfg in local_models:
            server, endpoint = local_endpoint(model_cfg, route_cfg)
            if server is not None:
                stack.callback(server.stop)
                servers[model_cfg.id] = server
            endpoints[model_cfg.id] = endpoint
        asyncio.run(_warm(endpoints, specs))
        asyncio.run(_through_router(run, specs, route_cfg, cfg, endpoints))
        models_bytes = {mid: memory.footprint_bytes(s.pid) for mid, s in servers.items() if s.pid is not None}
        router_bytes = memory.footprint_bytes(os.getpid())
        system_used = memory.system_used_bytes()
    entry = {
        "created_utc": _now_utc(),
        "method": "footprint (phys_footprint, includes Metal allocations)",
        "models_bytes": models_bytes,
        "router_bytes": router_bytes,
        "ours_bytes": sum(models_bytes.values()) + router_bytes,
        "system_used_bytes": system_used,
        "system_used_before_bytes": idle_used,
    }
    run.merge_json("memory.json", {"combined": entry})
    return entry
