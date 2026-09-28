"""Typer command bodies for ``route`` (plan / simulate / serve) and ``route replay``."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn

from local_enough import evaluate, paths
from local_enough.bench.ledger import Ledger
from local_enough.bench.rundir import RunDir
from local_enough.config import CloudModel, Config, LocalModel, RouteConfig, load_config, load_route
from local_enough.providers.mlx import MlxServer
from local_enough.providers.openai_compat import Endpoint, endpoint_for
from local_enough.route import gates as gates_module
from local_enough.route import planner
from local_enough.route import replay as replay_module
from local_enough.route import simulate as simulate_module
from local_enough.route.planner import Plan
from local_enough.route.server import create_app
from local_enough.tasks.base import TaskSpec

app = typer.Typer(
    help="Build a routing plan, replay it offline, or serve the OpenAI-compatible router.", no_args_is_help=True
)

_DEFAULT_ROUTE_YAML = Path("route.yaml")


def _resolve_route(route: Path | None) -> RouteConfig:
    if route is not None:
        return load_route(route)
    if _DEFAULT_ROUTE_YAML.exists():
        return load_route(_DEFAULT_ROUTE_YAML)
    from local_enough.bench.runner import DEFAULT_ROUTE_CONFIG

    return DEFAULT_ROUTE_CONFIG


def _fail(exc: Exception) -> typer.Exit:
    typer.echo(f"error: {exc}", err=True)
    return typer.Exit(code=1)


def _load_plan(
    run: str, route_path: Path | None, *, gates: bool
) -> tuple[RunDir, dict[str, TaskSpec], RouteConfig, Plan]:
    run_dir = RunDir(paths.resolve_run(run))
    specs = evaluate.task_specs_for_run(run_dir)
    route_cfg = _resolve_route(route_path)
    plan = planner.build_plan(run_dir, specs, route_cfg, gates=gates)
    return run_dir, specs, route_cfg, plan


def _fmt(value: float | None, spec: str = ".2f") -> str:
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return format(value, spec)


def _format_mixed_table(rows: list[simulate_module.MixedRow]) -> str:
    headers = ["ROW", "USD/1K", "LOCAL%", "ESCALATION%", "P50 MS", "P95 MS", "VS FRONTIER%", "VS CHEAPEST%"]
    body = [
        [
            row.label,
            _fmt(row.usd_per_1k),
            _fmt(row.served_locally_pct, ".1f"),
            _fmt(row.escalation_pct, ".1f"),
            _fmt(row.p50_s * 1000 if row.p50_s is not None else None, ".0f"),
            _fmt(row.p95_s * 1000 if row.p95_s is not None else None, ".0f"),
            _fmt(row.saving_vs_all_frontier_pct, ".1f"),
            _fmt(row.saving_vs_cheapest_cloud_pct, ".1f"),
        ]
        for row in rows
    ]
    widths = [max(len(headers[i]), *(len(r[i]) for r in body)) for i in range(len(headers))]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))]
    lines.extend("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in body)
    return "\n".join(lines)


def _model_endpoint(model_id: str, model_cfg: LocalModel | CloudModel | Any, route_cfg: RouteConfig) -> Endpoint | None:
    if isinstance(model_cfg, CloudModel):
        return endpoint_for(model_cfg, timeout_s=route_cfg.timeouts_s.cloud)
    if isinstance(model_cfg, LocalModel) and model_cfg.base_url is not None:
        return endpoint_for(model_cfg, timeout_s=route_cfg.timeouts_s.local)
    return None  # launched local models are handled separately (need a running MlxServer first)


async def _serve(
    run_dir: RunDir, specs: dict[str, TaskSpec], route_cfg: RouteConfig, plan: Plan, cfg: Config, host: str, port: int
) -> None:
    servers: list[MlxServer] = []
    endpoints: dict[str, Endpoint] = {}
    model_ids = {model_id for tp in plan.tasks.values() for model_id in tp.chain_model_ids()}

    try:
        for model_id in model_ids:
            try:
                model_cfg = cfg.model(model_id)
            except KeyError:
                continue
            if isinstance(model_cfg, LocalModel) and model_cfg.launch is not None:
                launch = model_cfg.launch
                server = MlxServer(
                    launch.repo,
                    launch.revision,
                    host=launch.host,
                    port=launch.port,
                    startup_timeout_s=launch.startup_timeout_s,
                )
                server.start()
                servers.append(server)
                endpoints[model_id] = endpoint_for(
                    model_cfg,
                    base_url=server.base_url,
                    model_name="default_model",
                    display_model=server.display_model,
                    timeout_s=route_cfg.timeouts_s.local,
                )
                continue
            endpoint = _model_endpoint(model_id, model_cfg, route_cfg)
            if endpoint is not None:
                endpoints[model_id] = endpoint

        gate_ctx = gates_module.build_gate_context(run_dir, specs)
        ledger = Ledger(cfg.project, cfg.budget_usd, cfg.budget_warn_usd)
        fastapi_app = create_app(plan, specs, endpoints, gate_ctx, ledger, route_cfg)

        uvicorn_config = uvicorn.Config(fastapi_app, host=host, port=port, log_level="info", access_log=False)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()
    finally:
        for server in servers:
            server.stop()


@app.callback(invoke_without_command=True)
def route(
    ctx: typer.Context,
    run: Annotated[str, typer.Option("--run", help="Run directory, or 'reference'.")] = paths.REFERENCE,
    models: Annotated[
        Path | None, typer.Option("--models", exists=True, help="config.yaml; required to serve.")
    ] = None,
    route_path: Annotated[
        Path | None, typer.Option("--route", help="route.yaml (default: ./route.yaml if present).")
    ] = None,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    print_plan_flag: Annotated[bool, typer.Option("--print-plan", help="Print the plan and exit.")] = False,
    simulate_flag: Annotated[
        bool, typer.Option("--simulate", help="Replay recorded test predictions and print the mixed table.")
    ] = False,
    no_gates: Annotated[bool, typer.Option("--no-gates", help="Disable gates; only valid with --simulate.")] = False,
) -> None:
    """Build the routing plan from ``--run``'s calib results; print it, simulate it, or serve it."""
    if ctx.invoked_subcommand is not None:
        return
    if no_gates and not simulate_flag:
        typer.echo("error: --no-gates is only valid together with --simulate", err=True)
        raise typer.Exit(code=1)

    try:
        run_dir, specs, route_cfg, plan = _load_plan(run, route_path, gates=not no_gates)
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(exc) from exc

    if print_plan_flag:
        typer.echo(planner.print_plan(plan))
        return

    if simulate_flag:
        rows = simulate_module.mixed_table(run_dir, plan, specs, route_cfg, split="test")
        typer.echo(_format_mixed_table(rows))
        return

    if models is None:
        typer.echo("error: --models is required to serve (not needed for --print-plan or --simulate)", err=True)
        raise typer.Exit(code=1)

    try:
        cfg = load_config(models)
    except Exception as exc:
        raise _fail(exc) from exc

    asyncio.run(_serve(run_dir, specs, route_cfg, plan, cfg, host, port))


@app.command("replay")
def replay(
    url: Annotated[str, typer.Option("--url", help="Base URL of a running router, e.g. http://127.0.0.1:8000.")],
    run: Annotated[Path, typer.Option("--run", exists=True, help="Run directory the router was built from.")],
    n: Annotated[int, typer.Option("--n", help="Number of mixed test items to replay.")] = 100,
    seed: Annotated[int, typer.Option("--seed", help="Sampling seed.")] = 7,
    models: Annotated[
        Path | None, typer.Option("--models", exists=True, help="config.yaml, for the direct-call comparison.")
    ] = None,
    route_path: Annotated[
        Path | None, typer.Option("--route", help="route.yaml (default: ./route.yaml if present).")
    ] = None,
) -> None:
    """Replay mixed test items through a running router and compare with the offline simulation."""
    run_dir = RunDir(run)
    specs = evaluate.task_specs_for_run(run_dir)
    route_cfg = _resolve_route(route_path)
    plan = planner.build_plan(run_dir, specs, route_cfg)

    endpoints: dict[str, Endpoint] | None = None
    if models is not None:
        cfg = load_config(models)
        endpoints = {}
        for tp in plan.tasks.values():
            for model_id in tp.chain_model_ids():
                if tp.candidate_kind(model_id) != "local":
                    continue
                try:
                    model_cfg = cfg.model(model_id)
                except KeyError:
                    continue
                if not isinstance(model_cfg, LocalModel):
                    continue
                if model_cfg.launch is not None:
                    launch = model_cfg.launch
                    endpoints[model_id] = Endpoint(
                        model_id=model_id,
                        base_url=f"http://{launch.host}:{launch.port}/v1",
                        model_name="default_model",
                        api_key="",
                        kind="local",
                        display_model=model_id,
                    )
                elif model_cfg.base_url is not None:
                    endpoints[model_id] = endpoint_for(model_cfg, timeout_s=route_cfg.timeouts_s.local)

    live_check = replay_module.run_replay(url, run_dir, specs, route_cfg, plan, n=n, seed=seed, endpoints=endpoints)
    replay_module.write_live_check(run_dir, live_check)
    typer.echo(
        f"decision match rate: {live_check.decision_match_rate:.3f}; local-only cloud calls: "
        f"{live_check.local_only_cloud_calls}; wrote {run_dir.file('live_check.json')}"
    )
