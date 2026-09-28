"""Typer command bodies for ``bench``, ``soak`` and ``power-probe``; registered by ``cli.py``."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from local_enough import paths
from local_enough.bench.ledger import BudgetExceeded
from local_enough.bench.probe import run_probe
from local_enough.bench.runner import DEFAULT_ROUTE_CONFIG, PreflightFailed, run_bench
from local_enough.bench.soak import run_soak
from local_enough.config import RouteConfig, load_config, load_route
from local_enough.providers.openai_compat import MissingApiKey
from local_enough.tasks import registry

_ERRORS = (BudgetExceeded, PreflightFailed, MissingApiKey, ValueError, FileNotFoundError)


def _resolve_route(route: Path | None) -> RouteConfig:
    if route is not None:
        return load_route(route)
    default_path = Path("route.yaml")
    if default_path.exists():
        return load_route(default_path)
    return DEFAULT_ROUTE_CONFIG


def _fail(exc: Exception) -> typer.Exit:
    typer.echo(f"error: {exc}", err=True)
    return typer.Exit(code=1)


def bench(
    models: Annotated[Path, typer.Option("--models", exists=True, help="Path to config.yaml.")],
    tasks: Annotated[str, typer.Option("--tasks", help="'all' and/or comma-separated task.yaml paths.")] = "all",
    split: Annotated[str, typer.Option("--split", help="Comma-separated splits to run.")] = "calib,test",
    limit: Annotated[int | None, typer.Option("--limit", help="Stratified sample size per task/split.")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Sampling seed.")] = 7,
    run: Annotated[Path | None, typer.Option("--run", help="Run directory (default: ./runs/<UTC stamp>).")] = None,
    resume: Annotated[Path | None, typer.Option("--resume", help="Existing run directory to resume.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Estimate calls/tokens/USD and exit.")] = False,
    local_only: Annotated[bool, typer.Option("--local-only", help="Skip baseline and cloud models.")] = False,
    concurrency: Annotated[int, typer.Option("--concurrency", help="Local-model concurrency (Pass B if > 1).")] = 1,
    route: Annotated[
        Path | None, typer.Option("--route", help="route.yaml (default: ./route.yaml if present).")
    ] = None,
) -> None:
    """Run configured models over the selected tasks/splits and write a run directory."""
    if run is not None and resume is not None:
        typer.echo("error: pass only one of --run or --resume", err=True)
        raise typer.Exit(code=1)
    if resume is not None:
        if not resume.exists():
            typer.echo(f"error: --resume {resume} does not exist", err=True)
            raise typer.Exit(code=1)
        run_dir = resume
    elif run is not None:
        run_dir = run
    else:
        run_dir = paths.new_run_dir()

    try:
        cfg = load_config(models)
        task_specs = registry.resolve_tasks(tasks)
        splits = [s.strip() for s in split.split(",") if s.strip()]
        route_cfg = _resolve_route(route)
        asyncio.run(
            run_bench(
                cfg,
                task_specs,
                splits,
                run_dir,
                limit=limit,
                seed=seed,
                local_only=local_only,
                concurrency=concurrency,
                dry_run=dry_run,
                route_cfg=route_cfg,
            )
        )
    except _ERRORS as exc:
        raise _fail(exc) from exc

    if not dry_run:
        typer.echo(f"run directory: {run_dir}")


def soak(
    model: Annotated[str, typer.Option("--model", help="Local model id from --models.")],
    models: Annotated[Path, typer.Option("--models", exists=True, help="Path to config.yaml.")],
    run: Annotated[Path, typer.Option("--run", help="Run directory to write soak.json into.")],
    minutes: Annotated[float, typer.Option("--minutes", help="Soak duration in minutes.")] = 20.0,
    concurrency: Annotated[int, typer.Option("--concurrency", help="Requests kept in flight.")] = 4,
    route: Annotated[
        Path | None, typer.Option("--route", help="route.yaml (default: ./route.yaml if present).")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the mixed-workload interleaving.")] = 7,
) -> None:
    """Sustained-throughput soak test (Pass C) for one local model."""
    try:
        cfg = load_config(models)
        route_cfg = _resolve_route(route)
        asyncio.run(run_soak(cfg, model, route_cfg, run, minutes=minutes, concurrency=concurrency, seed=seed))
    except _ERRORS as exc:
        raise _fail(exc) from exc
    typer.echo(f"soak results written to {run / 'soak.json'}")


def power_probe(
    model: Annotated[str, typer.Option("--model", help="Local model id from --models.")],
    models: Annotated[Path, typer.Option("--models", exists=True, help="Path to config.yaml.")],
    run: Annotated[Path, typer.Option("--run", help="Run directory to write power.json into.")],
    idle_s: Annotated[float, typer.Option("--idle-s", help="Idle measurement window in seconds.")] = 60.0,
    load_s: Annotated[float, typer.Option("--load-s", help="Load measurement window in seconds.")] = 120.0,
    concurrency: Annotated[int, typer.Option("--concurrency", help="Load concurrency during the load window.")] = 4,
) -> None:
    """Incremental-watts probe (no sudo) for one local model."""
    try:
        cfg = load_config(models)
        route_cfg = _resolve_route(None)
        asyncio.run(run_probe(cfg, model, route_cfg, run, idle_s=idle_s, load_s=load_s, concurrency=concurrency))
    except _ERRORS as exc:
        raise _fail(exc) from exc
    typer.echo(f"power-probe results written to {run / 'power.json'}")


def memory_check(
    models: Annotated[Path, typer.Option("--models", exists=True, help="Path to config.yaml.")],
    run: Annotated[Path, typer.Option("--run", help="Run directory to write memory.json into.")],
    route: Annotated[
        Path | None, typer.Option("--route", help="route.yaml (default: ./route.yaml if present).")
    ] = None,
) -> None:
    """Memory with every local model and the router loaded (no cloud calls)."""
    from local_enough.bench.memcheck import run_memory_check
    from local_enough.bench.rundir import RunDir

    try:
        entry = run_memory_check(load_config(models), _resolve_route(route), RunDir(run))
    except _ERRORS as exc:
        raise _fail(exc) from exc
    gib = 2**30
    ours, system = entry["ours_bytes"] / gib, entry["system_used_bytes"] / gib
    typer.echo(f"models + router: {ours:.2f} GiB; system in use: {system:.2f} GiB -> {run / 'memory.json'}")
