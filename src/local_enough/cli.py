"""Command-line entry point. Sub-apps register their commands here."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from local_enough import __version__
from local_enough.config import LocalModel, load_config
from local_enough.providers.hub import ModelBudgetExceeded, pull_models, write_downloads_json

app = typer.Typer(
    name="local-enough",
    help="Benchmark back-office AI tasks on local and cloud models, cost them, and route to the cheapest model "
    "that meets the quality bar.",
    no_args_is_help=True,
    add_completion=False,
)

models_app = typer.Typer(help="Manage local model downloads.", no_args_is_help=True)
app.add_typer(models_app, name="models")


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show version."),
) -> None:
    """local-enough CLI."""


@models_app.command("pull")
def models_pull(
    models: Annotated[Path, typer.Option("--models", exists=True, help="Path to config.yaml.")],
    run: Annotated[Path | None, typer.Option("--run", help="Run directory to write downloads.json into.")] = None,
) -> None:
    """Download every ``launch: mlx`` model in ``config.yaml``, under ``local_models_max_gb``."""
    run_dir = run or Path()
    cfg = load_config(models)
    launches = [(m.id, m.launch) for m in cfg.models if isinstance(m, LocalModel) and m.launch is not None]
    if not launches:
        typer.echo("No 'launch: mlx' models configured; nothing to pull.")
        raise typer.Exit(code=0)

    try:
        downloads = pull_models(launches, max_gb=cfg.local_models_max_gb)
    except ModelBudgetExceeded as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    run_dir.mkdir(parents=True, exist_ok=True)
    downloads_path = write_downloads_json(run_dir, downloads)
    total_gb = sum(d.bytes for _, d in downloads) / 1_000_000_000
    new_gb = sum(d.bytes_new for _, d in downloads) / 1_000_000_000
    typer.echo(
        f"Pulled {len(downloads)} model(s): {total_gb:.2f} GB total, {new_gb:.2f} GB newly downloaded "
        f"-> {downloads_path}"
    )
