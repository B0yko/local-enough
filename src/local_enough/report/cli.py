"""Typer command body for ``local-enough report``; registered by ``cli.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from local_enough import paths
from local_enough.report.build import build_report


def report(
    run: Annotated[
        str | None, typer.Option("--run", help="Run directory, 'reference', or omit for the latest ./runs/*.")
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Output directory for the report.")] = Path("report"),
) -> None:
    """Write index.html, report.md and the ADR from a run directory. Never fails on a partial run."""
    try:
        run_dir = paths.resolve_run(run)
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    written = build_report(run_dir, out)
    typer.echo(f"wrote {len(written)} file(s) to {out}")
    for path in written:
        typer.echo(f"  {path}")
