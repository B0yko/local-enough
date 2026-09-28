"""``local-enough judge calibrate`` / ``local-enough judge score``."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from local_enough.config import load_config
from local_enough.judge import calibrate as calibrate_mod
from local_enough.judge import score as score_mod

app = typer.Typer(help="Calibrate and run the summarisation judge.", no_args_is_help=True)


@app.command("calibrate")
def judge_calibrate(
    models: Annotated[Path, typer.Option("--models", exists=True, help="Path to config.yaml.")],
    run: Annotated[Path, typer.Option("--run", help="Run directory to write judge_calibration.json into.")],
) -> None:
    """Judge every candidate on judge-calib, select the best by balanced accuracy, measure it on judge-holdout."""
    cfg = load_config(models)
    result = asyncio.run(calibrate_mod.run_calibration(run, cfg))
    holdout = result["holdout"]
    typer.echo(
        f"selected {result['selected']!r}: holdout balanced_accuracy={holdout['balanced_accuracy']:.3f} "
        f"(target {result['target_balanced_accuracy']:.2f}, meets_target={result['meets_target']})"
    )


@app.command("score")
def judge_score(
    run: Annotated[Path, typer.Option("--run", help="Run directory with predictions and judge_calibration.json.")],
    models: Annotated[Path, typer.Option("--models", exists=True, help="Path to config.yaml.")],
) -> None:
    """Judge every non-baseline model's summarisation predictions in a run that are not already scored."""
    cfg = load_config(models)
    try:
        records = asyncio.run(score_mod.run_score(run, cfg))
    except score_mod.MissingCalibration as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"judged {len(records)} new summarisation prediction(s) in {run}")
