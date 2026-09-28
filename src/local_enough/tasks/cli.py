"""``local-enough datasets`` commands: list and validate task datasets."""

from __future__ import annotations

from typing import Annotated

import typer

from local_enough.tasks import registry
from local_enough.tasks.base import TaskSpec

app = typer.Typer(help="Inspect and validate task datasets.", no_args_is_help=True)


def _split_size(spec: TaskSpec, split: str) -> int | None:
    try:
        return len(registry.load_items(spec, split))
    except FileNotFoundError:
        return None


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    typer.echo(line)
    typer.echo("  ".join("-" * w for w in widths))
    for row in rows:
        typer.echo("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


@app.command("list")
def list_datasets(
    tasks: Annotated[str, typer.Option("--tasks", help="'all' plus optional bring-your-own task.yaml paths.")] = "all",
) -> None:
    """Print name, kind, split sizes, licence and metric for the selected tasks."""
    specs = registry.resolve_tasks(tasks)
    rows = []
    for spec in specs:
        calib_n = _split_size(spec, "calib")
        test_n = _split_size(spec, "test")
        train_n = _split_size(spec, "train") if spec.train else None
        licence = spec.card.licence if spec.card else ""
        metric = spec.card.metric if spec.card else ""
        rows.append(
            [
                spec.name,
                spec.kind,
                str(calib_n) if calib_n is not None else "-",
                str(test_n) if test_n is not None else "-",
                str(train_n) if train_n is not None else "-",
                licence,
                metric,
            ]
        )
    _print_table(["name", "kind", "calib", "test", "train", "licence", "metric"], rows)


@app.command("validate")
def validate_dataset(
    path: Annotated[str, typer.Argument(help="Directory containing task.yaml, or the task.yaml file itself.")],
) -> None:
    """Validate a dataset (bundled or bring-your-own) against its kind's schema."""
    try:
        spec = registry.load_task(path)
    except Exception as exc:
        typer.echo(f"error: could not load task.yaml at {path!r}: {exc}", err=True)
        raise typer.Exit(1) from None

    try:
        module = registry.get_kind(spec.kind)
    except KeyError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None

    errors: list[str] = []
    split_files = {"calib": spec.calib, "test": spec.test, "train": spec.train}
    for split, filename in split_files.items():
        if filename is None:
            continue
        try:
            items = registry.load_items(spec, split)
        except (FileNotFoundError, OSError) as exc:
            errors.append(f"{split}: {exc}")
            continue
        if not items:
            errors.append(f"{split}: no records in {filename!r}")
        for i, item in enumerate(items):
            for err in module.validate_item(spec, item):
                errors.append(f"{split}[{i}] ({item.get('id', '?')}): {err}")

    if errors:
        for err in errors:
            typer.echo(f"  - {err}", err=True)
        typer.echo(f"FAILED: {spec.name} ({spec.kind}): {len(errors)} error(s)", err=True)
        raise typer.Exit(1)
    typer.echo(f"OK: {spec.name} ({spec.kind}) is valid")
