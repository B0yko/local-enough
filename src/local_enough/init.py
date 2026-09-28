"""``local-enough init``: write starter config files and the bring-your-own-task example."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer

from local_enough import paths

# Relative to both the package's data/init/ directory and the destination directory. These are the
# same files committed under examples/; a test asserts the two trees stay byte-identical.
FILES: tuple[str, ...] = (
    "config.yaml",
    "quickstart.yaml",
    "local.yaml",
    "route.yaml",
    ".env.example",
    "custom-task/task.yaml",
    "custom-task/calib.jsonl",
    "custom-task/test.jsonl",
    "custom-task/train.jsonl",
)

NEXT_STEPS = """
Next steps:
  1. Offline, no keys:
       local-enough report --run reference
       local-enough route --simulate --run reference
  2. Live, one OpenRouter key, under $0.05:
       export OPENROUTER_API_KEY=...
       local-enough bench --tasks all --models quickstart.yaml --split calib --limit 10 --dry-run
       local-enough bench --tasks all --models quickstart.yaml --split calib --limit 10
  3. Local, no key:
       local-enough models pull --models local.yaml
       local-enough bench --tasks all --models local.yaml --split calib --limit 10 --run runs/local
       local-enough route --run runs/local --models local.yaml
""".strip("\n")


def init(
    directory: Annotated[
        Path,
        typer.Argument(
            metavar="DIR",
            help="Directory to write into (created if missing). Defaults to the current directory.",
        ),
    ] = Path(),
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite files that already exist instead of skipping them.")
    ] = False,
) -> None:
    """Write config.yaml, quickstart.yaml, local.yaml, route.yaml, .env.example and the custom-task example."""
    source = paths.init_templates_dir()
    written: list[str] = []
    skipped: list[str] = []
    for rel in FILES:
        dst = directory / rel
        if dst.exists() and not force:
            skipped.append(rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, dst)
        written.append(rel)

    for rel in written:
        typer.echo(f"wrote {directory / rel}")
    if skipped:
        typer.echo(f"skipped (already exist; pass --force to overwrite): {', '.join(skipped)}")

    typer.echo("")
    typer.echo(NEXT_STEPS)
