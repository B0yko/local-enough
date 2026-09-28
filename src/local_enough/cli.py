"""Command-line entry point. Sub-apps register their commands here."""

from __future__ import annotations

import typer

from local_enough import __version__

app = typer.Typer(
    name="local-enough",
    help="Benchmark back-office AI tasks on local and cloud models, cost them, and route to the cheapest model "
    "that meets the quality bar.",
    no_args_is_help=True,
    add_completion=False,
)


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show version."),
) -> None:
    """local-enough CLI."""
