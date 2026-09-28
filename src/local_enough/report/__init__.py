"""Report generation: run-directory tables, charts, and the ``local-enough report`` command.

Re-exports the two entry points named in docs/run-directory.md: :func:`build_report` (writes
``index.html``, ``report.md`` and the ADR) and :func:`readme_blocks` (the Markdown blocks
``scripts/check_readme.py`` compares against README.md).
"""

from __future__ import annotations

from local_enough.report.build import build_report, readme_blocks

__all__ = ["build_report", "readme_blocks"]
