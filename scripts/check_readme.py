#!/usr/bin/env python3
"""README-numbers drift check: regenerates every generated block from the reference run.

Every block in README.md between ``<!-- le:NAME:start -->`` / ``<!-- le:NAME:end -->`` markers must
match what ``local_enough.report.readme_blocks`` computes from the committed reference run
(``paths.reference_run_dir()``) right now. It also checks that the dataset and prompt-template
sha256 recorded in that reference run's ``env.json``/``tasks.json`` still match the bundled files on
disk, so neither the datasets nor the prompt templates can silently drift from the numbers a reader
sees in the README.

Usage:
    uv run python scripts/check_readme.py [--write] [--readme README.md]

Exits 1 and prints a unified diff per differing block (and any drift lines) when something does not
match; ``--write`` rewrites the differing blocks in place (this cannot fix dataset/template drift,
which needs a real rerun of the measured pipeline).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import sys
from pathlib import Path

from local_enough import paths
from local_enough.bench.rundir import RunDir
from local_enough.report.build import readme_blocks
from local_enough.report.markdown import BLOCK_NAMES
from local_enough.tasks import registry

_START = "<!-- le:{name}:start -->"
_END = "<!-- le:{name}:end -->"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_block(text: str, name: str) -> tuple[int, int] | None:
    """(content_start, content_end) between the named markers, or ``None`` if the start marker is absent."""
    start_marker, end_marker = _START.format(name=name), _END.format(name=name)
    start = text.find(start_marker)
    if start == -1:
        return None
    end = text.find(end_marker, start)
    if end == -1:
        raise ValueError(f"README has {start_marker!r} with no matching {end_marker!r}")
    return start + len(start_marker), end


def check_dataset_drift(run: RunDir) -> list[str]:
    """Compare env.json/tasks.json recorded sha256 against the bundled dataset/template files now."""
    problems: list[str] = []
    env = run.read_json("env.json", {}) or {}
    tasks_json = run.read_json("tasks.json", {}) or {}
    recorded_datasets: dict[str, str] = env.get("datasets_sha256") or {}
    recorded_templates: dict[str, str] = env.get("templates_sha256") or {}

    for spec in registry.bundled_tasks():
        for split_name, filename in (("calib", spec.calib), ("test", spec.test), ("train", spec.train)):
            if filename is None or spec.root is None:
                continue
            path = spec.root / filename
            if not path.exists():
                continue
            current = _sha256(path)
            key = f"{spec.name}:{split_name}"

            recorded = recorded_datasets.get(key)
            if recorded is not None and recorded != current:
                problems.append(f"dataset drift: env.json datasets_sha256[{key!r}] no longer matches {filename}")

            entry_sha = ((tasks_json.get(spec.name) or {}).get("sha256") or {}).get(split_name)
            if entry_sha is not None and entry_sha != current:
                problems.append(
                    f"dataset drift: tasks.json[{spec.name!r}].sha256[{split_name!r}] no longer matches {filename}"
                )

        template_path = registry.template_files()[spec.kind]
        if not template_path.exists():
            continue
        recorded_t = recorded_templates.get(spec.kind)
        if recorded_t is not None and recorded_t != _sha256(template_path):
            problems.append(f"template drift: env.json templates_sha256[{spec.kind!r}] no longer matches its template")

    return problems


def check_readme(readme_path: Path, run: RunDir, *, write: bool) -> tuple[list[str], list[str], list[str]]:
    """Returns ``(differing_blocks, missing_blocks, dataset_drift)``; rewrites in place when ``write``."""
    text = readme_path.read_text(encoding="utf-8")
    has_markers = any(_START.format(name=name) in text for name in BLOCK_NAMES)
    blocks = readme_blocks(run)

    differing: list[str] = []
    missing: list[str] = []
    new_text = text
    for name in BLOCK_NAMES:
        span = _find_block(new_text, name)
        if span is None:
            if has_markers:
                missing.append(name)
            continue
        start, end = span
        current_block = new_text[start:end]
        expected_block = "\n" + blocks[name].strip("\n") + "\n"
        if current_block == expected_block:
            continue
        differing.append(name)
        diff = "\n".join(
            difflib.unified_diff(
                current_block.splitlines(),
                expected_block.splitlines(),
                fromfile=f"README.md [{name}]",
                tofile=f"generated [{name}]",
                lineterm="",
            )
        )
        print(diff)
        if write:
            new_text = new_text[:start] + expected_block + new_text[end:]

    if write and differing:
        readme_path.write_text(new_text, encoding="utf-8")
        print(f"rewrote {len(differing)} block(s) in {readme_path}")
        differing = []

    return differing, missing, check_dataset_drift(run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="Rewrite differing blocks in README.md in place.")
    parser.add_argument("--readme", default=Path("README.md"), type=Path, help="Path to the README to check.")
    args = parser.parse_args(argv)

    readme_path: Path = args.readme
    if not readme_path.exists():
        print(f"error: {readme_path} not found", file=sys.stderr)
        return 1

    run = RunDir(paths.reference_run_dir())
    differing, missing, drift = check_readme(readme_path, run, write=args.write)

    if missing:
        print(f"README is missing generated block(s): {', '.join(missing)}")
    for line in drift:
        print(line)

    if differing or missing or drift:
        return 1
    print("README is up to date with the reference run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
