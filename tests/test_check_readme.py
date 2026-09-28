"""`scripts/check_readme.py`: README-numbers drift detection against the reference run."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from local_enough.report.build import readme_blocks
from local_enough.report.markdown import BLOCK_NAMES
from local_enough.tasks import registry
from test_report_build import route_cfg, write_full_synthetic_run

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_check_readme() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_readme", REPO_ROOT / "scripts" / "check_readme.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wrap(blocks: dict[str, str], names: list[str]) -> str:
    parts = ["# local-enough\n"]
    for name in names:
        parts.append(f"<!-- le:{name}:start -->\n{blocks[name].strip()}\n<!-- le:{name}:end -->\n")
    return "\n".join(parts)


def test_detects_a_changed_block_and_write_fixes_it(tmp_path: Path) -> None:
    check_readme = _load_check_readme()
    run, specs = write_full_synthetic_run(tmp_path / "run")
    blocks = readme_blocks(run, specs=specs, route_cfg=route_cfg())

    mutated = dict(blocks)
    mutated["spend"] = "this is stale content that does not match the run\n"
    readme_path = tmp_path / "README.md"
    readme_path.write_text(_wrap(mutated, list(BLOCK_NAMES)), encoding="utf-8")

    differing, missing, drift = check_readme.check_readme(readme_path, run, write=False)
    assert differing == ["spend"]
    assert missing == []
    assert drift == []
    # Unchanged on disk: the mismatch is still there when read again.
    assert "stale content" in readme_path.read_text(encoding="utf-8")

    differing2, missing2, drift2 = check_readme.check_readme(readme_path, run, write=True)
    assert differing2 == []  # fixed in place
    assert missing2 == []
    assert drift2 == []
    assert "stale content" not in readme_path.read_text(encoding="utf-8")

    # Idempotent: running again with the now-correct README finds nothing to change.
    differing3, missing3, drift3 = check_readme.check_readme(readme_path, run, write=False)
    assert (differing3, missing3, drift3) == ([], [], [])


def test_missing_blocks_are_an_error_only_when_markers_are_present(tmp_path: Path) -> None:
    check_readme = _load_check_readme()
    run, specs = write_full_synthetic_run(tmp_path / "run")
    blocks = readme_blocks(run, specs=specs, route_cfg=route_cfg())

    # Only "headline" is present as a marker; the rest are simply absent from the README.
    partial_path = tmp_path / "PARTIAL.md"
    partial_path.write_text(_wrap(blocks, ["headline"]), encoding="utf-8")
    _differing, missing, _drift = check_readme.check_readme(partial_path, run, write=False)
    assert set(missing) == set(BLOCK_NAMES) - {"headline"}

    # No markers at all: nothing is "missing" because there is nothing to check.
    bare_path = tmp_path / "BARE.md"
    bare_path.write_text("# local-enough\n\nNo generated blocks here.\n", encoding="utf-8")
    _differing2, missing2, _drift2 = check_readme.check_readme(bare_path, run, write=False)
    assert missing2 == []


def test_detects_dataset_sha256_drift(tmp_path: Path) -> None:
    check_readme = _load_check_readme()
    run, _specs = write_full_synthetic_run(tmp_path / "run")

    classification = next(s for s in registry.bundled_tasks() if s.name == "classification")
    assert classification.root is not None
    real_sha = hashlib.sha256((classification.root / classification.calib).read_bytes()).hexdigest()
    run.merge_json("env.json", {"datasets_sha256": {"classification:calib": real_sha}})
    assert check_readme.check_dataset_drift(run) == []

    run.merge_json("env.json", {"datasets_sha256": {"classification:calib": "0" * 64}})
    drift = check_readme.check_dataset_drift(run)
    assert any("classification:calib" in line for line in drift)


def test_main_exits_1_on_mismatch_and_0_once_fixed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    check_readme = _load_check_readme()
    run, specs = write_full_synthetic_run(tmp_path / "run")
    blocks = readme_blocks(run, specs=specs, route_cfg=route_cfg())
    mutated = dict(blocks)
    mutated["plan"] = "not the real plan text\n"
    readme_path = tmp_path / "README.md"
    readme_path.write_text(_wrap(mutated, list(BLOCK_NAMES)), encoding="utf-8")

    monkeypatch.setattr("local_enough.paths.reference_run_dir", lambda: run.path)

    assert check_readme.main(["--readme", str(readme_path)]) == 1
    assert check_readme.main(["--readme", str(readme_path), "--write"]) == 0
    assert check_readme.main(["--readme", str(readme_path)]) == 0
