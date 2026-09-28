"""``local-enough init``: writes package data into a target directory, never clobbering by default."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from local_enough import paths
from local_enough.cli import app
from local_enough.init import FILES

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"


def test_init_templates_are_byte_identical_to_examples() -> None:
    """The CLI reads src/local_enough/data/init/; examples/ is the copy users read on GitHub. Never let them drift."""
    source = paths.init_templates_dir()
    for rel in FILES:
        packaged = (source / rel).read_bytes()
        committed = (EXAMPLES / rel).read_bytes()
        assert packaged == committed, f"examples/{rel} and data/init/{rel} differ"


def test_init_writes_every_file_into_a_fresh_directory(tmp_path: Path) -> None:
    target = tmp_path / "project"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0, result.output
    for rel in FILES:
        assert (target / rel).is_file(), rel
    assert "Next steps" in result.output
    assert "route --simulate --run reference" in result.output


def test_init_defaults_to_the_current_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "custom-task" / "train.jsonl").is_file()


def test_init_skips_existing_files_without_force(tmp_path: Path) -> None:
    target = tmp_path / "project"
    runner.invoke(app, ["init", str(target)])
    marker = (target / "config.yaml").read_text()
    (target / "config.yaml").write_text("# user edits kept\n")

    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0, result.output
    assert "skipped" in result.output
    assert "config.yaml" in result.output
    assert (target / "config.yaml").read_text() == "# user edits kept\n"
    assert (target / "config.yaml").read_text() != marker


def test_init_force_overwrites_existing_files(tmp_path: Path) -> None:
    target = tmp_path / "project"
    runner.invoke(app, ["init", str(target)])
    (target / "config.yaml").write_text("# user edits\n")

    result = runner.invoke(app, ["init", str(target), "--force"])
    assert result.exit_code == 0, result.output
    assert "skipped" not in result.output
    assert (target / "config.yaml").read_text() != "# user edits\n"


def test_init_reports_partial_skip_when_only_some_files_exist(tmp_path: Path) -> None:
    target = tmp_path / "project"
    target.mkdir()
    (target / "route.yaml").write_text("# pre-existing\n")

    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0, result.output
    assert (target / "route.yaml").read_text() == "# pre-existing\n"
    assert (target / "config.yaml").is_file()
    assert "route.yaml" in result.output
    assert "wrote" in result.output
