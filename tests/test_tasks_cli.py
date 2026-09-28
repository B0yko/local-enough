"""`local-enough datasets list/validate` via typer.testing.CliRunner."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from local_enough.cli import app

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_TASK_DIR = REPO_ROOT / "examples" / "custom-task"

runner = CliRunner()


def test_datasets_list_bundled_tasks() -> None:
    result = runner.invoke(app, ["datasets", "list"])
    assert result.exit_code == 0
    for name in ["extraction", "classification", "pii_redaction", "summarisation", "entity_matching"]:
        assert name in result.output
    assert "licence" in result.output and "metric" in result.output


def test_datasets_list_includes_custom_task() -> None:
    result = runner.invoke(app, ["datasets", "list", "--tasks", f"all,{EXAMPLE_TASK_DIR / 'task.yaml'}"])
    assert result.exit_code == 0
    assert "custom_support_tickets" in result.output


def test_datasets_validate_bundled_task_ok() -> None:
    result = runner.invoke(
        app, ["datasets", "validate", str(REPO_ROOT / "src/local_enough/data/datasets/entity_matching")]
    )
    assert result.exit_code == 0
    assert "OK" in result.output


def test_datasets_validate_custom_example_ok() -> None:
    result = runner.invoke(app, ["datasets", "validate", str(EXAMPLE_TASK_DIR)])
    assert result.exit_code == 0
    assert "custom_support_tickets" in result.output


def test_datasets_validate_missing_task_yaml_errors() -> None:
    result = runner.invoke(app, ["datasets", "validate", "/nonexistent/path"])
    assert result.exit_code == 1


def test_datasets_validate_broken_dataset_reports_errors_and_exits_1(tmp_path: Path) -> None:
    task_dir = tmp_path / "broken"
    task_dir.mkdir()
    (task_dir / "task.yaml").write_text(
        "name: broken\nkind: pii_redaction\ncalib: calib.jsonl\ntest: test.jsonl\npii_types: [PERSON]\n",
        encoding="utf-8",
    )
    (task_dir / "calib.jsonl").write_text(
        '{"id": "a1", "text": "Hello John", "spans": [{"start": 6, "end": 20, "type": "PERSON"}]}\n'
        '{"id": "a2", "spans": []}\n',
        encoding="utf-8",
    )
    (task_dir / "test.jsonl").write_text('{"id": "b1", "text": "Nothing here", "spans": []}\n', encoding="utf-8")

    result = runner.invoke(app, ["datasets", "validate", str(task_dir)])
    assert result.exit_code == 1
    assert "invalid offsets" in result.output
    assert "missing or invalid 'text'" in result.output
    assert "FAILED" in result.output


def test_datasets_validate_empty_split_errors(tmp_path: Path) -> None:
    task_dir = tmp_path / "empty"
    task_dir.mkdir()
    (task_dir / "task.yaml").write_text(
        "name: empty\nkind: classification\ncalib: calib.jsonl\ntest: test.jsonl\nlabels: [a, b]\n",
        encoding="utf-8",
    )
    (task_dir / "calib.jsonl").write_text("", encoding="utf-8")
    (task_dir / "test.jsonl").write_text('{"id": "1", "text": "x", "label": "a"}\n', encoding="utf-8")

    result = runner.invoke(app, ["datasets", "validate", str(task_dir)])
    assert result.exit_code == 1
    assert "no records" in result.output
