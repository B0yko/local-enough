from __future__ import annotations

from pathlib import Path

import respx
from typer.testing import CliRunner

import fakes
from local_enough import paths
from local_enough.cli import app

runner = CliRunner()
LOCAL_URL = "https://local.test/v1"
CLASSIFICATION_TASK = paths.datasets_dir() / "classification" / "task.yaml"


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
project: cli-test
budget_usd: 5.0
models:
  - id: local-test
    kind: local
    base_url: {LOCAL_URL}
    model: local-model
  - id: tfidf-baseline
    kind: baseline
    task: classification
"""
    )
    return config_path


def test_bench_command_runs_and_writes_run_dir(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    run_dir = tmp_path / "run"
    with respx.mock(assert_all_called=False) as mock:
        fakes.register_chat(mock, LOCAL_URL, fakes.fixed_content_handler("card_swallowed"))
        result = runner.invoke(
            app,
            [
                "bench",
                "--models",
                str(config_path),
                "--tasks",
                str(CLASSIFICATION_TASK),
                "--split",
                "calib",
                "--limit",
                "1",
                "--run",
                str(run_dir),
            ],
        )
    assert result.exit_code == 0, result.output
    assert (run_dir / "predictions.jsonl.gz").exists()
    assert "run directory" in result.output


def test_bench_resume_requires_existing_dir(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "bench",
            "--models",
            str(config_path),
            "--tasks",
            str(CLASSIFICATION_TASK),
            "--resume",
            str(tmp_path / "missing"),
        ],
    )
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_bench_dry_run_exits_zero(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    run_dir = tmp_path / "run"
    result = runner.invoke(
        app,
        [
            "bench",
            "--models",
            str(config_path),
            "--tasks",
            str(CLASSIFICATION_TASK),
            "--split",
            "calib",
            "--limit",
            "1",
            "--run",
            str(run_dir),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "total: $" in result.output
    assert not run_dir.exists()


def test_soak_command_writes_soak_json(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    route_path = tmp_path / "route.yaml"
    route_path.write_text("workload_mix: {classification: 1.0}\nreference_monthly_volume: 1000\n")
    run_dir = tmp_path / "run"
    with respx.mock(assert_all_called=False) as mock:
        fakes.register_chat(mock, LOCAL_URL, fakes.fixed_content_handler("card_swallowed"))
        result = runner.invoke(
            app,
            [
                "soak",
                "--model",
                "local-test",
                "--models",
                str(config_path),
                "--run",
                str(run_dir),
                "--minutes",
                "0.02",
                "--concurrency",
                "1",
                "--route",
                str(route_path),
            ],
        )
    assert result.exit_code == 0, result.output
    assert (run_dir / "soak.json").exists()


def test_power_probe_command_errors_on_non_local_model(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "project: cli-test\nbudget_usd: 5.0\nmodels:\n"
        "  - id: cloud-a\n    kind: cloud\n    base_url: https://cloud.test/api/v1\n"
        "    api_key_env: OPENROUTER_API_KEY\n    model: vendor/model-a\n"
    )
    result = runner.invoke(
        app,
        ["power-probe", "--model", "cloud-a", "--models", str(config_path), "--run", str(tmp_path / "run")],
    )
    assert result.exit_code == 1
    assert "local" in result.output
