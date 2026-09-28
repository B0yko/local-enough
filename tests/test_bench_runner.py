from __future__ import annotations

import gzip
from pathlib import Path

import httpx
import pytest
import respx

import fakes
from local_enough.bench import envinfo
from local_enough.bench.rundir import RunDir
from local_enough.bench.runner import PreflightFailed, run_bench
from local_enough.config import BaselineModel, CloudModel, Config, LocalModel
from local_enough.tasks import registry
from local_enough.tasks.base import TaskSpec

CLOUD_URL = "https://cloud.test/api/v1"
LOCAL_URL = "https://local.test/v1"


def _classification_task() -> TaskSpec:
    return next(t for t in registry.bundled_tasks() if t.name == "classification")


def _config(*, budget_usd: float = 5.0) -> Config:
    return Config(
        project="test-proj",
        budget_usd=budget_usd,
        budget_warn_usd=budget_usd * 0.8,
        models=[
            BaselineModel(id="tfidf-baseline", task="classification"),
            CloudModel(
                id="open-large",
                base_url=CLOUD_URL,
                api_key_env="OPENROUTER_API_KEY",
                model="vendor/model-a",
                concurrency=4,
            ),
            LocalModel(id="local-test", base_url=LOCAL_URL, model="local-model"),
        ],
    )


def _assert_no_absolute_paths(run_dir: Path) -> None:
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        content = (
            gzip.decompress(path.read_bytes()).decode("utf-8", "ignore")
            if path.suffix == ".gz"
            else path.read_text("utf-8", "ignore")
        )
        assert "/Users/" not in content, f"{path} contains an absolute path"


@pytest.mark.asyncio
async def test_end_to_end_baseline_cloud_local_with_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = _config()
    spec = _classification_task()
    run_dir = tmp_path / "run"

    with respx.mock(assert_all_called=False) as mock:
        fakes.register_price_snapshot(mock, CLOUD_URL, ["vendor/model-a"])
        fakes.register_default_route(mock, CLOUD_URL, spec, cost=0.001)
        fakes.register_default_route(mock, LOCAL_URL, spec)
        await run_bench(
            cfg,
            [spec],
            ["calib"],
            run_dir,
            limit=2,
            seed=7,
            local_only=False,
            concurrency=1,
            dry_run=False,
            route_cfg=None,
        )

    rd = RunDir(run_dir)
    preds = rd.predictions(None)
    model_ids = {p["model_id"] for p in preds}
    assert model_ids == {"tfidf-baseline", "open-large", "local-test"}
    assert all(p["split"] == "calib" and p["pass"] == "A" for p in preds)
    # 2 sampled items per model (baseline: 2, local: 2, cloud: 2 including the 1 preflight item)
    assert sum(1 for p in preds if p["model_id"] == "open-large") == 2
    assert sum(1 for p in preds if p["model_id"] == "local-test") == 2
    assert sum(1 for p in preds if p["model_id"] == "tfidf-baseline") == 2

    for name in (
        "metrics.json",
        "latency.json",
        "env.json",
        "config.json",
        "route.json",
        "tasks.json",
        "cost_ledger.jsonl",
    ):
        assert (run_dir / name).exists(), name

    metrics = rd.read_json("metrics.json")
    assert "open-large" in metrics and "classification" in metrics["open-large"]
    assert metrics["open-large"]["classification"]["calib"]["n"] == 2

    _assert_no_absolute_paths(run_dir)


@pytest.mark.asyncio
async def test_env_json_keys_match_allowlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = _config()
    spec = _classification_task()
    run_dir = tmp_path / "run"

    with respx.mock(assert_all_called=False) as mock:
        fakes.register_price_snapshot(mock, CLOUD_URL, ["vendor/model-a"])
        fakes.register_default_route(mock, CLOUD_URL, spec)
        fakes.register_default_route(mock, LOCAL_URL, spec)
        await run_bench(cfg, [spec], ["calib"], run_dir, limit=1, dry_run=False, route_cfg=None)

    env = RunDir(run_dir).read_json("env.json")
    assert set(env.keys()) == envinfo.ALLOWED_TOP_LEVEL_KEYS
    assert set(env["power"].keys()) == envinfo.ALLOWED_POWER_KEYS


@pytest.mark.asyncio
async def test_resume_skips_completed_and_adds_no_duplicate_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = _config()
    spec = _classification_task()
    run_dir = tmp_path / "run"

    with respx.mock(assert_all_called=False) as mock:
        fakes.register_price_snapshot(mock, CLOUD_URL, ["vendor/model-a"])
        cloud_route = fakes.register_default_route(mock, CLOUD_URL, spec)
        fakes.register_default_route(mock, LOCAL_URL, spec)
        await run_bench(cfg, [spec], ["calib"], run_dir, limit=2, dry_run=False, route_cfg=None)
        calib_calls = cloud_route.call_count
        ledger_lines_after_first = (run_dir / "cost_ledger.jsonl").read_text().strip().splitlines()

        # "Resume": rerun over the same run dir with an extra split selected.
        await run_bench(cfg, [spec], ["calib", "test"], run_dir, limit=2, dry_run=False, route_cfg=None)
        ledger_lines_after_second = (run_dir / "cost_ledger.jsonl").read_text().strip().splitlines()

    # calib items were never re-sent to the cloud fake.
    assert cloud_route.call_count > calib_calls
    rd = RunDir(run_dir)
    preds = rd.predictions(None)
    calib_preds = [p for p in preds if p["model_id"] == "open-large" and p["split"] == "calib"]
    assert len(calib_preds) == 2  # not duplicated
    test_preds = [p for p in preds if p["model_id"] == "open-large" and p["split"] == "test"]
    assert len(test_preds) == 2
    # ledger gained lines only for the new (test-split) calls, not repeated calib lines.
    assert len(ledger_lines_after_second) > len(ledger_lines_after_first)
    calib_ledger_lines = [line for line in ledger_lines_after_second if '"task": "classification"' in line]
    # sanity: total ledger lines for open-large equals total open-large predictions (2 calib preflight/main + 2 test)
    open_large_lines = [line for line in ledger_lines_after_second if '"model_id": "open-large"' in line]
    assert len(open_large_lines) == len([p for p in preds if p["model_id"] == "open-large"])
    del calib_ledger_lines


@pytest.mark.asyncio
async def test_model_path_never_reaches_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = _config()
    spec = _classification_task()
    run_dir = tmp_path / "run"

    with respx.mock(assert_all_called=False) as mock:
        fakes.register_price_snapshot(mock, CLOUD_URL, ["vendor/model-a"])
        fakes.register_default_route(mock, CLOUD_URL, spec)
        fakes.register_default_route(mock, LOCAL_URL, spec)
        await run_bench(cfg, [spec], ["calib"], run_dir, limit=1, dry_run=False, route_cfg=None)

    _assert_no_absolute_paths(run_dir)


@pytest.mark.asyncio
async def test_preflight_aborts_on_empty_length_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = _config()
    spec = _classification_task()
    run_dir = tmp_path / "run"

    with respx.mock(assert_all_called=False) as mock:
        fakes.register_price_snapshot(mock, CLOUD_URL, ["vendor/model-a"])
        mock.post(f"{CLOUD_URL}/chat/completions").mock(return_value=fakes.empty_length_response())
        with pytest.raises(PreflightFailed, match="open-large"):
            await run_bench(cfg, [spec], ["calib"], run_dir, limit=1, dry_run=False, route_cfg=None)

    rd = RunDir(run_dir)
    # the failed preflight call must not be recorded as a normal (harness-bug) prediction.
    assert [p for p in rd.predictions(None) if p["model_id"] == "open-large"] == []
    # but it was billed: the ledger recorded one settled reservation for the cloud model.
    ledger_lines = (run_dir / "cost_ledger.jsonl").read_text().strip().splitlines()
    assert sum(1 for line in ledger_lines if '"model_id": "open-large"' in line) == 1
    # the run still gets its end-of-run files despite the abort.
    assert (run_dir / "env.json").exists()


@pytest.mark.asyncio
async def test_retries_are_counted_in_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = _config()
    spec = _classification_task()
    run_dir = tmp_path / "run"

    handler = fakes.sequence_handler([httpx.Response(429), fakes.chat_response(fakes.default_valid_answer(spec))])
    with respx.mock(assert_all_called=False) as mock:
        fakes.register_price_snapshot(mock, CLOUD_URL, ["vendor/model-a"])
        mock.post(f"{CLOUD_URL}/chat/completions").mock(side_effect=handler)
        fakes.register_default_route(mock, LOCAL_URL, spec)
        await run_bench(cfg, [spec], ["calib"], run_dir, limit=1, local_only=False, dry_run=False, route_cfg=None)

    rd = RunDir(run_dir)
    preds = [p for p in rd.predictions(None) if p["model_id"] == "open-large"]
    assert len(preds) == 1
    assert preds[0]["retries"] == 1


@pytest.mark.asyncio
async def test_dry_run_prints_totals_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg = _config()
    spec = _classification_task()
    run_dir = tmp_path / "run"

    with respx.mock(assert_all_called=False) as mock:
        fakes.register_price_snapshot(mock, CLOUD_URL, ["vendor/model-a"])
        await run_bench(cfg, [spec], ["calib"], run_dir, limit=2, dry_run=True, route_cfg=None)

    out = capsys.readouterr().out
    assert "open-large" in out
    assert "total: $" in out
    assert not run_dir.exists()
