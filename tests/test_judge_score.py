"""``judge score`` end-to-end on a tiny fake run dir written with RunDir."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from local_enough.bench.rundir import RunDir
from local_enough.config import BaselineModel, CloudModel, Config, JudgeConfig
from local_enough.judge import score as score_mod

BASE_URL = "https://judge.test/api/v1"
CANDIDATE = "fake/judge-1"
CALIB_ITEM = "summarisation-calib-0000"  # bundled summarisation calib[0]: 5 required facts, max_words 120
TEST_ITEM = "summarisation-test-0000"  # bundled summarisation test[0]: 6 required facts

FAITHFUL_VERDICT_5 = {
    "facts": [{"index": i, "present": True, "correct": True} for i in range(5)],
    "unsupported_claims": [],
}


def _cfg() -> Config:
    return Config(
        project="score-test",
        budget_usd=5.0,
        models=[
            CloudModel(id="cloud-a", model="vendor/cloud-a", api_key_env="OPENROUTER_API_KEY"),
            BaselineModel(id="baseline-x", task="entity_matching"),
        ],
        judge=JudgeConfig(
            candidates=[CANDIDATE], base_url=BASE_URL, api_key_env="OPENROUTER_API_KEY", concurrency=4, max_tokens=300
        ),
    )


def _seed_run(run_dir: Path) -> RunDir:
    run = RunDir(run_dir)
    run.write_json(
        "judge_calibration.json",
        {
            "selected": CANDIDATE,
            "target_balanced_accuracy": 0.90,
            "meets_target": True,
            "candidates": {},
            "holdout": {
                "tpr": 1.0,
                "tnr": 1.0,
                "balanced_accuracy": 1.0,
                "kappa": 1.0,
                "n": 0,
                "invalid_rate": 0.0,
                "per_item": [],
            },
            "judge_calib_n": 0,
            "created_utc": "2026-09-28T00:00:00Z",
        },
    )
    run.write_json(
        "price_snapshot.json",
        {
            "taken_at_utc": "2026-09-28T00:00:00Z",
            "source_url": f"{BASE_URL}/models",
            "models": {CANDIDATE: {"prompt": 0.0000004, "completion": 0.0000016}},
        },
    )
    run.append_records(
        "predictions.jsonl.gz",
        [
            {
                "model_id": "cloud-a",
                "task": "summarisation",
                "split": "calib",
                "item_id": CALIB_ITEM,
                "pass": "A",
                "raw": "A short faithful summary text.",
                "content": "A short faithful summary text.",
                "valid": True,
            },
            {
                "model_id": "cloud-a",
                "task": "summarisation",
                "split": "test",
                "item_id": TEST_ITEM,
                "pass": "A",
                "raw": "not json at all",
                "content": None,
                "valid": False,
                "error": "invalid_output",
            },
            {
                "model_id": "baseline-x",
                "task": "summarisation",
                "split": "calib",
                "item_id": CALIB_ITEM,
                "pass": "A",
                "raw": "baseline output",
                "content": "baseline output",
                "valid": True,
            },
        ],
    )
    return run


@pytest.mark.asyncio
async def test_score_missing_calibration_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    with pytest.raises(score_mod.MissingCalibration):
        await score_mod.run_score(tmp_path, _cfg())


@pytest.mark.asyncio
async def test_score_end_to_end_skips_baseline_and_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    run = _seed_run(tmp_path)
    cfg = _cfg()

    payload = {
        "choices": [{"message": {"content": json.dumps(FAITHFUL_VERDICT_5)}}],
        "usage": {"cost": 0.0003, "prompt_tokens": 500, "completion_tokens": 80},
    }
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(f"{BASE_URL}/chat/completions").mock(return_value=httpx.Response(200, json=payload))
        records = await score_mod.run_score(tmp_path, cfg)

    assert route.call_count == 1  # only the valid cloud-a/calib item was sent; the invalid one was not
    assert len(records) == 2
    by_key = {(r["model_id"], r["split"], r["item_id"]): r for r in records}

    scored = by_key[("cloud-a", "calib", CALIB_ITEM)]
    assert scored["judge_model"] == CANDIDATE
    assert scored["passed"] is True
    assert scored["fact_recall"] == pytest.approx(1.0)
    assert scored["within_limit"] is True
    assert len(scored["key_token"]) == 5
    assert all(row["judge"] is True and row["agree"] is not None for row in scored["key_token"])

    failed = by_key[("cloud-a", "test", TEST_ITEM)]
    assert failed["verdict"] is None
    assert failed["passed"] is False
    assert failed["fact_recall"] == pytest.approx(0.0)
    assert failed["within_limit"] is False
    assert failed["key_token"] == []

    assert ("baseline-x", "calib", CALIB_ITEM) not in by_key

    stored = list(run.iter_records("judge_scores.jsonl.gz"))
    assert {(r["model_id"], r["split"], r["item_id"]) for r in stored} == set(by_key)

    ledger_lines = (tmp_path / "cost_ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(ledger_lines) == 1
    assert json.loads(ledger_lines[0])["command"] == "judge score"


@pytest.mark.asyncio
async def test_score_resume_skips_already_scored_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    _seed_run(tmp_path)
    cfg = _cfg()
    payload = {
        "choices": [{"message": {"content": json.dumps(FAITHFUL_VERDICT_5)}}],
        "usage": {"cost": 0.0003},
    }

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(f"{BASE_URL}/chat/completions").mock(return_value=httpx.Response(200, json=payload))
        first = await score_mod.run_score(tmp_path, cfg)
        assert route.call_count == 1
        assert len(first) == 2

        second = await score_mod.run_score(tmp_path, cfg)
        assert route.call_count == 1  # no new HTTP calls: both items were already scored
        assert second == []
