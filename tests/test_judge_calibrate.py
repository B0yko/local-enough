"""End-to-end judge calibration against a fake judge derived from the construction labels."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from local_enough.bench.rundir import RunDir
from local_enough.config import Config, JudgeConfig
from local_enough.judge import calibrate
from local_enough.judge.stats import tpr_tnr
from local_enough.providers.openai_compat import ChatClient, Endpoint
from local_enough.stats import cohen_kappa
from local_enough.tasks import registry

BASE_URL = "https://judge.test/api/v1"
CANDIDATE = "fake/judge-1"
CANDIDATE_B = "fake/judge-2"
_SUMMARY_RE = re.compile(r"Summary under review:\n(.*)\n\nReturn the JSON verdict now\.", re.DOTALL)


def _dataset_root() -> Path:
    spec = next(s for s in registry.bundled_tasks() if s.kind == "summarisation")
    assert spec.root is not None
    return spec.root


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _corrupted_meta(records: list[dict[str, Any]], corrupt_every: int) -> dict[str, dict[str, Any]]:
    """Map summary text -> {record, corrupted}; every ``corrupt_every``-th item (by file order) is flipped."""
    return {r["summary"]: {"record": r, "corrupted": i % corrupt_every == 0} for i, r in enumerate(records)}


def _expected_predicted(record: dict[str, Any], corrupted: bool) -> bool:
    """What the fake judge below is built to make ``decide_pass`` produce for this item."""
    return (not record["label_pass"]) if corrupted else bool(record["label_pass"])


def _fake_judge_responder(meta: dict[str, dict[str, Any]]) -> Any:
    """A fake judge: returns the exact construction verdict (predicted == label_pass), except that every
    ``corrupt_every``-th item is answered with a verdict engineered to flip the decision the other way --
    zero present-and-correct facts for a should-pass item, or every fact present-and-correct with no
    unsupported claims for a should-fail item. Matched to items in a real 600-1000 word transcript by the
    exact (unique) summary text sent in the request, since the request carries no item id.
    """

    def responder(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][-1]["content"]
        match = _SUMMARY_RE.search(content)
        assert match is not None
        info = meta[match.group(1)]
        record, corrupted = info["record"], info["corrupted"]
        if corrupted:
            present_correct = not record["label_pass"]
            facts = [
                {"index": f["index"], "present": present_correct, "correct": present_correct} for f in record["facts"]
            ]
            claims: list[str] = []
        else:
            facts = [{"index": f["index"], "present": f["present"], "correct": f["correct"]} for f in record["facts"]]
            claims = ["unsupported claim"] * int(record["unsupported_claims"])
        payload = {
            "choices": [{"message": {"content": json.dumps({"facts": facts, "unsupported_claims": claims})}}],
            "usage": {"cost": 0.0002, "prompt_tokens": 400, "completion_tokens": 60},
        }
        return httpx.Response(200, json=payload)

    return responder


@pytest.mark.asyncio
async def test_calibrate_end_to_end_tpr_tnr_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    root = _dataset_root()
    calib_records = _load_jsonl(root / "judge_calib.jsonl")
    holdout_records = _load_jsonl(root / "judge_holdout.jsonl")
    meta = {**_corrupted_meta(calib_records, 4), **_corrupted_meta(holdout_records, 5)}
    assert len(meta) == len(calib_records) + len(holdout_records)  # summaries are unique across both sets

    cfg = Config(
        project="judge-test",
        budget_usd=5.0,
        models=[],
        judge=JudgeConfig(
            candidates=[CANDIDATE], base_url=BASE_URL, api_key_env="OPENROUTER_API_KEY", concurrency=16, max_tokens=300
        ),
    )
    models_payload = {
        "data": [
            {"id": CANDIDATE, "pricing": {"prompt": "0.0000004", "completion": "0.0000016"}, "supported_parameters": []}
        ]
    }

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(200, json=models_payload))
        mock.post(f"{BASE_URL}/chat/completions").mock(side_effect=_fake_judge_responder(meta))
        result = await calibrate.run_calibration(tmp_path, cfg)

    calib_labels = [bool(r["label_pass"]) for r in calib_records]
    calib_preds = [_expected_predicted(r, i % 4 == 0) for i, r in enumerate(calib_records)]
    exp_tpr, exp_tnr = tpr_tnr(calib_labels, calib_preds)
    exp_kappa = cohen_kappa(calib_labels, calib_preds)

    cand = result["candidates"][CANDIDATE]["calib"]
    assert cand["tpr"] == pytest.approx(exp_tpr)
    assert cand["tnr"] == pytest.approx(exp_tnr)
    assert cand["balanced_accuracy"] == pytest.approx((exp_tpr + exp_tnr) / 2)
    assert cand["kappa"] == pytest.approx(exp_kappa)
    assert cand["n"] == 200
    assert result["candidates"][CANDIDATE]["invalid_rate"] == 0.0
    assert result["candidates"][CANDIDATE]["cost_usd"] == pytest.approx(0.0002 * 200)

    holdout_labels = [bool(r["label_pass"]) for r in holdout_records]
    holdout_preds = [_expected_predicted(r, i % 5 == 0) for i, r in enumerate(holdout_records)]
    exp_h_tpr, exp_h_tnr = tpr_tnr(holdout_labels, holdout_preds)

    assert result["selected"] == CANDIDATE
    assert result["holdout"]["tpr"] == pytest.approx(exp_h_tpr)
    assert result["holdout"]["tnr"] == pytest.approx(exp_h_tnr)
    assert result["holdout"]["n"] == 200
    assert len(result["holdout"]["per_item"]) == 200
    assert result["judge_calib_n"] == 200
    assert result["target_balanced_accuracy"] == 0.90
    assert result["meets_target"] == (((exp_h_tpr + exp_h_tnr) / 2) >= 0.90)
    assert {(p["id"], p["label_pass"]) for p in result["holdout"]["per_item"]} == {
        (r["id"], bool(r["label_pass"])) for r in holdout_records
    }

    run = RunDir(tmp_path)
    assert run.read_json("judge_calibration.json") == result
    snapshot = run.read_json("price_snapshot.json")
    assert snapshot["models"][CANDIDATE]["prompt"] == pytest.approx(0.0000004)
    assert len(list(run.iter_records("judge_raw.jsonl.gz"))) == 400


def test_select_candidate_highest_balanced_accuracy() -> None:
    candidates_out = {
        "a": {"calib": {"balanced_accuracy": 0.80}},
        "b": {"calib": {"balanced_accuracy": 0.92}},
        "c": {"calib": {"balanced_accuracy": 0.85}},
    }
    price_map = {"models": {"a": {"prompt": 0.0, "completion": 0.0}, "b": {}, "c": {}}}
    assert calibrate._select_candidate(candidates_out, price_map) == "b"


def test_select_candidate_tie_breaks_on_lower_snapshot_price() -> None:
    candidates_out = {
        "expensive": {"calib": {"balanced_accuracy": 0.90}},
        "cheap": {"calib": {"balanced_accuracy": 0.90}},
    }
    price_map = {
        "models": {
            "expensive": {"prompt": 0.000002, "completion": 0.000008},
            "cheap": {"prompt": 0.0000001, "completion": 0.0000004},
        }
    }
    assert calibrate._select_candidate(candidates_out, price_map) == "cheap"


def test_select_candidate_nan_balanced_accuracy_loses_to_any_real_score() -> None:
    candidates_out = {
        "nan": {"calib": {"balanced_accuracy": float("nan")}},
        "real": {"calib": {"balanced_accuracy": 0.5}},
    }
    price_map = {"models": {"nan": {}, "real": {}}}
    assert calibrate._select_candidate(candidates_out, price_map) == "real"


class _FakeLedgerReservation:
    def __init__(self) -> None:
        self.settled: list[tuple[float, str]] = []

    async def __aenter__(self) -> _FakeLedgerReservation:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    def settle(self, actual_usd: float, source: str, usage: dict[str, Any] | None = None) -> None:
        self.settled.append((actual_usd, source))


class _FakeLedger:
    """Just enough of Ledger.reserve's interface for a resume test with no real budget tracking."""

    def reserve(self, estimate_usd: float, meta: dict[str, Any]) -> _FakeLedgerReservation:
        return _FakeLedgerReservation()


@pytest.mark.asyncio
async def test_judge_set_resume_skips_cached_items(tmp_path: Path) -> None:
    run = RunDir(tmp_path)
    endpoint = Endpoint(
        model_id=CANDIDATE,
        base_url=BASE_URL,
        model_name=CANDIDATE,
        api_key="sk-test",
        kind="cloud",
        display_model=CANDIDATE,
    )
    items = [
        {
            "id": f"item-{i}",
            "text": "transcript text",
            "required_facts": [{"index": 0, "fact": "x", "kind": "amount", "key_tokens": {"amount": "1"}}],
            "max_words": 120,
            "summary": f"summary {i}",
            "label_pass": True,
        }
        for i in range(3)
    ]
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"facts": [{"index": 0, "present": True, "correct": True}], "unsupported_claims": []}
                    )
                }
            }
        ],
        "usage": {"cost": 0.0001},
    }
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{BASE_URL}/chat/completions").mock(return_value=httpx.Response(200, json=payload))
        async with ChatClient() as client:
            first = await calibrate._judge_set(
                run,
                client,
                _FakeLedger(),
                endpoint,
                items,
                "calib",
                100,
                {"prompt": 0.0, "completion": 0.0},
                4,
                "judge calibrate",
            )
            assert route.call_count == 3
            assert len(first) == 3
            assert all(r["predicted_pass"] for r in first)

            second = await calibrate._judge_set(
                run,
                client,
                _FakeLedger(),
                endpoint,
                items,
                "calib",
                100,
                {"prompt": 0.0, "completion": 0.0},
                4,
                "judge calibrate",
            )
            assert route.call_count == 3  # no new calls: everything was cached
            assert len(second) == 3
