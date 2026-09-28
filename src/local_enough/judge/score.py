"""``judge score``: judge every non-baseline model's Pass A summarisation predictions in a run.

Runs after ``judge calibrate`` (its ``judge_calibration.json`` selects the judge model) and before
``report``. An invalid or empty model output is never sent to the judge; it is recorded as a failed item
with a null verdict, at no cost.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

from local_enough.bench.ledger import Ledger, usage_meta
from local_enough.bench.rundir import RunDir
from local_enough.config import Config
from local_enough.judge import judge as judge_core
from local_enough.judge import keytoken, pricing
from local_enough.providers.openai_compat import ChatClient
from local_enough.tasks import registry
from local_enough.tasks.base import Item, TaskSpec


class MissingCalibration(RuntimeError):
    """Raised when ``judge score`` runs before ``judge calibrate`` has written judge_calibration.json."""


def _summarisation_spec() -> TaskSpec:
    for spec in registry.bundled_tasks():
        if spec.kind == "summarisation":
            return spec
    raise RuntimeError("bundled summarisation task not found")


def _empty_record(pred: dict[str, Any], judge_model: str, n_facts: int) -> dict[str, Any]:
    return {
        "model_id": pred["model_id"],
        "split": pred["split"],
        "item_id": pred["item_id"],
        "judge_model": judge_model,
        "verdict": None,
        "passed": False,
        "fact_recall": 0.0 if n_facts else math.nan,
        "within_limit": False,
        "key_token": [],
    }


def _key_token_rows(item: Item, summary: str, verdict: dict[str, Any] | None) -> list[dict[str, Any]]:
    by_index = {f["index"]: f for f in verdict["facts"]} if verdict is not None else {}
    rows = []
    for fact in item["required_facts"]:
        idx = fact["index"]
        check = keytoken.check_fact(fact.get("key_tokens", {}), summary)
        judge_entry = by_index.get(idx)
        judge_value = bool(judge_entry["present"] and judge_entry["correct"]) if judge_entry else None
        agree = (check == judge_value) if judge_value is not None else None
        rows.append({"index": idx, "check": check, "judge": judge_value, "agree": agree})
    return rows


async def run_score(run_dir: str | Path, cfg: Config) -> list[dict[str, Any]]:
    """Judge every non-baseline model's summarisation predictions (both splits) not already scored."""
    if cfg.judge is None:
        raise ValueError("config.yaml has no 'judge' section")
    run = RunDir(run_dir)
    calibration = run.read_json("judge_calibration.json")
    if calibration is None:
        raise MissingCalibration(
            f"{run.path}/judge_calibration.json not found; run 'local-enough judge calibrate' first"
        )
    selected = str(calibration["selected"])
    price_snapshot = run.read_json("price_snapshot.json") or {}
    price = (price_snapshot.get("models") or {}).get(selected, {})

    spec = _summarisation_spec()
    items_by_split: dict[str, dict[str, Item]] = {
        "calib": {str(r["id"]): r for r in registry.load_items(spec, "calib")},
        "test": {str(r["id"]): r for r in registry.load_items(spec, "test")},
    }

    baseline_ids = {m.id for m in cfg.models if m.kind == "baseline"}
    predictions = [
        r for r in run.predictions(pass_="A") if r["task"] == "summarisation" and r["model_id"] not in baseline_ids
    ]
    existing = {(r["model_id"], r["split"], str(r["item_id"])) for r in run.iter_records("judge_scores.jsonl.gz")}
    todo = [p for p in predictions if (p["model_id"], p["split"], str(p["item_id"])) not in existing]

    ledger = Ledger(cfg.project, cfg.budget_usd, cfg.budget_warn_usd, run_dir=run.path)
    endpoint = pricing.judge_endpoint(cfg.judge, selected)
    max_tokens = cfg.judge.max_tokens
    semaphore = asyncio.Semaphore(max(1, cfg.judge.concurrency))
    new_records: list[dict[str, Any]] = []

    async def _score_one(pred: dict[str, Any], client: ChatClient) -> None:
        split_items = items_by_split.get(pred["split"])
        item = split_items.get(str(pred["item_id"])) if split_items is not None else None
        if item is None:
            return
        n_facts = len(item["required_facts"])
        max_words = item.get("max_words") or spec.max_words or 120

        if not pred.get("valid") or not pred.get("content"):
            record = _empty_record(pred, selected, n_facts)
            new_records.append(record)
            run.append_records("judge_scores.jsonl.gz", [record])
            return

        summary = str(pred["content"])
        messages = judge_core.render_judge_messages(item["text"], item["required_facts"], summary)
        async with semaphore:
            estimate = pricing.reserve_estimate(messages, max_tokens, endpoint.reasoning_allowance_tokens, price)
            meta = {
                "model_id": endpoint.model_id,
                "command": "judge score",
                "task": "summarisation",
                "item_id": pred["item_id"],
            }
            async with ledger.reserve(estimate, meta) as reservation:
                result = await client.complete(endpoint, messages, max_tokens)
                actual, source = pricing.settle_cost(result, price)
                reservation.settle(actual, source, usage_meta(result))

        verdict = judge_core.parse_verdict(result.text, n_facts)
        words = len(summary.split())
        passed = judge_core.decide_pass(verdict, n_facts, words, max_words)
        present_correct = judge_core.present_and_correct_count(verdict) if verdict is not None else 0
        fact_recall = present_correct / n_facts if n_facts else math.nan

        record = {
            "model_id": pred["model_id"],
            "split": pred["split"],
            "item_id": pred["item_id"],
            "judge_model": selected,
            "verdict": verdict,
            "passed": passed,
            "fact_recall": fact_recall,
            "within_limit": words <= max_words,
            "key_token": _key_token_rows(item, summary, verdict),
        }
        new_records.append(record)
        run.append_records("judge_scores.jsonl.gz", [record])

    async with ChatClient() as client:
        await asyncio.gather(*(_score_one(pred, client) for pred in todo))

    return new_records
