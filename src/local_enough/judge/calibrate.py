"""``judge calibrate``: score every judge candidate on judge-calib, pick the best, measure it on judge-holdout.

Reads the bundled summarisation task's ``calib.jsonl`` (transcripts for judge-calib) and
``judge_holdout_transcripts.jsonl`` (transcripts for judge-holdout) to attach each construction-labelled
summary to its transcript, required facts and word limit.
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from local_enough.bench.ledger import Ledger
from local_enough.bench.prices import snapshot_prices
from local_enough.bench.rundir import RunDir
from local_enough.config import Config, JudgeConfig
from local_enough.judge import judge as judge_core
from local_enough.judge import pricing
from local_enough.judge.stats import tpr_tnr
from local_enough.providers.openai_compat import ChatClient, Endpoint
from local_enough.stats import cohen_kappa
from local_enough.tasks import registry
from local_enough.tasks.base import TaskSpec

TARGET_BALANCED_ACCURACY = 0.90


def _summarisation_spec() -> TaskSpec:
    for spec in registry.bundled_tasks():
        if spec.kind == "summarisation":
            return spec
    raise RuntimeError("bundled summarisation task not found")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _judge_items(records: list[dict[str, Any]], transcripts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach transcript text, required facts and word limit to each judge-set construction-labelled summary."""
    out = []
    for rec in records:
        transcript = transcripts[rec["transcript_id"]]
        out.append(
            {
                "id": rec["id"],
                "summary": rec["summary"],
                "label_pass": bool(rec["label_pass"]),
                "required_facts": transcript["required_facts"],
                "max_words": transcript["max_words"],
                "text": transcript["text"],
            }
        )
    return out


async def _judge_one(
    client: ChatClient,
    ledger: Ledger,
    endpoint: Endpoint,
    item: dict[str, Any],
    max_tokens: int,
    price: dict[str, Any],
    command: str,
) -> dict[str, Any]:
    messages = judge_core.render_judge_messages(item["text"], item["required_facts"], item["summary"])
    estimate = pricing.reserve_estimate(messages, max_tokens, endpoint.reasoning_allowance_tokens, price)
    meta = {"model_id": endpoint.model_id, "command": command, "task": "summarisation", "item_id": item["id"]}
    async with ledger.reserve(estimate, meta) as reservation:
        result = await client.complete(endpoint, messages, max_tokens)
        actual, source = pricing.settle_cost(result, price)
        usage = {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens}
        reservation.settle(actual, source, usage)

    n_facts = len(item["required_facts"])
    verdict = judge_core.parse_verdict(result.text, n_facts)
    words = len(item["summary"].split())
    predicted_pass = judge_core.decide_pass(verdict, n_facts, words, item["max_words"])
    return {
        "label_pass": item["label_pass"],
        "predicted_pass": predicted_pass,
        "invalid": verdict is None,
        "cost_usd": actual,
        "raw": result.text,
    }


async def _judge_set(
    run: RunDir,
    client: ChatClient,
    ledger: Ledger,
    endpoint: Endpoint,
    items: list[dict[str, Any]],
    set_name: str,
    max_tokens: int,
    price: dict[str, Any],
    concurrency: int,
    command: str,
) -> list[dict[str, Any]]:
    """Judge every item, reusing any cached raw output and appending only newly-judged ones."""
    cached = {(r["judge_model"], r["set"], r["item_id"]): r for r in run.iter_records("judge_raw.jsonl.gz")}
    results: list[dict[str, Any] | None] = [None] * len(items)
    todo: list[tuple[int, dict[str, Any]]] = []
    for i, item in enumerate(items):
        key = (endpoint.model_id, set_name, item["id"])
        cached_record = cached.get(key)
        if cached_record is not None:
            results[i] = cached_record
        else:
            todo.append((i, item))

    semaphore = asyncio.Semaphore(max(1, concurrency))
    new_records: list[dict[str, Any]] = []

    async def _run(i: int, item: dict[str, Any]) -> None:
        async with semaphore:
            outcome = await _judge_one(client, ledger, endpoint, item, max_tokens, price, command)
        record = {"judge_model": endpoint.model_id, "set": set_name, "item_id": item["id"], **outcome}
        results[i] = record
        new_records.append(record)

    await asyncio.gather(*(_run(i, item) for i, item in todo))
    if new_records:
        run.append_records("judge_raw.jsonl.gz", new_records)
    return [r for r in results if r is not None]


def _metrics(records: list[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    labels = [bool(r["label_pass"]) for r in records]
    preds = [bool(r["predicted_pass"]) for r in records]
    tpr, tnr = tpr_tnr(labels, preds)
    balanced_accuracy = (tpr + tnr) / 2 if not (math.isnan(tpr) or math.isnan(tnr)) else math.nan
    n = len(records)
    invalid_rate = sum(1 for r in records if r["invalid"]) / n if n else math.nan
    metrics = {
        "tpr": tpr,
        "tnr": tnr,
        "balanced_accuracy": balanced_accuracy,
        "kappa": cohen_kappa(labels, preds),
        "n": n,
    }
    return metrics, invalid_rate


def _snapshot_price(price_map: dict[str, Any], candidate: str) -> float:
    entry = price_map.get("models", {}).get(candidate, {})
    return (entry.get("prompt") or 0.0) + (entry.get("completion") or 0.0)


def _select_candidate(candidates_out: dict[str, Any], price_map: dict[str, Any]) -> str:
    def sort_key(candidate: str) -> tuple[float, float]:
        balanced_accuracy = candidates_out[candidate]["calib"]["balanced_accuracy"]
        rank = -balanced_accuracy if not math.isnan(balanced_accuracy) else math.inf
        return (rank, _snapshot_price(price_map, candidate))

    return min(candidates_out, key=sort_key)


async def run_calibration(run_dir: str | Path, cfg: Config) -> dict[str, Any]:
    """Calibrate every candidate, select the best by balanced accuracy, measure it on judge-holdout."""
    if cfg.judge is None:
        raise ValueError("config.yaml has no 'judge' section")
    judge_cfg: JudgeConfig = cfg.judge
    run = RunDir(run_dir)
    spec = _summarisation_spec()
    assert spec.root is not None

    calib_transcripts = {str(r["id"]): r for r in registry.load_items(spec, "calib")}
    holdout_transcripts = {str(r["id"]): r for r in _load_jsonl(spec.root / "judge_holdout_transcripts.jsonl")}
    calib_items = _judge_items(_load_jsonl(spec.root / "judge_calib.jsonl"), calib_transcripts)
    holdout_items = _judge_items(_load_jsonl(spec.root / "judge_holdout.jsonl"), holdout_transcripts)

    price_map = await snapshot_prices(judge_cfg.base_url, {c: c for c in judge_cfg.candidates})
    run.merge_json("price_snapshot.json", price_map)

    ledger = Ledger(cfg.project, cfg.budget_usd, cfg.budget_warn_usd, run_dir=run.path)

    candidates_out: dict[str, Any] = {}
    async with ChatClient() as client:
        for candidate in judge_cfg.candidates:
            endpoint = pricing.judge_endpoint(judge_cfg, candidate)
            price = price_map["models"].get(candidate, {})
            records = await _judge_set(
                run,
                client,
                ledger,
                endpoint,
                calib_items,
                "calib",
                judge_cfg.max_tokens,
                price,
                judge_cfg.concurrency,
                "judge calibrate",
            )
            metrics, invalid_rate = _metrics(records)
            candidates_out[candidate] = {
                "calib": metrics,
                "invalid_rate": invalid_rate,
                "cost_usd": sum(r["cost_usd"] for r in records),
            }

        selected = _select_candidate(candidates_out, price_map)
        winner_endpoint = pricing.judge_endpoint(judge_cfg, selected)
        winner_price = price_map["models"].get(selected, {})
        holdout_records = await _judge_set(
            run,
            client,
            ledger,
            winner_endpoint,
            holdout_items,
            "holdout",
            judge_cfg.max_tokens,
            winner_price,
            judge_cfg.concurrency,
            "judge calibrate",
        )

    holdout_metrics, holdout_invalid_rate = _metrics(holdout_records)
    holdout_out = {
        **holdout_metrics,
        "invalid_rate": holdout_invalid_rate,
        "per_item": [
            {"id": r["item_id"], "label_pass": r["label_pass"], "predicted_pass": r["predicted_pass"]}
            for r in holdout_records
        ],
    }
    meets_target = (
        not math.isnan(holdout_metrics["balanced_accuracy"])
        and holdout_metrics["balanced_accuracy"] >= TARGET_BALANCED_ACCURACY
    )
    result = {
        "selected": selected,
        "target_balanced_accuracy": TARGET_BALANCED_ACCURACY,
        "meets_target": meets_target,
        "candidates": candidates_out,
        "holdout": holdout_out,
        "judge_calib_n": len(calib_items),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    run.write_json("judge_calibration.json", result)
    return result
