"""Meeting transcript to an action summary, judged against required facts.

Without a judge verdict (``extra=None``), ``pass_rate`` is NaN (reports show "not judged"); ``fact_recall``
still uses a deterministic key-token presence check so reports have a signal before judging runs. With
``extra={"judge": {"facts": [...], "unsupported_claims": [...]}}`` (M3's judge output), the code pass rule
applies: at least ``ceil(0.8 * n)`` facts present and correct, zero unsupported claims, within the word limit.
"""

from __future__ import annotations

import math
from typing import Any

from local_enough.config import TaskKind
from local_enough.tasks import parsing, registry
from local_enough.tasks.base import Item, ItemScore, Message, Parsed, TaskSpec

kind: TaskKind = "summarisation"
primary_metric = "pass_rate"


def render_messages(spec: TaskSpec, item: Item) -> list[Message]:
    max_words = item.get("max_words") or spec.max_words or 120
    return registry.render_prompt("summarisation", text=item["text"], max_words=max_words)


def user_input(spec: TaskSpec, item: Item) -> str:
    return str(item["text"])


def item_from_input(spec: TaskSpec, raw: str) -> Item:
    return {"id": "router", "text": raw, "required_facts": [], "max_words": spec.max_words or 120}


def parse(spec: TaskSpec, raw_text: str) -> Parsed:
    cleaned = parsing.strip_wrappers(raw_text)
    if not cleaned:
        return Parsed(ok=False, error="empty summary")
    return Parsed(ok=True, value=cleaned, content=cleaned)


def score(spec: TaskSpec, item: Item, parsed: Parsed, extra: dict[str, Any] | None = None) -> ItemScore:
    required_facts = item.get("required_facts", [])
    max_words = item.get("max_words") or spec.max_words or 120
    if not parsed.ok:
        return ItemScore(
            item_id=item["id"],
            valid=False,
            stats={
                "n_facts": len(required_facts),
                "fact_present_count": 0,
                "within_word_limit": False,
                "judged": False,
                "judged_pass": None,
            },
        )
    summary = str(parsed.value)
    within = len(summary.split()) <= max_words
    lowered = summary.casefold()
    present_count = 0
    for fact in required_facts:
        values = [str(v) for v in fact.get("key_tokens", {}).values() if v is not None]
        if values and all(v.casefold() in lowered for v in values):
            present_count += 1
    stats: dict[str, Any] = {
        "n_facts": len(required_facts),
        "fact_present_count": present_count,
        "within_word_limit": within,
        "judged": False,
        "judged_pass": None,
    }
    judge = (extra or {}).get("judge")
    if judge is not None:
        n = len(required_facts)
        present_correct = sum(1 for fv in judge.get("facts", []) if fv.get("present") and fv.get("correct"))
        unsupported = len(judge.get("unsupported_claims", []))
        stats["judged"] = True
        stats["judged_pass"] = (n == 0 or present_correct >= math.ceil(0.8 * n)) and unsupported == 0 and within
    return ItemScore(item_id=item["id"], valid=True, stats=stats)


def aggregate(spec: TaskSpec, scores: list[ItemScore]) -> dict[str, float]:
    n = len(scores)
    if n == 0:
        keys = ["primary", "pass_rate", "fact_recall", "within_word_limit", "invalid_output_rate"]
        return dict.fromkeys(keys, math.nan)
    invalid_rate = sum(1 for s in scores if not s.valid) / n
    within_rate = sum(1 for s in scores if s.stats.get("within_word_limit")) / n
    fact_total = sum(s.stats.get("n_facts", 0) for s in scores)
    fact_present = sum(s.stats.get("fact_present_count", 0) for s in scores)
    fact_recall = fact_present / fact_total if fact_total else math.nan
    judged = [s for s in scores if s.stats.get("judged")]
    pass_rate = (sum(1 for s in judged if s.stats.get("judged_pass")) / len(judged)) if judged else math.nan
    return {
        "primary": pass_rate,
        "pass_rate": pass_rate,
        "fact_recall": fact_recall,
        "within_word_limit": within_rate,
        "invalid_output_rate": invalid_rate,
    }


def validate_item(spec: TaskSpec, item: Item) -> list[str]:
    errors = []
    if "id" not in item:
        errors.append("missing 'id'")
    if not isinstance(item.get("text"), str):
        errors.append("missing or invalid 'text'")
    facts = item.get("required_facts")
    if not isinstance(facts, list):
        errors.append("missing or invalid 'required_facts'")
    else:
        for i, f in enumerate(facts):
            if not isinstance(f, dict) or not {"fact", "kind", "key_tokens"} <= f.keys():
                errors.append(f"required_facts[{i}] missing fact/kind/key_tokens")
    if not isinstance(item.get("max_words"), int):
        errors.append("missing or invalid 'max_words'")
    return errors


class _Module:
    kind = kind
    primary_metric = primary_metric
    render_messages = staticmethod(render_messages)
    user_input = staticmethod(user_input)
    item_from_input = staticmethod(item_from_input)
    parse = staticmethod(parse)
    score = staticmethod(score)
    aggregate = staticmethod(aggregate)
    validate_item = staticmethod(validate_item)


MODULE = _Module()
