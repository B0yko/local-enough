"""Banking customer-intent classification (also used by bring-your-own classifiers)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from local_enough.config import TaskKind
from local_enough.tasks import parsing, registry
from local_enough.tasks.base import Item, ItemScore, Message, Parsed, TaskSpec

kind: TaskKind = "classification"
primary_metric = "accuracy"


def render_messages(spec: TaskSpec, item: Item) -> list[Message]:
    return registry.render_prompt("classification", labels=spec.labels or [], text=item["text"])


def user_input(spec: TaskSpec, item: Item) -> str:
    return str(item["text"])


def item_from_input(spec: TaskSpec, raw: str) -> Item:
    return {"id": "router", "text": raw}


def parse(spec: TaskSpec, raw_text: str) -> Parsed:
    try:
        label = parsing.first_line_label(raw_text)
    except ValueError as exc:
        return Parsed(ok=False, error=str(exc))
    # Labels are compared without trailing punctuation on both sides: BANKING77's official label set has
    # "reverted_card_payment?", which a model may or may not reproduce with the question mark.
    canonical = {lbl.casefold().strip(parsing.LABEL_STRIP_CHARS): lbl for lbl in spec.labels or []}
    match = canonical.get(label.casefold())
    if match is None:
        return Parsed(ok=False, error=f"label {label!r} not in label set", value=label)
    return Parsed(ok=True, value=match, content=match)


def score(spec: TaskSpec, item: Item, parsed: Parsed, extra: dict[str, Any] | None = None) -> ItemScore:
    gold = item["label"]
    pred = parsed.value if parsed.ok else None
    return ItemScore(
        item_id=item["id"],
        valid=parsed.ok,
        stats={"correct": bool(parsed.ok and pred == gold), "gold": gold, "pred": pred},
    )


def aggregate(spec: TaskSpec, scores: list[ItemScore]) -> dict[str, float]:
    n = len(scores)
    if n == 0:
        return {"primary": math.nan, "accuracy": math.nan, "macro_f1": math.nan, "invalid_output_rate": math.nan}
    accuracy = sum(1 for s in scores if s.stats.get("correct")) / n
    invalid_rate = sum(1 for s in scores if not s.valid) / n
    # One pass over the items (this runs thousands of times inside the bootstrap).
    gold_n: Counter[str] = Counter()
    pred_n: Counter[str] = Counter()
    hit_n: Counter[str] = Counter()
    for s in scores:
        gold, pred = s.stats["gold"], s.stats.get("pred")
        gold_n[gold] += 1
        if pred:
            pred_n[pred] += 1
            if pred == gold:
                hit_n[gold] += 1
    f1s = []
    for lbl in sorted(set(gold_n) | set(pred_n)):
        tp = hit_n[lbl]
        fp, fn = pred_n[lbl] - tp, gold_n[lbl] - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {"primary": accuracy, "accuracy": accuracy, "macro_f1": macro_f1, "invalid_output_rate": invalid_rate}


def validate_item(spec: TaskSpec, item: Item) -> list[str]:
    errors = []
    if "id" not in item:
        errors.append("missing 'id'")
    if "text" not in item:
        errors.append("missing 'text'")
    if "label" not in item:
        errors.append("missing 'label'")
    elif spec.labels and item["label"] not in spec.labels:
        errors.append(f"label {item['label']!r} not in task label set")
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
