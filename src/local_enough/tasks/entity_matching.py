"""Vendor/customer master dedupe: two company records to a match/no-match decision."""

from __future__ import annotations

import json
import math
from typing import Any

from local_enough.config import TaskKind
from local_enough.tasks import parsing, registry
from local_enough.tasks.base import Item, ItemScore, Message, Parsed, TaskSpec

kind: TaskKind = "entity_matching"
primary_metric = "f1"


def render_messages(spec: TaskSpec, item: Item) -> list[Message]:
    return registry.render_prompt("entity_matching", left=item["left"], right=item["right"])


def user_input(spec: TaskSpec, item: Item) -> str:
    return json.dumps({"left": item["left"], "right": item["right"]}, ensure_ascii=False)


def item_from_input(spec: TaskSpec, raw: str) -> Item:
    obj = json.loads(raw)
    return {"id": "router", "left": obj["left"], "right": obj["right"]}


def parse(spec: TaskSpec, raw_text: str) -> Parsed:
    try:
        obj = parsing.first_json_value(raw_text)
    except ValueError as exc:
        return Parsed(ok=False, error=str(exc))
    if not isinstance(obj, dict) or not isinstance(obj.get("match"), bool):
        return Parsed(ok=False, error="expected an object with a boolean 'match' field")
    value = {"match": obj["match"]}
    return Parsed(ok=True, value=value, content=json.dumps(value))


def score(spec: TaskSpec, item: Item, parsed: Parsed, extra: dict[str, Any] | None = None) -> ItemScore:
    gold = bool(item["match"])
    pred = parsed.value["match"] if parsed.ok else False
    return ItemScore(
        item_id=item["id"],
        valid=parsed.ok,
        stats={
            "tp": int(pred and gold),
            "fp": int(pred and not gold),
            "fn": int((not pred) and gold),
            "gold": gold,
            "pred": pred if parsed.ok else None,
        },
    )


def aggregate(spec: TaskSpec, scores: list[ItemScore]) -> dict[str, float]:
    n = len(scores)
    if n == 0:
        return dict.fromkeys(["primary", "f1", "precision", "recall", "invalid_output_rate"], math.nan)
    invalid_rate = sum(1 for s in scores if not s.valid) / n
    tp = sum(s.stats["tp"] for s in scores)
    fp = sum(s.stats["fp"] for s in scores)
    fn = sum(s.stats["fn"] for s in scores)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"primary": f1, "f1": f1, "precision": precision, "recall": recall, "invalid_output_rate": invalid_rate}


def validate_item(spec: TaskSpec, item: Item) -> list[str]:
    errors = []
    if "id" not in item:
        errors.append("missing 'id'")
    if not isinstance(item.get("left"), dict):
        errors.append("missing or invalid 'left'")
    if not isinstance(item.get("right"), dict):
        errors.append("missing or invalid 'right'")
    if not isinstance(item.get("match"), bool):
        errors.append("missing or invalid 'match'")
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
