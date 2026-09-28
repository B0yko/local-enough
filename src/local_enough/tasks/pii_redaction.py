"""Redact personal data from business notes.

Scoring rules: a predicted ``{type, text}`` is aligned to every
whitespace-normalised occurrence of ``text`` in the document; a gold span counts as caught when some
occurrence covers >=80% of its characters. Catching (recall) ignores type; precision requires the type
of the covering occurrence to match the gold span's type.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from local_enough.config import TaskKind
from local_enough.tasks import parsing, registry
from local_enough.tasks.base import Item, ItemScore, Message, Parsed, TaskSpec

kind: TaskKind = "pii_redaction"
primary_metric = "f2"
_COVERAGE_THRESHOLD = 0.8


def render_messages(spec: TaskSpec, item: Item) -> list[Message]:
    return registry.render_prompt("pii_redaction", text=item["text"])


def user_input(spec: TaskSpec, item: Item) -> str:
    return str(item["text"])


def item_from_input(spec: TaskSpec, raw: str) -> Item:
    return {"id": "router", "text": raw}


def parse(spec: TaskSpec, raw_text: str) -> Parsed:
    try:
        obj = parsing.first_json_value(raw_text)
    except ValueError as exc:
        return Parsed(ok=False, error=str(exc))
    if not isinstance(obj, list):
        return Parsed(ok=False, error="parsed JSON is not a list")
    cleaned = [
        {"type": entry.get("type"), "text": entry["text"]}
        for entry in obj
        if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"].strip()
    ]
    return Parsed(ok=True, value=cleaned, content=json.dumps(cleaned, ensure_ascii=False))


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _normalize_with_offsets(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed text plus a map from each of its indices back to ``text``."""
    out: list[str] = []
    offsets: list[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                offsets.append(i)
                prev_space = True
            continue
        out.append(ch)
        offsets.append(i)
        prev_space = False
    return "".join(out), offsets


def _find_occurrences(doc_text: str, norm_doc: str, offsets: list[int], query: str) -> list[tuple[int, int]]:
    norm_query = _normalize_ws(query)
    if not norm_query:
        return []
    spans = []
    start = 0
    while True:
        idx = norm_doc.find(norm_query, start)
        if idx == -1:
            break
        end = idx + len(norm_query)
        orig_start = offsets[idx] if idx < len(offsets) else len(doc_text)
        orig_end = (offsets[end - 1] + 1) if end - 1 < len(offsets) else len(doc_text)
        spans.append((orig_start, orig_end))
        start = idx + 1
    return spans


def _coverage(occ: tuple[int, int], gold: tuple[int, int]) -> float:
    overlap = max(0, min(occ[1], gold[1]) - max(occ[0], gold[0]))
    glen = gold[1] - gold[0]
    return overlap / glen if glen > 0 else 0.0


def score(spec: TaskSpec, item: Item, parsed: Parsed, extra: dict[str, Any] | None = None) -> ItemScore:
    gold_spans = item.get("spans", [])
    if not parsed.ok:
        return ItemScore(
            item_id=item["id"],
            valid=False,
            stats={"gold_n": len(gold_spans), "gold_caught": 0, "pred_n": 0, "pred_tp": 0, "zero_leak": not gold_spans},
        )
    doc_text = item["text"]
    norm_doc, offsets = _normalize_with_offsets(doc_text)
    caught = [False] * len(gold_spans)
    pred_n = pred_tp = 0
    for entry in parsed.value or []:
        pred_n += 1
        occs = _find_occurrences(doc_text, norm_doc, offsets, entry["text"])
        entry_tp = False
        for occ in occs:
            for gi, g in enumerate(gold_spans):
                if _coverage(occ, (g["start"], g["end"])) >= _COVERAGE_THRESHOLD:
                    caught[gi] = True
                    if entry.get("type") == g["type"]:
                        entry_tp = True
        if entry_tp:
            pred_tp += 1
    gold_caught = sum(caught)
    return ItemScore(
        item_id=item["id"],
        valid=True,
        stats={
            "gold_n": len(gold_spans),
            "gold_caught": gold_caught,
            "pred_n": pred_n,
            "pred_tp": pred_tp,
            "zero_leak": gold_caught == len(gold_spans),
        },
    )


def aggregate(spec: TaskSpec, scores: list[ItemScore]) -> dict[str, float]:
    n = len(scores)
    if n == 0:
        keys = ["primary", "f2", "recall", "precision", "leak_rate", "zero_leak_doc_rate", "invalid_output_rate"]
        return dict.fromkeys(keys, math.nan)
    invalid_rate = sum(1 for s in scores if not s.valid) / n
    gold_n = sum(s.stats["gold_n"] for s in scores)
    gold_caught = sum(s.stats["gold_caught"] for s in scores)
    pred_n = sum(s.stats["pred_n"] for s in scores)
    pred_tp = sum(s.stats["pred_tp"] for s in scores)
    recall = (gold_caught / gold_n) if gold_n else 1.0
    precision = (pred_tp / pred_n) if pred_n else 0.0
    leak_rate = 1.0 - recall
    zero_leak_doc_rate = sum(1 for s in scores if s.stats["zero_leak"]) / n
    denom = 4 * precision + recall
    f2 = (5 * precision * recall / denom) if denom > 0 else 0.0
    return {
        "primary": f2,
        "f2": f2,
        "recall": recall,
        "precision": precision,
        "leak_rate": leak_rate,
        "zero_leak_doc_rate": zero_leak_doc_rate,
        "invalid_output_rate": invalid_rate,
    }


def validate_item(spec: TaskSpec, item: Item) -> list[str]:
    errors = []
    if "id" not in item:
        errors.append("missing 'id'")
    text = item.get("text")
    if not isinstance(text, str):
        errors.append("missing or invalid 'text'")
        return errors
    spans = item.get("spans")
    if not isinstance(spans, list):
        errors.append("missing or invalid 'spans'")
        return errors
    for i, s in enumerate(spans):
        if not isinstance(s, dict) or not {"start", "end", "type"} <= s.keys():
            errors.append(f"spans[{i}] missing start/end/type")
            continue
        start, end = s["start"], s["end"]
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
            errors.append(f"spans[{i}] has invalid offsets")
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
