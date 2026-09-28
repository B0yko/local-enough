"""Inbound B2B enquiry email to CRM fields."""

from __future__ import annotations

import json
import math
import re
from datetime import date
from typing import Any

from local_enough.config import TaskKind
from local_enough.tasks import parsing, registry
from local_enough.tasks.base import FieldSpec, Item, ItemScore, Message, Parsed, TaskSpec

kind: TaskKind = "extraction"
primary_metric = "field_accuracy"

_DATE_HEADER_RE = re.compile(r"^Date:\s*\w+,\s*(\d{1,2})\s+(\w{3})\s+(\d{4})", re.MULTILINE)
_MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
_COUNTRY_NAME_TO_CODE = {
    "UNITED KINGDOM": "GB",
    "UK": "GB",
    "GREAT BRITAIN": "GB",
    "UNITED STATES": "US",
    "UNITED STATES OF AMERICA": "US",
    "USA": "US",
    "CANADA": "CA",
}


def render_messages(spec: TaskSpec, item: Item) -> list[Message]:
    return registry.render_prompt("extraction", fields=spec.fields or {}, text=item["text"])


def user_input(spec: TaskSpec, item: Item) -> str:
    return str(item["text"])


def item_from_input(spec: TaskSpec, raw: str) -> Item:
    reference_date = None
    m = _DATE_HEADER_RE.search(raw)
    if m:
        day, mon, year = m.groups()
        month = _MONTHS.get(mon)
        if month:
            reference_date = f"{year}-{month:02d}-{int(day):02d}"
    return {"id": "router", "text": raw, "reference_date": reference_date}


def parse(spec: TaskSpec, raw_text: str) -> Parsed:
    try:
        obj = parsing.first_json_value(raw_text)
    except ValueError as exc:
        return Parsed(ok=False, error=str(exc))
    if not isinstance(obj, dict):
        return Parsed(ok=False, error="parsed JSON is not an object")
    fields = spec.fields or {}
    ordered = {name: obj.get(name) for name in fields}
    return Parsed(ok=True, value=ordered, content=json.dumps(ordered, ensure_ascii=False))


def normalize_date(value: str) -> str:
    v = value.strip()
    try:
        return date.fromisoformat(v[:10]).isoformat()
    except ValueError:
        return v.casefold()


def normalize_country(value: str) -> str:
    v = value.strip().upper()
    return _COUNTRY_NAME_TO_CODE.get(v, v)


def normalize_enum(value: str, allowed: list[str]) -> str:
    v = value.strip()
    for a in allowed:
        if a.casefold() == v.casefold():
            return a
    return v


def normalize_field(field_type: str, value: Any, field_spec: FieldSpec) -> Any:
    if value is None:
        return None
    if field_type == "string":
        return re.sub(r"\s+", " ", str(value)).strip().casefold()
    if field_type == "email":
        return str(value).strip().lower()
    if field_type == "phone":
        digits = re.sub(r"[^0-9]", "", str(value))
        return digits or None
    if field_type == "country":
        return normalize_country(str(value))
    if field_type == "date":
        return normalize_date(str(value))
    if field_type in ("enum", "currency"):
        return normalize_enum(str(value), field_spec.values or [])
    if field_type == "int":
        try:
            return int(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            return None
    if field_type == "amount":
        try:
            cleaned = re.sub(r"[^0-9.\-]", "", str(value))
            return round(float(cleaned), 2) if cleaned not in ("", "-", ".") else None
        except (TypeError, ValueError):
            return None
    return value


def score(spec: TaskSpec, item: Item, parsed: Parsed, extra: dict[str, Any] | None = None) -> ItemScore:
    fields = spec.fields or {}
    gold = item.get("gold", {})
    if not parsed.ok:
        field_correct = dict.fromkeys(fields, False)
        return ItemScore(item_id=item["id"], valid=False, stats={"field_correct": field_correct, "exact_match": False})
    pred = parsed.value or {}
    field_correct = {}
    for name, fspec in fields.items():
        gv = normalize_field(fspec.type, gold.get(name), fspec)
        pv = normalize_field(fspec.type, pred.get(name), fspec)
        field_correct[name] = gv == pv
    exact = all(field_correct.values()) if field_correct else False
    return ItemScore(item_id=item["id"], valid=True, stats={"field_correct": field_correct, "exact_match": exact})


def aggregate(spec: TaskSpec, scores: list[ItemScore]) -> dict[str, float]:
    n = len(scores)
    if n == 0:
        return dict.fromkeys(["primary", "field_accuracy", "exact_match", "invalid_output_rate"], math.nan)
    invalid_rate = sum(1 for s in scores if not s.valid) / n
    exact = sum(1 for s in scores if s.stats.get("exact_match")) / n
    total = correct = 0
    for s in scores:
        fc = s.stats.get("field_correct", {})
        total += len(fc)
        correct += sum(1 for v in fc.values() if v)
    field_accuracy = correct / total if total else math.nan
    return {
        "primary": field_accuracy,
        "field_accuracy": field_accuracy,
        "exact_match": exact,
        "invalid_output_rate": invalid_rate,
    }


def validate_item(spec: TaskSpec, item: Item) -> list[str]:
    errors = []
    if "id" not in item:
        errors.append("missing 'id'")
    if "text" not in item:
        errors.append("missing 'text'")
    gold = item.get("gold")
    if not isinstance(gold, dict):
        errors.append("missing or invalid 'gold' object")
        return errors
    for name in spec.fields or {}:
        if name not in gold:
            errors.append(f"gold missing field {name!r}")
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
