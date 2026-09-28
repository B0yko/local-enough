"""extraction: type-aware normalisation and scoring on hand-made examples."""

from __future__ import annotations

import json

from local_enough.tasks import extraction as mod
from local_enough.tasks.base import FieldSpec, TaskSpec

FIELDS = {
    "company_name": FieldSpec(type="string", required=True, grounded=True, nullable=False),
    "contact_email": FieldSpec(type="email", required=True, grounded=True, nullable=False),
    "phone": FieldSpec(type="phone", required=True, nullable=False),
    "country": FieldSpec(type="country", required=True, nullable=False),
    "product": FieldSpec(type="enum", values=["Alpha", "Beta"], required=True, nullable=False),
    "seats": FieldSpec(type="int", nullable=True),
    "budget_amount": FieldSpec(type="amount", nullable=True),
    "budget_currency": FieldSpec(type="currency", values=["EUR", "USD", "GBP"], nullable=True),
    "requested_meeting_date": FieldSpec(type="date", nullable=True),
}
SPEC = TaskSpec(name="t", kind="extraction", fields=FIELDS)

GOLD = {
    "company_name": "Acme Widgets Ltd",
    "contact_email": "jane.doe@acme.example.com",
    "phone": "+447700900123",
    "country": "GB",
    "product": "Alpha",
    "seats": 40,
    "budget_amount": 12500.0,
    "budget_currency": "EUR",
    "requested_meeting_date": "2026-04-14",
}


def item_with_gold(gold: dict) -> dict:
    return {"id": "x", "text": "irrelevant", "gold": gold}


def test_normalize_field_variants() -> None:
    assert mod.normalize_field("string", "  Acme   Widgets Ltd ", FIELDS["company_name"]) == "acme widgets ltd"
    assert (
        mod.normalize_field("email", "Jane.Doe@ACME.example.com", FIELDS["contact_email"])
        == "jane.doe@acme.example.com"
    )
    assert mod.normalize_field("phone", "+44 7700 900123", FIELDS["phone"]) == "447700900123"
    assert mod.normalize_field("phone", "07700900123", FIELDS["phone"]) == "07700900123"
    assert mod.normalize_field("country", "United Kingdom", FIELDS["country"]) == "GB"
    assert mod.normalize_field("country", "gb", FIELDS["country"]) == "GB"
    assert mod.normalize_field("date", "2026-04-14T00:00:00", FIELDS["requested_meeting_date"]) == "2026-04-14"
    assert mod.normalize_field("enum", "alpha", FIELDS["product"]) == "Alpha"
    assert mod.normalize_field("int", "40", FIELDS["seats"]) == 40
    assert mod.normalize_field("int", "1,024", FIELDS["seats"]) == 1024
    assert mod.normalize_field("amount", "EUR 12,500", FIELDS["budget_amount"]) == 12500.0
    assert mod.normalize_field("amount", "not a number", FIELDS["budget_amount"]) is None
    assert mod.normalize_field("currency", "eur", FIELDS["budget_currency"]) == "EUR"
    assert mod.normalize_field("string", None, FIELDS["company_name"]) is None


def test_parse_valid_json_object() -> None:
    parsed = mod.parse(SPEC, json.dumps(GOLD))
    assert parsed.ok
    assert parsed.value == GOLD
    assert json.loads(parsed.content)["company_name"] == GOLD["company_name"]


def test_parse_rejects_non_object() -> None:
    parsed = mod.parse(SPEC, "[1, 2, 3]")
    assert not parsed.ok


def test_parse_rejects_no_json() -> None:
    parsed = mod.parse(SPEC, "I cannot help with that.")
    assert not parsed.ok


def test_score_exact_match_on_gold() -> None:
    item = item_with_gold(GOLD)
    parsed = mod.parse(SPEC, json.dumps(GOLD))
    score = mod.score(SPEC, item, parsed)
    assert score.valid
    assert score.stats["exact_match"] is True
    assert all(score.stats["field_correct"].values())


def test_score_tolerates_format_differences() -> None:
    # Same information, different formatting: phone spaced, email uppercased, amount as string with symbol.
    pred = dict(GOLD)
    pred["phone"] = "+44 7700 900123"
    pred["contact_email"] = "JANE.DOE@ACME.example.com"
    pred["budget_amount"] = "EUR 12,500.00"
    pred["country"] = "United Kingdom"
    item = item_with_gold(GOLD)
    parsed = mod.parse(SPEC, json.dumps(pred))
    score = mod.score(SPEC, item, parsed)
    assert score.stats["exact_match"] is True


def test_score_null_gold_matches_null_pred() -> None:
    gold = dict(GOLD)
    gold["seats"] = None
    pred = dict(GOLD)
    pred["seats"] = None
    item = item_with_gold(gold)
    parsed = mod.parse(SPEC, json.dumps(pred))
    score = mod.score(SPEC, item, parsed)
    assert score.stats["field_correct"]["seats"] is True


def test_score_wrong_field_marks_only_that_field() -> None:
    pred = dict(GOLD)
    pred["seats"] = 999
    item = item_with_gold(GOLD)
    parsed = mod.parse(SPEC, json.dumps(pred))
    score = mod.score(SPEC, item, parsed)
    assert score.stats["field_correct"]["seats"] is False
    assert score.stats["field_correct"]["company_name"] is True
    assert score.stats["exact_match"] is False


def test_score_invalid_output_marks_all_fields_wrong() -> None:
    item = item_with_gold(GOLD)
    parsed = mod.parse(SPEC, "not json")
    score = mod.score(SPEC, item, parsed)
    assert not score.valid
    assert not any(score.stats["field_correct"].values())


def test_aggregate_field_accuracy_is_micro_over_all_fields() -> None:
    item = item_with_gold(GOLD)
    pred_one_wrong = dict(GOLD)
    pred_one_wrong["seats"] = 1
    scores = [
        mod.score(SPEC, item, mod.parse(SPEC, json.dumps(GOLD))),
        mod.score(SPEC, item, mod.parse(SPEC, json.dumps(pred_one_wrong))),
    ]
    agg = mod.aggregate(SPEC, scores)
    n_fields = len(FIELDS)
    expected = (n_fields + (n_fields - 1)) / (2 * n_fields)
    assert abs(agg["field_accuracy"] - expected) < 1e-9
    assert agg["exact_match"] == 0.5
    assert agg["primary"] == agg["field_accuracy"]


def test_validate_item() -> None:
    assert mod.validate_item(SPEC, item_with_gold(GOLD)) == []
    errors = mod.validate_item(SPEC, {"id": "x", "text": "t", "gold": {"company_name": "Acme"}})
    assert any("seats" in e for e in errors)
