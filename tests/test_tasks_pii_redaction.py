"""pii_redaction: span alignment, coverage threshold, F2 and leak rate on hand-made examples."""

from __future__ import annotations

import json

from local_enough.tasks import pii_redaction as mod
from local_enough.tasks.base import Parsed, TaskSpec

SPEC = TaskSpec(name="t", kind="pii_redaction", pii_types=["PERSON", "EMAIL", "PHONE"])

TEXT = "Contact Jane Doe at jane@example.com or call 07700 900123. Order ORD-4821 confirmed."
PERSON_SPAN = {"start": TEXT.index("Jane Doe"), "end": TEXT.index("Jane Doe") + len("Jane Doe"), "type": "PERSON"}
EMAIL_SPAN = {
    "start": TEXT.index("jane@example.com"),
    "end": TEXT.index("jane@example.com") + len("jane@example.com"),
    "type": "EMAIL",
}
PHONE_SPAN = {
    "start": TEXT.index("07700 900123"),
    "end": TEXT.index("07700 900123") + len("07700 900123"),
    "type": "PHONE",
}


def item(spans: list[dict]) -> dict:
    return {"id": "x", "text": TEXT, "spans": spans}


def predict(*entries: tuple[str, str]) -> Parsed:
    payload = [{"type": t, "text": txt} for t, txt in entries]
    return mod.parse(SPEC, json.dumps(payload))


def test_parse_extracts_list() -> None:
    parsed = predict(("PERSON", "Jane Doe"))
    assert parsed.ok
    assert parsed.value == [{"type": "PERSON", "text": "Jane Doe"}]


def test_parse_rejects_non_list() -> None:
    parsed = mod.parse(SPEC, '{"type": "PERSON", "text": "Jane Doe"}')
    assert not parsed.ok


def test_perfect_prediction_gives_f2_recall_precision_of_one() -> None:
    it = item([PERSON_SPAN, EMAIL_SPAN, PHONE_SPAN])
    parsed = predict(("PERSON", "Jane Doe"), ("EMAIL", "jane@example.com"), ("PHONE", "07700 900123"))
    score = mod.score(SPEC, it, parsed)
    assert score.stats["zero_leak"] is True
    agg = mod.aggregate(SPEC, [score])
    assert agg["recall"] == 1.0
    assert agg["precision"] == 1.0
    assert agg["f2"] == 1.0
    assert agg["leak_rate"] == 0.0
    assert agg["zero_leak_doc_rate"] == 1.0


def test_partial_recall_weighs_more_than_precision_in_f2() -> None:
    it = item([PERSON_SPAN, EMAIL_SPAN, PHONE_SPAN])
    parsed = predict(("PERSON", "Jane Doe"))  # catches 1 of 3 gold spans, 1 correct prediction
    score = mod.score(SPEC, it, parsed)
    agg = mod.aggregate(SPEC, [score])
    assert agg["precision"] == 1.0
    assert abs(agg["recall"] - 1 / 3) < 1e-9
    expected_f2 = 5 * 1.0 * (1 / 3) / (4 * 1.0 + 1 / 3)
    assert abs(agg["f2"] - expected_f2) < 1e-9
    assert agg["leak_rate"] == 1 - 1 / 3
    assert score.stats["zero_leak"] is False


def test_wrong_type_catches_for_recall_but_not_precision() -> None:
    it = item([PERSON_SPAN])
    parsed = predict(("EMAIL", "Jane Doe"))  # right text/location, wrong type
    score = mod.score(SPEC, it, parsed)
    agg = mod.aggregate(SPEC, [score])
    assert agg["recall"] == 1.0  # recall ignores type
    assert agg["precision"] == 0.0  # precision requires type
    assert agg["f2"] == 0.0


def test_prediction_not_found_in_text_is_a_false_positive() -> None:
    it = item([PERSON_SPAN])
    parsed = predict(("PERSON", "Someone Else"))  # not a substring of TEXT anywhere
    score = mod.score(SPEC, it, parsed)
    agg = mod.aggregate(SPEC, [score])
    assert agg["precision"] == 0.0
    assert agg["recall"] == 0.0


def test_coverage_threshold_80_percent_boundary() -> None:
    text = "Reference number 1234567890 recorded."
    start = text.index("1234567890")
    gold = [{"start": start, "end": start + 10, "type": "PHONE"}]
    at_threshold = mod.score(SPEC, {"id": "a", "text": text, "spans": gold}, predict(("PHONE", "12345678")))
    below_threshold = mod.score(SPEC, {"id": "b", "text": text, "spans": gold}, predict(("PHONE", "1234567")))
    assert at_threshold.stats["zero_leak"] is True  # 8/10 = 80% covered
    assert below_threshold.stats["zero_leak"] is False  # 7/10 = 70% covered


def test_invalid_output_counts_as_full_leak() -> None:
    it = item([PERSON_SPAN])
    score = mod.score(SPEC, it, mod.parse(SPEC, "not json"))
    assert not score.valid
    agg = mod.aggregate(SPEC, [score])
    assert agg["invalid_output_rate"] == 1.0
    assert agg["recall"] == 0.0


def test_empty_gold_document_is_trivially_zero_leak() -> None:
    it = item([])
    score = mod.score(SPEC, it, predict())
    assert score.stats["zero_leak"] is True
    agg = mod.aggregate(SPEC, [score])
    assert agg["zero_leak_doc_rate"] == 1.0


def test_validate_item_catches_bad_offsets() -> None:
    assert mod.validate_item(SPEC, item([PERSON_SPAN])) == []
    bad = item([{"start": 5, "end": 2, "type": "PERSON"}])
    errors = mod.validate_item(SPEC, bad)
    assert errors and "invalid offsets" in errors[0]
