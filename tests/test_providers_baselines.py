from __future__ import annotations

import json
from pathlib import Path

from local_enough.config import BaselineModel
from local_enough.providers.baselines import (
    FuzzyMatchBaseline,
    RegexPiiBaseline,
    TfidfBaseline,
    baseline_for,
)
from local_enough.tasks.base import TaskSpec

TRAIN = [
    {"text": "my card was charged twice this month", "label": "billing_issue"},
    {"text": "i was billed twice for the same order", "label": "billing_issue"},
    {"text": "double charge appeared on my statement", "label": "billing_issue"},
    {"text": "where is my package, it has not arrived", "label": "delivery_issue"},
    {"text": "my parcel never showed up at the address", "label": "delivery_issue"},
    {"text": "the delivery is very late and i am worried", "label": "delivery_issue"},
]


def test_tfidf_predicts_the_matching_label() -> None:
    baseline = TfidfBaseline(TRAIN)
    assert baseline.predict("i think i got charged two times for one purchase") == "billing_issue"
    assert baseline.predict("my order has still not arrived at my house") == "delivery_issue"
    assert baseline.respond("my card was billed twice") == baseline.predict("my card was billed twice")


def test_tfidf_requires_nonempty_train() -> None:
    import pytest

    with pytest.raises(ValueError):
        TfidfBaseline([])


def test_regex_pii_finds_email_and_uk_phone() -> None:
    baseline = RegexPiiBaseline()
    text = "Contact Jane at jane.doe@example.com or call 07700 900123 for details."
    found = baseline.find(text)
    types = {item["type"] for item in found}
    assert "EMAIL" in types
    assert "PHONE" in types
    assert any(item["text"] == "jane.doe@example.com" for item in found)
    assert any(item["text"] == "07700 900123" for item in found)


def test_regex_pii_finds_us_phone_and_intl_uk_phone() -> None:
    baseline = RegexPiiBaseline()
    text = "US line: (415) 555-0142. UK line: +44 7700 900456."
    found = baseline.find(text)
    types = {item["type"] for item in found}
    assert "PHONE" in types
    assert len([i for i in found if i["type"] == "PHONE"]) >= 2


def test_regex_pii_validates_iban_checksum() -> None:
    baseline = RegexPiiBaseline()
    valid = "Bank details: GB82WEST12345698765432 for the transfer."
    invalid = "Reference code: GB82WEST12345698765431 only, not a real IBAN."
    assert any(item["type"] == "IBAN" for item in baseline.find(valid))
    assert not any(item["type"] == "IBAN" for item in baseline.find(invalid))


def test_regex_pii_dob_needs_a_cue_word() -> None:
    baseline = RegexPiiBaseline()
    with_cue = "Patient date of birth: 1990-03-12, seen on referral."
    without_cue = "The invoice is dated 1990-03-12 and was paid in full."
    assert any(item["type"] == "DATE_OF_BIRTH" for item in baseline.find(with_cue))
    assert not any(item["type"] == "DATE_OF_BIRTH" for item in baseline.find(without_cue))


def test_regex_pii_respond_is_json_list() -> None:
    baseline = RegexPiiBaseline()
    out = baseline.respond("Email me at a@example.com")
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert parsed[0]["type"] == "EMAIL"


CALIB_PAIRS = [
    {
        "left": {
            "name": "Acme Ltd",
            "street": "1 High St",
            "postcode": "AB1 2CD",
            "city": "London",
            "country": "UK",
            "domain": "acme.example.com",
            "vat_id": "GB123456789",
            "phone": "07700900111",
        },
        "right": {
            "name": "Acme Limited",
            "street": "1 High Street",
            "postcode": "AB1 2CD",
            "city": "London",
            "country": "UK",
            "domain": "acme.example.com",
            "vat_id": "GB123456789",
            "phone": "07700900111",
        },
        "match": True,
    },
    {
        "left": {
            "name": "Acme Ltd",
            "street": "1 High St",
            "postcode": "AB1 2CD",
            "city": "London",
            "country": "UK",
            "domain": "acme.example.com",
            "vat_id": "GB123456789",
            "phone": "07700900111",
        },
        "right": {
            "name": "Zenith Corp",
            "street": "9 Low Rd",
            "postcode": "ZZ9 9ZZ",
            "city": "Glasgow",
            "country": "UK",
            "domain": "zenith.example.org",
            "vat_id": "GB999999999",
            "phone": "07700900999",
        },
        "match": False,
    },
    {
        "left": {
            "name": "Beta Supplies",
            "street": "5 Park Ave",
            "postcode": "BB2 3EE",
            "city": "Bristol",
            "country": "UK",
            "domain": "beta.example.net",
            "vat_id": "GB222222222",
            "phone": "07700900222",
        },
        "right": {
            "name": "Beta Supplies Ltd",
            "street": "5 Park Avenue",
            "postcode": "BB2 3EE",
            "city": "Bristol",
            "country": "UK",
            "domain": "beta.example.net",
            "vat_id": "GB222222222",
            "phone": "07700900222",
        },
        "match": True,
    },
    {
        "left": {
            "name": "Beta Supplies",
            "street": "5 Park Ave",
            "postcode": "BB2 3EE",
            "city": "Bristol",
            "country": "UK",
            "domain": "beta.example.net",
            "vat_id": "GB222222222",
            "phone": "07700900222",
        },
        "right": {
            "name": "Gamma Facilities",
            "street": "77 Mill Ln",
            "postcode": "GG7 7GG",
            "city": "Leeds",
            "country": "UK",
            "domain": "gamma.example.com",
            "vat_id": "GB777777777",
            "phone": "07700900777",
        },
        "match": False,
    },
]


def test_fuzzy_match_tunes_threshold_and_band_in_range() -> None:
    baseline = FuzzyMatchBaseline()
    baseline.tune(CALIB_PAIRS)
    assert baseline.threshold is not None
    assert 0.0 <= baseline.threshold <= 1.0
    assert baseline.band is not None
    lo, hi = baseline.band
    assert 0.0 <= lo <= hi <= 1.0


def test_fuzzy_match_predicts_matches_and_nonmatches() -> None:
    baseline = FuzzyMatchBaseline()
    baseline.tune(CALIB_PAIRS)
    assert baseline.predict(CALIB_PAIRS[0]["left"], CALIB_PAIRS[0]["right"]) is True
    assert baseline.predict(CALIB_PAIRS[1]["left"], CALIB_PAIRS[1]["right"]) is False
    assert baseline.score(CALIB_PAIRS[0]) == baseline.similarity(CALIB_PAIRS[0]["left"], CALIB_PAIRS[0]["right"])


def test_fuzzy_match_predict_before_tune_raises() -> None:
    import pytest

    baseline = FuzzyMatchBaseline()
    with pytest.raises(RuntimeError):
        baseline.predict({"name": "a"}, {"name": "b"})


def test_fuzzy_match_respond_returns_match_json() -> None:
    baseline = FuzzyMatchBaseline()
    baseline.tune(CALIB_PAIRS)
    item_input = json.dumps({"left": CALIB_PAIRS[0]["left"], "right": CALIB_PAIRS[0]["right"]})
    out = baseline.respond(item_input)
    assert json.loads(out) == {"match": True}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_baseline_for_classification_trains_from_task_root(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "train.jsonl", TRAIN)
    spec = TaskSpec(name="classification", kind="classification", train="train.jsonl", root=tmp_path)
    baseline = baseline_for(BaselineModel(id="tfidf-baseline", task="classification"), spec)
    assert isinstance(baseline, TfidfBaseline)


def test_baseline_for_classification_without_train_is_none(tmp_path: Path) -> None:
    spec = TaskSpec(name="classification", kind="classification", root=tmp_path)
    baseline = baseline_for(BaselineModel(id="tfidf-baseline", task="classification"), spec)
    assert baseline is None


def test_baseline_for_pii_redaction(tmp_path: Path) -> None:
    spec = TaskSpec(name="pii_redaction", kind="pii_redaction", root=tmp_path)
    baseline = baseline_for(BaselineModel(id="regex-baseline", task="pii_redaction"), spec)
    assert isinstance(baseline, RegexPiiBaseline)


def test_baseline_for_entity_matching_tunes_on_calib(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "calib.jsonl", CALIB_PAIRS)
    spec = TaskSpec(name="entity_matching", kind="entity_matching", root=tmp_path)
    baseline = baseline_for(BaselineModel(id="rapidfuzz-baseline", task="entity_matching"), spec)
    assert isinstance(baseline, FuzzyMatchBaseline)
    assert baseline.threshold is not None
