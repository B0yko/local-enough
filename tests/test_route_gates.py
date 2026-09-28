"""One test per gate kind, plus the date-expression resolver and threshold tuning."""

from __future__ import annotations

import json

from local_enough.bench.rundir import PREDICTIONS, RunDir
from local_enough.providers.baselines import FuzzyMatchBaseline, RegexPiiBaseline, TfidfBaseline
from local_enough.route import gates
from local_enough.route.gates import GateContext, GateThresholds
from local_enough.tasks.base import FieldSpec, Parsed, TaskSpec

# -- fixtures -------------------------------------------------------------------------------------------


def _classification_spec() -> TaskSpec:
    return TaskSpec(name="classification", kind="classification", labels=["billing", "support", "sales"])


def _classification_ctx() -> GateContext:
    train = [
        {"text": "my invoice is wrong", "label": "billing"},
        {"text": "refund my payment please", "label": "billing"},
        {"text": "the app keeps crashing", "label": "support"},
        {"text": "help me fix a bug", "label": "support"},
        {"text": "I want to buy more seats", "label": "sales"},
        {"text": "tell me about pricing plans", "label": "sales"},
    ]
    thresholds = GateThresholds(classification_gate_enabled={"classification": True})
    return GateContext(
        tfidf={"classification": TfidfBaseline(train)}, fuzzy={}, regex_pii=RegexPiiBaseline(), thresholds=thresholds
    )


def _extraction_spec() -> TaskSpec:
    fields = {
        "company_name": FieldSpec(type="string", required=True, grounded=True),
        "phone": FieldSpec(type="phone", required=True, grounded=False),
        "country": FieldSpec(type="country", required=True, grounded=False),
        "requested_meeting_date": FieldSpec(type="date", required=False, grounded=False),
        "product": FieldSpec(type="enum", values=["Nimbus Workspace", "Ledgerline ERP"], required=True, grounded=False),
        "budget_amount": FieldSpec(type="amount", required=False, grounded=False),
    }
    return TaskSpec(name="extraction", kind="extraction", fields=fields)


def _extraction_ctx(threshold: float = 85.0) -> GateContext:
    thresholds = GateThresholds(extraction_grounding_ratio={"extraction": threshold})
    return GateContext(tfidf={}, fuzzy={}, regex_pii=RegexPiiBaseline(), thresholds=thresholds)


def _pii_spec() -> TaskSpec:
    return TaskSpec(name="pii_redaction", kind="pii_redaction")


def _pii_ctx() -> GateContext:
    return GateContext(tfidf={}, fuzzy={}, regex_pii=RegexPiiBaseline(), thresholds=GateThresholds())


def _summarisation_spec() -> TaskSpec:
    return TaskSpec(name="summarisation", kind="summarisation", max_words=50)


def _entity_matching_spec() -> TaskSpec:
    return TaskSpec(name="entity_matching", kind="entity_matching")


_EM_LEFT = {
    "name": "Acme Ltd", "street": "1 High St", "postcode": "AB1 2CD", "city": "London",
    "country": "GB", "domain": "acme.example.com", "vat_id": "GB123456789", "phone": "+441234567890",
}  # fmt: skip
_EM_RIGHT = {
    "name": "Zenith Corp", "street": "99 Low Rd", "postcode": "ZZ9 9ZZ", "city": "Manchester",
    "country": "GB", "domain": "zenith.example.org", "vat_id": "GB987654321", "phone": "+449876543210",
}  # fmt: skip


def _entity_matching_ctx(band: tuple[float, float], threshold: float) -> GateContext:
    baseline = FuzzyMatchBaseline()
    baseline.threshold = threshold
    baseline.band = band
    thresholds = GateThresholds(
        entity_matching_band={"entity_matching": band}, entity_matching_threshold={"entity_matching": threshold}
    )
    return GateContext(
        tfidf={}, fuzzy={"entity_matching": baseline}, regex_pii=RegexPiiBaseline(), thresholds=thresholds
    )


# -- classification ---------------------------------------------------------------------------------------


def test_classification_gate_passes_when_it_agrees_with_baseline():
    spec, ctx = _classification_spec(), _classification_ctx()
    item = {"text": "my invoice is wrong, please refund it"}
    parsed = Parsed(ok=True, value="billing", content="billing")
    assert gates.gate(spec, item, parsed, ctx).passed


def test_classification_gate_escalates_on_disagreement_with_baseline():
    spec, ctx = _classification_spec(), _classification_ctx()
    item = {"text": "my invoice is wrong, please refund it"}
    parsed = Parsed(ok=True, value="sales", content="sales")
    result = gates.gate(spec, item, parsed, ctx)
    assert not result.passed and result.reason == "disagrees_with_baseline"


def test_classification_gate_format_only_skips_baseline_comparison():
    spec, ctx = _classification_spec(), _classification_ctx()
    item = {"text": "my invoice is wrong, please refund it"}
    parsed = Parsed(ok=True, value="sales", content="sales")
    assert gates.gate(spec, item, parsed, ctx, format_only=True).passed


def test_classification_gate_fails_invalid_label():
    spec, ctx = _classification_spec(), _classification_ctx()
    result = gates.gate(spec, {"text": "x"}, Parsed(ok=False, error="no label"), ctx)
    assert not result.passed and result.reason == "invalid_label"


# -- extraction ---------------------------------------------------------------------------------------


def _base_extraction_value(**overrides):
    value = {
        "company_name": "Acme Widgets Ltd",
        "phone": "+447700900123",
        "country": "GB",
        "requested_meeting_date": "2026-10-15",
        "product": "Nimbus Workspace",
        "budget_amount": "12500",
    }
    value.update(overrides)
    return value


def test_extraction_gate_passes_grounded_fields_by_containment():
    spec, ctx = _extraction_spec(), _extraction_ctx()
    source = (
        "Hello, this is Acme Widgets Ltd. Call us on 07700 900123. We are based in the United Kingdom. "
        "Can we meet tomorrow? We are interested in Nimbus Workspace, budget around 12,500 GBP."
    )
    item = {"text": source, "reference_date": "2026-10-14"}
    parsed = Parsed(ok=True, value=_base_extraction_value(requested_meeting_date="2026-10-15"))
    assert gates.gate(spec, item, parsed, ctx).passed


def test_extraction_gate_fails_ungrounded_free_text():
    spec, ctx = _extraction_spec(), _extraction_ctx()
    source = "Hello, we are interested in Nimbus Workspace. Call 07700 900123, based in the UK."
    item = {"text": source, "reference_date": "2026-10-14"}
    parsed = Parsed(ok=True, value=_base_extraction_value(company_name="Totally Different Corp"))
    result = gates.gate(spec, item, parsed, ctx)
    assert not result.passed and result.reason == "ungrounded:company_name"


def test_extraction_gate_fails_missing_required_field():
    spec, ctx = _extraction_spec(), _extraction_ctx()
    item = {"text": "Acme Widgets Ltd, 07700 900123, UK, Nimbus Workspace", "reference_date": "2026-10-14"}
    value = _base_extraction_value()
    value["phone"] = None
    parsed = Parsed(ok=True, value=value)
    result = gates.gate(spec, item, parsed, ctx)
    assert not result.passed and result.reason == "missing_required:phone"


def test_extraction_gate_fails_phone_digits_not_in_source():
    spec, ctx = _extraction_spec(), _extraction_ctx()
    item = {"text": "Acme Widgets Ltd, based in the UK, wants Nimbus Workspace", "reference_date": "2026-10-14"}
    parsed = Parsed(ok=True, value=_base_extraction_value())
    result = gates.gate(spec, item, parsed, ctx)
    assert not result.passed and result.reason == "phone_not_in_source:phone"


def test_extraction_gate_fails_country_not_in_source():
    spec, ctx = _extraction_spec(), _extraction_ctx()
    # A UK mobile number is evidence for GB, not for US.
    item = {"text": "Acme Widgets Ltd, 07700 900123, wants Nimbus Workspace", "reference_date": "2026-10-14"}
    parsed = Parsed(ok=True, value=_base_extraction_value(country="US"))
    result = gates.gate(spec, item, parsed, ctx)
    assert not result.passed and result.reason == "country_not_in_source:country"


def test_extraction_gate_date_rederivable_from_relative_expression():
    spec, ctx = _extraction_spec(), _extraction_ctx()
    item = {
        "text": "Acme Widgets Ltd, 07700 900123, UK, Nimbus Workspace, budget 12500. Can we meet next Monday?",
        "reference_date": "2026-10-14",  # a Wednesday
    }
    parsed = Parsed(ok=True, value=_base_extraction_value(requested_meeting_date="2026-10-19"))
    assert gates.gate(spec, item, parsed, ctx).passed


def test_extraction_gate_fails_date_not_rederivable():
    spec, ctx = _extraction_spec(), _extraction_ctx()
    item = {
        "text": "Acme Widgets Ltd, 07700 900123, UK, Nimbus Workspace, budget 12500. Can we meet tomorrow?",
        "reference_date": "2026-10-14",
    }
    parsed = Parsed(ok=True, value=_base_extraction_value(requested_meeting_date="2026-12-25"))
    result = gates.gate(spec, item, parsed, ctx)
    assert not result.passed and result.reason == "date_not_rederivable:requested_meeting_date"


def test_extraction_gate_fails_product_not_in_enum():
    spec, ctx = _extraction_spec(), _extraction_ctx()
    item = {"text": "Acme Widgets Ltd, 07700 900123, UK", "reference_date": "2026-10-14"}
    parsed = Parsed(ok=True, value=_base_extraction_value(product="Some Other Product", requested_meeting_date=None))
    result = gates.gate(spec, item, parsed, ctx)
    assert not result.passed and result.reason == "not_in_enum:product"


def test_extraction_gate_fails_amount_digits_not_in_source():
    spec, ctx = _extraction_spec(), _extraction_ctx()
    item = {
        "text": "Acme Widgets Ltd, 07700 900123, UK, Nimbus Workspace",
        "reference_date": "2026-10-14",
    }
    parsed = Parsed(ok=True, value=_base_extraction_value(requested_meeting_date=None))
    result = gates.gate(spec, item, parsed, ctx)
    assert not result.passed and result.reason == "amount_not_in_source:budget_amount"


def test_extraction_gate_amount_grounded_despite_a_trailing_zero_decimal():
    """Gold/model JSON often carries a whole-number amount as a float (``12500.0``); the source never
    spells out the ".0", so a naive digit-for-digit match on the raw string would reject a correct amount."""
    spec, ctx = _extraction_spec(), _extraction_ctx()
    item = {
        "text": "Acme Widgets Ltd, 07700 900123, UK, Nimbus Workspace, budget 12,500 GBP.",
        "reference_date": "2026-10-14",
    }
    parsed = Parsed(ok=True, value=_base_extraction_value(requested_meeting_date=None, budget_amount="12500.0"))
    assert gates.gate(spec, item, parsed, ctx).passed


def test_extraction_gate_amount_grounded_via_k_shorthand():
    spec, ctx = _extraction_spec(), _extraction_ctx()
    item = {
        "text": "Acme Widgets Ltd, 07700 900123, UK, Nimbus Workspace. We've set aside $8k for the first year.",
        "reference_date": "2026-10-14",
    }
    parsed = Parsed(ok=True, value=_base_extraction_value(requested_meeting_date=None, budget_amount="8000"))
    assert gates.gate(spec, item, parsed, ctx).passed


# -- pii ---------------------------------------------------------------------------------------


def test_pii_gate_passes_when_spans_are_grounded_and_cover_regex_baseline():
    spec, ctx = _pii_spec(), _pii_ctx()
    text = "Please call John Smith at john.smith@example.com regarding the ticket."
    parsed = Parsed(
        ok=True, value=[{"type": "PERSON", "text": "John Smith"}, {"type": "EMAIL", "text": "john.smith@example.com"}]
    )
    assert gates.gate(spec, {"text": text}, parsed, ctx).passed


def test_pii_gate_fails_when_predicted_span_not_in_text():
    spec, ctx = _pii_spec(), _pii_ctx()
    text = "Please call John Smith about the ticket."
    parsed = Parsed(ok=True, value=[{"type": "PERSON", "text": "Someone Else"}])
    result = gates.gate(spec, {"text": text}, parsed, ctx)
    assert not result.passed and result.reason == "predicted_span_not_in_text"


def test_pii_gate_fails_when_regex_baseline_span_missed():
    spec, ctx = _pii_spec(), _pii_ctx()
    text = "Contact john.smith@example.com about the invoice."
    parsed = Parsed(ok=True, value=[])
    result = gates.gate(spec, {"text": text}, parsed, ctx)
    assert not result.passed and result.reason == "regex_baseline_span_missed"


def test_pii_gate_format_only_skips_baseline_coverage_check():
    spec, ctx = _pii_spec(), _pii_ctx()
    text = "Contact john.smith@example.com about the invoice."
    parsed = Parsed(ok=True, value=[])
    assert gates.gate(spec, {"text": text}, parsed, ctx, format_only=True).passed


# -- summarisation ---------------------------------------------------------------------------------------


def test_summarisation_gate_passes_within_limit_and_grounded_numbers():
    spec = _summarisation_spec()
    source = "The deal is worth 12000 GBP and closes on 2026-10-20."
    summary = "The deal worth 12000 GBP closes on 2026-10-20."
    result = gates.gate(spec, {"text": source, "max_words": 50}, Parsed(ok=True, value=summary), _pii_ctx())
    assert result.passed


def test_summarisation_gate_fails_over_word_limit():
    spec = _summarisation_spec()
    summary = " ".join(["word"] * 60)
    result = gates.gate(spec, {"text": "irrelevant", "max_words": 50}, Parsed(ok=True, value=summary), _pii_ctx())
    assert not result.passed and result.reason == "over_word_limit"


def test_summarisation_gate_fails_number_not_in_source():
    spec = _summarisation_spec()
    source = "The deal is worth 12000 GBP."
    summary = "The deal is worth 99999 GBP."
    result = gates.gate(spec, {"text": source, "max_words": 50}, Parsed(ok=True, value=summary), _pii_ctx())
    assert not result.passed and result.reason == "number_not_in_source"


# -- entity matching ---------------------------------------------------------------------------------------


def _em_score() -> float:
    return FuzzyMatchBaseline().similarity(_EM_LEFT, _EM_RIGHT)


def test_entity_matching_gate_passes_when_score_is_inside_uncertainty_band():
    score = _em_score()
    spec = _entity_matching_spec()
    ctx = _entity_matching_ctx(band=(score - 0.01, score + 0.01), threshold=score + 0.2)
    item = {"left": _EM_LEFT, "right": _EM_RIGHT}
    # predicted "match" disagrees with the threshold-implied verdict, but the score sits inside the band.
    assert gates.gate(spec, item, Parsed(ok=True, value={"match": True}), ctx).passed


def test_entity_matching_gate_escalates_on_disagreement_outside_band():
    score = _em_score()
    spec = _entity_matching_spec()
    ctx = _entity_matching_ctx(band=(score + 0.2, score + 0.3), threshold=score + 0.2)
    item = {"left": _EM_LEFT, "right": _EM_RIGHT}
    result = gates.gate(spec, item, Parsed(ok=True, value={"match": True}), ctx)
    assert not result.passed and result.reason == "disagrees_with_baseline"


def test_entity_matching_gate_format_only_skips_baseline_comparison():
    score = _em_score()
    spec = _entity_matching_spec()
    ctx = _entity_matching_ctx(band=(score + 0.2, score + 0.3), threshold=score + 0.2)
    item = {"left": _EM_LEFT, "right": _EM_RIGHT}
    assert gates.gate(spec, item, Parsed(ok=True, value={"match": True}), ctx, format_only=True).passed


# -- date-expression resolution ---------------------------------------------------------------------------------------


def test_resolve_date_expressions_covers_every_documented_format():
    ref = "2026-10-14"  # Wednesday
    cases = {
        "meet on 2026-11-01": {"2026-11-01"},
        "meet on 01/11/2026": {"2026-11-01"},
        "meet on 14 October 2026": {"2026-10-14"},
        "meet on 14 October": {"2026-10-14"},
        "meet on October 14, 2026": {"2026-10-14"},
        "let's meet next Monday": {"2026-10-19"},
        "let's meet this Friday": {"2026-10-16"},
        "let's meet tomorrow": {"2026-10-15"},
        "let's meet in a week": {"2026-10-21"},
        "let's meet in two weeks": {"2026-10-28"},
        "let's meet in 3 days": {"2026-10-17"},
        "let's meet on the 20th": {"2026-10-20"},
    }
    for text, expected in cases.items():
        assert expected <= gates.resolve_date_expressions(text, ref), text


def test_resolve_on_day_of_month_rolls_to_next_month_when_day_has_passed():
    from datetime import date

    assert gates.resolve_on_day_of_month(date(2026, 10, 14), 5) == date(2026, 11, 5)
    assert gates.resolve_on_day_of_month(date(2026, 10, 14), 20) == date(2026, 10, 20)


# -- build_gate_context: thresholds tuned on calib ---------------------------------------------------------


def _write_extraction_task(tmp_path):
    root = tmp_path / "extraction"
    root.mkdir()
    items = [
        {
            "id": "1",
            "text": "Hello, this is Acme Widgets Ltd writing in.",
            "gold": {"company_name": "Acme Widgets Ltd"},
        },
        {"id": "2", "text": "Regards, Beta Traders Inc here.", "gold": {"company_name": "Beta Traders Inc"}},
        {"id": "3", "text": "Kind regards from Gamma Supplies LLC.", "gold": {"company_name": "Gamma Supplies LLC"}},
    ]
    with (root / "calib.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    spec = TaskSpec(
        name="extraction",
        kind="extraction",
        fields={"company_name": FieldSpec(type="string", required=True, grounded=True)},
    )
    spec.root = root
    return spec, items


def test_build_gate_context_tunes_extraction_grounding_ratio_from_calib(tmp_path):
    spec, items = _write_extraction_task(tmp_path)
    run = RunDir(tmp_path / "run")
    base = {"task": "extraction", "split": "calib", "pass": "A"}
    # Two calib predictions echo the gold company name verbatim (fuzzy ratio 100); grounding tuning should
    # keep essentially all of them passing, so the resulting threshold is bounded but not overly strict.
    records = [
        {**base, "model_id": "m", "item_id": it["id"], "raw": json.dumps({"company_name": it["gold"]["company_name"]})}
        for it in items
    ]
    run.append_records(PREDICTIONS, records)

    ctx = gates.build_gate_context(run, {"extraction": spec})
    ratio = ctx.thresholds.extraction_grounding_ratio["extraction"]
    assert 80.0 <= ratio <= 100.0


def test_build_gate_context_disables_classification_gate_without_train_split(tmp_path):
    root = tmp_path / "classification"
    root.mkdir()
    spec = TaskSpec(name="classification", kind="classification", labels=["a", "b"], train=None)
    spec.root = root
    run = RunDir(tmp_path / "run")

    ctx = gates.build_gate_context(run, {"classification": spec})
    assert ctx.thresholds.classification_gate_enabled["classification"] is False
    assert "classification" not in ctx.tfidf


def test_build_gate_context_tunes_entity_matching_band_from_calib(tmp_path):
    root = tmp_path / "entity_matching"
    root.mkdir()
    pairs = [
        {"id": "1", "left": _EM_LEFT, "right": _EM_LEFT, "match": True},
        {"id": "2", "left": _EM_LEFT, "right": _EM_RIGHT, "match": False},
    ]
    with (root / "calib.jsonl").open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    spec = TaskSpec(name="entity_matching", kind="entity_matching")
    spec.root = root
    run = RunDir(tmp_path / "run")

    ctx = gates.build_gate_context(run, {"entity_matching": spec})
    assert "entity_matching" in ctx.fuzzy
    assert "entity_matching" in ctx.thresholds.entity_matching_band


def test_country_can_be_grounded_by_the_phone_calling_code() -> None:
    from local_enough.route.gates import _country_in_source

    uk = "You can reach us at 31 Mill Lane, Glasgow.\nPhone: 0044 7700 900282"
    us = "Call me on (403) 555-0197 any afternoon."
    assert _country_in_source("GB", uk, {})
    assert not _country_in_source("US", uk, {})
    assert _country_in_source("US", us, {}) and _country_in_source("CA", us, {})
    assert not _country_in_source("GB", us, {})
