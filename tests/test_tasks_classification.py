"""classification: parse/score/aggregate/validate_item on hand-made examples."""

from __future__ import annotations

from local_enough.tasks import classification as mod
from local_enough.tasks.base import TaskSpec

SPEC = TaskSpec(name="t", kind="classification", labels=["billing", "bug_report", "how_to"])


def test_parse_exact_label() -> None:
    parsed = mod.parse(SPEC, "billing")
    assert parsed.ok
    assert parsed.value == "billing"
    assert parsed.content == "billing"


def test_parse_is_case_insensitive_and_strips_wrapping() -> None:
    parsed = mod.parse(SPEC, '  "Bug_Report".  ')
    assert parsed.ok
    assert parsed.value == "bug_report"  # canonical casing from the label set


def test_parse_rejects_label_outside_set() -> None:
    parsed = mod.parse(SPEC, "not_a_label")
    assert not parsed.ok


def test_parse_uses_first_line_ignoring_reasoning() -> None:
    parsed = mod.parse(SPEC, "<think>hmm</think>how_to\nextra trailing text")
    assert parsed.ok
    assert parsed.value == "how_to"


def test_render_messages_includes_all_labels_and_text() -> None:
    item = {"id": "x", "text": "Where is my refund?"}
    messages = mod.render_messages(SPEC, item)
    joined = " ".join(m["content"] for m in messages)
    assert "billing" in joined and "bug_report" in joined and "how_to" in joined
    assert "Where is my refund?" in joined


def test_score_and_aggregate_on_hand_made_examples() -> None:
    items = [
        {"id": "1", "text": "a", "label": "billing"},
        {"id": "2", "text": "b", "label": "bug_report"},
        {"id": "3", "text": "c", "label": "how_to"},
        {"id": "4", "text": "d", "label": "billing"},
        {"id": "5", "text": "e", "label": "how_to"},
    ]
    # 1: correct, 2: wrong (predicted how_to), 3: correct, 4: invalid output, 5: correct
    predictions = ["billing", "how_to", "how_to", "not_a_label", "how_to"]
    scores = [mod.score(SPEC, item, mod.parse(SPEC, pred)) for item, pred in zip(items, predictions, strict=True)]

    agg = mod.aggregate(SPEC, scores)
    assert agg["accuracy"] == 3 / 5
    assert agg["invalid_output_rate"] == 1 / 5
    assert agg["primary"] == agg["accuracy"]
    assert 0.0 <= agg["macro_f1"] <= 1.0


def test_aggregate_empty_is_nan() -> None:
    agg = mod.aggregate(SPEC, [])
    assert agg["accuracy"] != agg["accuracy"]  # NaN


def test_aggregate_perfect_predictions() -> None:
    items = [{"id": str(i), "text": "x", "label": "billing"} for i in range(4)]
    scores = [mod.score(SPEC, item, mod.parse(SPEC, "billing")) for item in items]
    agg = mod.aggregate(SPEC, scores)
    assert agg["accuracy"] == 1.0
    assert agg["macro_f1"] == 1.0
    assert agg["invalid_output_rate"] == 0.0


def test_validate_item() -> None:
    assert mod.validate_item(SPEC, {"id": "1", "text": "x", "label": "billing"}) == []
    errors = mod.validate_item(SPEC, {"id": "1", "text": "x", "label": "not_in_set"})
    assert errors and "not in task label set" in errors[0]
    errors2 = mod.validate_item(SPEC, {"text": "x"})
    assert any("id" in e for e in errors2) and any("label" in e for e in errors2)


def test_item_from_input_and_user_input_roundtrip() -> None:
    item = {"id": "x", "text": "hello"}
    assert mod.user_input(SPEC, item) == "hello"
    rebuilt = mod.item_from_input(SPEC, "hello")
    assert rebuilt["text"] == "hello"


def test_label_with_trailing_punctuation_in_the_label_set_is_valid() -> None:
    from local_enough.tasks import classification
    from local_enough.tasks.base import TaskSpec

    spec = TaskSpec(name="c", kind="classification", labels=["reverted_card_payment?", "card_arrival"])
    assert classification.parse(spec, "reverted_card_payment?").content == "reverted_card_payment?"
    assert classification.parse(spec, "`card_arrival`.").content == "card_arrival"
    assert classification.parse(spec, "reverted_card_payment").content == "reverted_card_payment?"
