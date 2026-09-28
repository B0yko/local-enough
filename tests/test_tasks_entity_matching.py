"""entity_matching: parse/score/aggregate/validate_item on hand-made examples."""

from __future__ import annotations

import json

from local_enough.tasks import entity_matching as mod
from local_enough.tasks.base import TaskSpec

SPEC = TaskSpec(name="t", kind="entity_matching")


def item(match: bool) -> dict:
    return {"id": "x", "left": {"name": "Acme Ltd"}, "right": {"name": "Acme Limited"}, "match": match}


def test_parse_valid_match_object() -> None:
    parsed = mod.parse(SPEC, '{"match": true}')
    assert parsed.ok
    assert parsed.value == {"match": True}
    assert parsed.content == json.dumps({"match": True})


def test_parse_rejects_missing_match_field() -> None:
    assert not mod.parse(SPEC, "{}").ok


def test_parse_rejects_non_boolean_match() -> None:
    assert not mod.parse(SPEC, '{"match": "yes"}').ok


def test_user_input_and_item_from_input_roundtrip() -> None:
    it = {"id": "x", "left": {"name": "A"}, "right": {"name": "B"}}
    raw = mod.user_input(SPEC, it)
    rebuilt = mod.item_from_input(SPEC, raw)
    assert rebuilt["left"] == {"name": "A"}
    assert rebuilt["right"] == {"name": "B"}


def test_score_and_aggregate_confusion_matrix() -> None:
    items = [item(True), item(True), item(False), item(False)]
    # tp, fn (predicted false when true), fp (predicted true when false), tn
    predictions = ['{"match": true}', '{"match": false}', '{"match": true}', '{"match": false}']
    scores = [mod.score(SPEC, it, mod.parse(SPEC, p)) for it, p in zip(items, predictions, strict=True)]
    agg = mod.aggregate(SPEC, scores)
    assert agg["precision"] == 0.5  # 1 tp / (1 tp + 1 fp)
    assert agg["recall"] == 0.5  # 1 tp / (1 tp + 1 fn)
    assert agg["f1"] == 0.5
    assert agg["primary"] == agg["f1"]


def test_invalid_output_counts_as_no_match_and_invalid() -> None:
    it = item(True)
    score = mod.score(SPEC, it, mod.parse(SPEC, "garbage"))
    assert not score.valid
    assert score.stats["fn"] == 1  # gold True, treated as predicted False
    agg = mod.aggregate(SPEC, [score])
    assert agg["invalid_output_rate"] == 1.0
    assert agg["recall"] == 0.0


def test_perfect_predictions_give_f1_one() -> None:
    items = [item(True), item(False), item(True)]
    scores = [mod.score(SPEC, it, mod.parse(SPEC, json.dumps({"match": it["match"]}))) for it in items]
    agg = mod.aggregate(SPEC, scores)
    assert agg["f1"] == 1.0
    assert agg["precision"] == 1.0
    assert agg["recall"] == 1.0


def test_validate_item() -> None:
    assert mod.validate_item(SPEC, item(True)) == []
    errors = mod.validate_item(SPEC, {"id": "x", "left": {}, "match": "not-bool"})
    assert any("right" in e for e in errors)
    assert any("match" in e for e in errors)
