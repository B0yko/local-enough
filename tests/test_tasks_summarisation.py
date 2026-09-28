"""summarisation: parse/score/aggregate on hand-made examples, with and without a judge verdict."""

from __future__ import annotations

import math

from local_enough.tasks import summarisation as mod
from local_enough.tasks.base import TaskSpec

SPEC = TaskSpec(name="t", kind="summarisation", max_words=120)

REQUIRED_FACTS = [
    {
        "index": 0,
        "kind": "action_item",
        "fact": "Priya will send the proposal by Friday.",
        "key_tokens": {"owner": "Priya", "date": "Friday"},
    },
    {
        "index": 1,
        "kind": "decision",
        "fact": "The team decided to extend the pilot by 14 April.",
        "key_tokens": {"date": "14 April"},
    },
    {
        "index": 2,
        "kind": "amount",
        "fact": "Budget for the rollout was set at $18,000.",
        "key_tokens": {"amount": "$18,000"},
    },
]


def item(max_words: int = 120, facts: list[dict] | None = None) -> dict:
    return {"id": "x", "text": "transcript...", "max_words": max_words, "required_facts": facts or REQUIRED_FACTS}


def test_parse_strips_wrappers_and_keeps_plain_text() -> None:
    parsed = mod.parse(SPEC, "<think>drafting</think>Priya will send the proposal by Friday.")
    assert parsed.ok
    assert parsed.value == "Priya will send the proposal by Friday."


def test_parse_rejects_empty_summary() -> None:
    assert not mod.parse(SPEC, "<think>only reasoning</think>").ok


def test_score_without_judge_uses_key_token_recall_and_nan_pass_rate() -> None:
    summary = "Priya will send the proposal by Friday. Budget for the rollout was set at $18,000."
    parsed = mod.parse(SPEC, summary)
    score = mod.score(SPEC, item(), parsed)
    assert score.stats["fact_present_count"] == 2  # owner+date found, amount found; decision's date not mentioned
    assert score.stats["judged"] is False
    agg = mod.aggregate(SPEC, [score])
    assert math.isnan(agg["pass_rate"])
    assert abs(agg["fact_recall"] - 2 / 3) < 1e-9


def test_score_within_word_limit() -> None:
    long_summary = " ".join(["word"] * 130)
    parsed = mod.parse(SPEC, long_summary)
    score = mod.score(SPEC, item(max_words=120), parsed)
    assert score.stats["within_word_limit"] is False


def test_judge_verdict_faithful_passes() -> None:
    parsed = mod.parse(SPEC, "faithful summary text")
    judge = {
        "facts": [{"index": i, "present": True, "correct": True} for i in range(3)],
        "unsupported_claims": [],
    }
    score = mod.score(SPEC, item(), parsed, extra={"judge": judge})
    assert score.stats["judged"] is True
    assert score.stats["judged_pass"] is True
    agg = mod.aggregate(SPEC, [score])
    assert agg["pass_rate"] == 1.0


def test_judge_verdict_unsupported_claim_fails_even_if_all_facts_correct() -> None:
    parsed = mod.parse(SPEC, "summary with an invented commitment")
    judge = {
        "facts": [{"index": i, "present": True, "correct": True} for i in range(3)],
        "unsupported_claims": ["invented commitment"],
    }
    score = mod.score(SPEC, item(), parsed, extra={"judge": judge})
    assert score.stats["judged_pass"] is False


def test_judge_verdict_ceil_0_8_boundary_n5() -> None:
    facts5 = [{"index": i, "kind": "amount", "fact": f"fact {i}", "key_tokens": {"amount": f"${i}"}} for i in range(5)]
    parsed = mod.parse(SPEC, "summary")
    # 4 of 5 present-and-correct: ceil(0.8*5) = 4, so this passes.
    judge_pass = {
        "facts": [{"index": i, "present": True, "correct": i != 4} for i in range(5)],
        "unsupported_claims": [],
    }
    score_pass = mod.score(SPEC, item(facts=facts5), parsed, extra={"judge": judge_pass})
    assert score_pass.stats["judged_pass"] is True

    # 3 of 5 present-and-correct: below ceil(0.8*5) = 4, so this fails.
    judge_fail = {
        "facts": [{"index": i, "present": True, "correct": i < 3} for i in range(5)],
        "unsupported_claims": [],
    }
    score_fail = mod.score(SPEC, item(facts=facts5), parsed, extra={"judge": judge_fail})
    assert score_fail.stats["judged_pass"] is False


def test_judge_verdict_over_word_limit_fails_regardless_of_facts() -> None:
    long_summary = " ".join(["word"] * 130)
    parsed = mod.parse(SPEC, long_summary)
    judge = {
        "facts": [{"index": i, "present": True, "correct": True} for i in range(3)],
        "unsupported_claims": [],
    }
    score = mod.score(SPEC, item(max_words=120), parsed, extra={"judge": judge})
    assert score.stats["judged_pass"] is False


def test_aggregate_mixes_judged_and_unjudged_items() -> None:
    parsed = mod.parse(SPEC, "text")
    judged = mod.score(
        SPEC,
        item(),
        parsed,
        extra={
            "judge": {
                "facts": [{"index": i, "present": True, "correct": True} for i in range(3)],
                "unsupported_claims": [],
            }
        },
    )
    unjudged = mod.score(SPEC, item(), parsed)
    agg = mod.aggregate(SPEC, [judged, unjudged])
    assert agg["pass_rate"] == 1.0  # only the judged item counts towards pass_rate


def test_invalid_output_is_not_within_limit_and_counted_invalid() -> None:
    score = mod.score(SPEC, item(), mod.parse(SPEC, "<think>only reasoning</think>"))
    assert not score.valid
    agg = mod.aggregate(SPEC, [score])
    assert agg["invalid_output_rate"] == 1.0
    assert agg["within_word_limit"] == 0.0


def test_validate_item() -> None:
    assert mod.validate_item(SPEC, item()) == []
    errors = mod.validate_item(SPEC, {"id": "x", "text": "t", "required_facts": [{"fact": "f"}]})
    assert any("max_words" in e for e in errors)
    assert any("required_facts" in e for e in errors)
