"""Verdict parsing edge cases, decide_pass boundaries, and prompt rendering."""

from __future__ import annotations

import json

from local_enough.judge import judge


def _verdict(facts: list[tuple[int, bool, bool]], claims: list[str] | None = None) -> str:
    return json.dumps(
        {
            "facts": [{"index": i, "present": p, "correct": c} for i, p, c in facts],
            "unsupported_claims": claims or [],
        }
    )


def test_parse_verdict_valid_shape_is_sorted_by_index() -> None:
    raw = _verdict([(1, True, True), (0, True, False)])
    verdict = judge.parse_verdict(raw, 2)
    assert verdict is not None
    assert [f["index"] for f in verdict["facts"]] == [0, 1]
    assert verdict["unsupported_claims"] == []


def test_parse_verdict_strips_code_fence_and_think_block() -> None:
    raw = "<think>let me check</think>```json\n" + _verdict([(0, True, True)]) + "\n```"
    assert judge.parse_verdict(raw, 1) is not None


def test_parse_verdict_not_json_is_none() -> None:
    assert judge.parse_verdict("I think it passes.", 1) is None


def test_parse_verdict_missing_facts_key_is_none() -> None:
    assert judge.parse_verdict(json.dumps({"unsupported_claims": []}), 1) is None


def test_parse_verdict_missing_unsupported_claims_key_is_none() -> None:
    assert judge.parse_verdict(json.dumps({"facts": [{"index": 0, "present": True, "correct": True}]}), 1) is None


def test_parse_verdict_duplicate_index_is_none() -> None:
    raw = _verdict([(0, True, True), (0, False, False)])
    assert judge.parse_verdict(raw, 1) is None


def test_parse_verdict_index_out_of_range_is_none() -> None:
    raw = _verdict([(0, True, True), (2, True, True)])
    assert judge.parse_verdict(raw, 2) is None


def test_parse_verdict_missing_index_coverage_is_none() -> None:
    # n_facts=2 but only index 0 reported
    raw = _verdict([(0, True, True)])
    assert judge.parse_verdict(raw, 2) is None


def test_parse_verdict_negative_index_is_none() -> None:
    raw = _verdict([(-1, True, True)])
    assert judge.parse_verdict(raw, 1) is None


def test_parse_verdict_non_bool_present_is_none() -> None:
    raw = json.dumps({"facts": [{"index": 0, "present": "yes", "correct": True}], "unsupported_claims": []})
    assert judge.parse_verdict(raw, 1) is None


def test_parse_verdict_bool_index_is_none() -> None:
    raw = json.dumps({"facts": [{"index": True, "present": True, "correct": True}], "unsupported_claims": []})
    assert judge.parse_verdict(raw, 1) is None


def test_parse_verdict_non_string_unsupported_claim_is_none() -> None:
    raw = json.dumps({"facts": [{"index": 0, "present": True, "correct": True}], "unsupported_claims": [1]})
    assert judge.parse_verdict(raw, 1) is None


def test_parse_verdict_non_dict_fact_entry_is_none() -> None:
    raw = json.dumps({"facts": ["not a dict"], "unsupported_claims": []})
    assert judge.parse_verdict(raw, 1) is None


def test_parse_verdict_top_level_not_object_is_none() -> None:
    assert judge.parse_verdict(json.dumps([1, 2, 3]), 1) is None


def test_parse_verdict_n_facts_zero_accepts_empty_facts() -> None:
    raw = _verdict([])
    verdict = judge.parse_verdict(raw, 0)
    assert verdict == {"facts": [], "unsupported_claims": []}


def test_parse_verdict_keeps_unsupported_claims_text() -> None:
    raw = _verdict([(0, True, True)], claims=["invented a follow-up call"])
    verdict = judge.parse_verdict(raw, 1)
    assert verdict is not None
    assert verdict["unsupported_claims"] == ["invented a follow-up call"]


def test_decide_pass_invalid_verdict_is_always_fail() -> None:
    assert judge.decide_pass(None, 5, 10, 120) is False


def test_decide_pass_boundary_n4_needs_4_of_4() -> None:
    # ceil(0.8*4) == 4
    passing = _facts_verdict(4, present_correct=4)
    failing = _facts_verdict(4, present_correct=3)
    assert judge.decide_pass(passing, 4, 10, 120) is True
    assert judge.decide_pass(failing, 4, 10, 120) is False


def test_decide_pass_boundary_n5_needs_4_of_5() -> None:
    # ceil(0.8*5) == 4
    passing = _facts_verdict(5, present_correct=4)
    failing = _facts_verdict(5, present_correct=3)
    assert judge.decide_pass(passing, 5, 10, 120) is True
    assert judge.decide_pass(failing, 5, 10, 120) is False


def test_decide_pass_boundary_n6_needs_5_of_6() -> None:
    # ceil(0.8*6) == 5
    passing = _facts_verdict(6, present_correct=5)
    failing = _facts_verdict(6, present_correct=4)
    assert judge.decide_pass(passing, 6, 10, 120) is True
    assert judge.decide_pass(failing, 6, 10, 120) is False


def test_decide_pass_fails_on_any_unsupported_claim() -> None:
    verdict = _facts_verdict(5, present_correct=5, unsupported=["extra thing"])
    assert judge.decide_pass(verdict, 5, 10, 120) is False


def test_decide_pass_fails_over_word_limit() -> None:
    verdict = _facts_verdict(5, present_correct=5)
    assert judge.decide_pass(verdict, 5, 121, 120) is False
    assert judge.decide_pass(verdict, 5, 120, 120) is True


def test_present_and_correct_count_requires_both() -> None:
    verdict = {
        "facts": [
            {"index": 0, "present": True, "correct": True},
            {"index": 1, "present": True, "correct": False},
            {"index": 2, "present": False, "correct": False},
        ],
        "unsupported_claims": [],
    }
    assert judge.present_and_correct_count(verdict) == 1


def test_render_judge_messages_includes_transcript_facts_and_summary() -> None:
    facts = [{"index": 0, "fact": "Amara pays 100 by Friday", "kind": "action_item", "key_tokens": {}}]
    messages = judge.render_judge_messages("the transcript text", facts, "the summary text")
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    user_content = messages[-1]["content"]
    assert "the transcript text" in user_content
    assert "0. Amara pays 100 by Friday" in user_content
    assert "the summary text" in user_content


def test_template_file_exists() -> None:
    assert judge.template_file().is_file()


def _facts_verdict(n: int, present_correct: int, unsupported: list[str] | None = None) -> dict[str, object]:
    facts = [{"index": i, "present": i < present_correct, "correct": i < present_correct} for i in range(n)]
    return {"facts": facts, "unsupported_claims": unsupported or []}
