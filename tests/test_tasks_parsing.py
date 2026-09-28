"""Parser edge cases: code fences, <think> blocks (closed and unterminated), trailing text."""

from __future__ import annotations

import pytest

from local_enough.tasks.parsing import first_json_value, first_line_label, strip_wrappers


def test_strip_wrappers_removes_closed_think_block() -> None:
    assert strip_wrappers("<think>reasoning about it</think>final answer") == "final answer"


def test_strip_wrappers_removes_unterminated_think_block() -> None:
    assert strip_wrappers("<think>still reasoning, ran out of tokens") == ""


def test_strip_wrappers_removes_unterminated_think_before_answer() -> None:
    # An unterminated <think> swallows everything after it, including a real answer -
    # this is the harness-bug case (empty content from a hit cap).
    assert strip_wrappers("<think>reasoning\nlabel") == ""


def test_strip_wrappers_extracts_code_fence() -> None:
    text = 'noise before\n```json\n{"a": 1}\n```\nnoise after'
    assert strip_wrappers(text) == '{"a": 1}'


def test_strip_wrappers_fence_without_language() -> None:
    assert strip_wrappers("```\nplain\n```") == "plain"


def test_strip_wrappers_plain_text_untouched() -> None:
    assert strip_wrappers("  just a label  ") == "just a label"


def test_strip_wrappers_think_then_fence() -> None:
    text = "<think>internal</think>```json\n[1, 2]\n```"
    assert strip_wrappers(text) == "[1, 2]"


def test_first_line_label_strips_quotes_backticks_punctuation() -> None:
    assert first_line_label("  `card_arrival`.  \n\nextra stuff") == "card_arrival"


def test_first_line_label_plain() -> None:
    assert first_line_label("bug_report") == "bug_report"


def test_first_line_label_skips_blank_lines() -> None:
    assert first_line_label("\n\n  how_to  \nsecond line") == "how_to"


def test_first_line_label_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="no label"):
        first_line_label("<think>only reasoning, nothing else")


def test_first_json_value_finds_object_after_prose() -> None:
    assert first_json_value('Sure, here it is: {"match": true} - hope that helps') == {"match": True}


def test_first_json_value_finds_list() -> None:
    assert first_json_value('prefix [{"type": "PERSON", "text": "Jane"}] suffix') == [
        {"type": "PERSON", "text": "Jane"}
    ]


def test_first_json_value_skips_invalid_brace_before_real_json() -> None:
    # A stray '{' with no valid JSON after it should not stop the scan.
    assert first_json_value('{ not json } then {"ok": 1}') == {"ok": 1}


def test_first_json_value_nested_structures() -> None:
    value = first_json_value('{"a": [1, 2, {"b": 3}], "c": null}')
    assert value == {"a": [1, 2, {"b": 3}], "c": None}


def test_first_json_value_raises_when_absent() -> None:
    with pytest.raises(ValueError, match="no JSON"):
        first_json_value("no json anywhere in this response")


def test_first_json_value_through_fence_and_think() -> None:
    text = '<think>let me think</think>```json\n{"match": false}\n```'
    assert first_json_value(text) == {"match": False}
