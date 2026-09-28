"""summarisation dataset: sentence-repetition cap and judge-set label logic."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "src" / "local_enough" / "data" / "datasets" / "summarisation"


def _records(filename: str) -> list[dict]:
    return [json.loads(line) for line in (TASK_DIR / filename).read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalized_sentences(text: str) -> set[str]:
    parts = re.split(r"[.!?]+", text)
    out = set()
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip().lower()
        if p:
            out.add(p)
    return out


def test_transcript_word_counts_are_in_range() -> None:
    for filename in ["calib.jsonl", "test.jsonl", "judge_holdout_transcripts.jsonl"]:
        for record in _records(filename):
            n_words = len(record["text"].split())
            assert 600 <= n_words <= 1000, f"{filename}#{record['id']}: {n_words} words"


def test_required_facts_count_is_in_range() -> None:
    for filename in ["calib.jsonl", "test.jsonl", "judge_holdout_transcripts.jsonl"]:
        for record in _records(filename):
            assert 4 <= len(record["required_facts"]) <= 6


def test_no_sentence_appears_in_more_than_20_percent_of_transcripts() -> None:
    transcripts = [
        *_records("calib.jsonl"),
        *_records("test.jsonl"),
        *_records("judge_holdout_transcripts.jsonl"),
    ]
    n = len(transcripts)
    counts: dict[str, int] = {}
    for record in transcripts:
        for sentence in _normalized_sentences(record["text"]):
            counts[sentence] = counts.get(sentence, 0) + 1
    worst_sentence, worst_count = max(counts.items(), key=lambda kv: kv[1])
    assert worst_count / n <= 0.20, f"{worst_sentence!r} appears in {worst_count}/{n} transcripts"


def _transcripts_by_id(filename: str) -> dict[str, dict]:
    return {r["id"]: r for r in _records(filename)}


def _check_judge_set(summary_file: str, transcript_lookup: dict[str, dict]) -> None:
    for record in _records(summary_file):
        transcript = transcript_lookup[record["transcript_id"]]
        n = len(transcript["required_facts"])
        present_correct = sum(1 for f in record["facts"] if f["present"] and f["correct"])
        within_limit = len(record["summary"].split()) <= transcript["max_words"]
        expected_pass = present_correct >= math.ceil(0.8 * n) and record["unsupported_claims"] == 0 and within_limit
        assert record["label_pass"] == expected_pass, (
            f"{summary_file}#{record['id']}: label_pass={record['label_pass']} but recomputed {expected_pass} "
            f"(present_correct={present_correct}, n={n}, unsupported={record['unsupported_claims']}, "
            f"within_limit={within_limit})"
        )
        assert {f["index"] for f in record["facts"]} == set(range(n))
        assert record["style"] in (0, 1, 2)
        assert record["variant"] in {
            "faithful",
            "fact_dropped",
            "wrong_number_or_date",
            "wrong_owner",
            "invented_commitment",
        }


def test_judge_calib_label_pass_matches_the_code_rule() -> None:
    _check_judge_set("judge_calib.jsonl", _transcripts_by_id("calib.jsonl"))


def test_judge_holdout_label_pass_matches_the_code_rule() -> None:
    _check_judge_set("judge_holdout.jsonl", _transcripts_by_id("judge_holdout_transcripts.jsonl"))


def test_judge_sets_have_five_variants_per_transcript() -> None:
    for summary_file, transcript_file in [
        ("judge_calib.jsonl", "calib.jsonl"),
        ("judge_holdout.jsonl", "judge_holdout_transcripts.jsonl"),
    ]:
        transcripts = _transcripts_by_id(transcript_file)
        by_transcript: dict[str, int] = {}
        for record in _records(summary_file):
            by_transcript[record["transcript_id"]] = by_transcript.get(record["transcript_id"], 0) + 1
        assert set(by_transcript) == set(transcripts)
        assert all(count == 5 for count in by_transcript.values())


def test_invented_commitment_variant_always_has_an_unsupported_claim() -> None:
    for summary_file in ["judge_calib.jsonl", "judge_holdout.jsonl"]:
        for record in _records(summary_file):
            if record["variant"] == "invented_commitment":
                assert record["unsupported_claims"] == 1
                assert record["label_pass"] is False
