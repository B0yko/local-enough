"""pii_redaction dataset: span offset exactness and the ~15% PII-free document rate."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "src" / "local_enough" / "data" / "datasets" / "pii_redaction"
ALLOWED_TYPES = {"PERSON", "EMAIL", "PHONE", "IBAN", "ADDRESS", "DATE_OF_BIRTH"}


def _records() -> list[dict]:
    records = []
    for filename in ["calib.jsonl", "test.jsonl"]:
        for line in (TASK_DIR / filename).read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def test_span_offsets_are_exact_and_within_bounds() -> None:
    for record in _records():
        text = record["text"]
        for span in record["spans"]:
            start, end = span["start"], span["end"]
            assert 0 <= start < end <= len(text), f"{record['id']}: span {span} out of bounds"
            assert span["type"] in ALLOWED_TYPES, f"{record['id']}: unexpected span type {span['type']!r}"
            # The generator asserts this at build time too; re-verify independently here.
            assert text[start:end], f"{record['id']}: empty span text"


def test_spans_have_at_most_six_entries() -> None:
    for record in _records():
        assert 0 <= len(record["spans"]) <= 6, f"{record['id']}: {len(record['spans'])} spans"


def test_about_15_percent_of_documents_are_pii_free() -> None:
    records = _records()
    pii_free = sum(1 for r in records if not r["spans"])
    rate = pii_free / len(records)
    assert 0.08 <= rate <= 0.22, f"PII-free rate {rate:.3f} is far from the ~15% target"


def test_gold_spans_do_not_overlap_within_a_document() -> None:
    for record in _records():
        spans = sorted(record["spans"], key=lambda s: s["start"])
        for a, b in itertools.pairwise(spans):
            assert a["end"] <= b["start"], f"{record['id']}: overlapping spans {a} and {b}"


def test_hard_negative_order_ids_are_never_gold_spans() -> None:
    for record in _records():
        gold_texts = {record["text"][s["start"] : s["end"]] for s in record["spans"]}
        for marker in ("ORD-", "INV-"):
            for token in record["text"].split():
                cleaned = token.strip(".,;:")
                if cleaned.startswith(marker):
                    assert cleaned not in gold_texts, f"{record['id']}: {cleaned!r} should not be a gold span"
