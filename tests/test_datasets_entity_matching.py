"""entity_matching dataset: ~40% match ratio and record shape."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "src" / "local_enough" / "data" / "datasets" / "entity_matching"
REQUIRED_FIELDS = {"name", "street", "postcode", "city", "country", "domain", "vat_id", "phone"}


def _records(filename: str) -> list[dict]:
    return [json.loads(line) for line in (TASK_DIR / filename).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_match_ratio_is_about_40_percent() -> None:
    for filename in ["calib.jsonl", "test.jsonl"]:
        records = _records(filename)
        matches = sum(1 for r in records if r["match"])
        ratio = matches / len(records)
        assert abs(ratio - 0.4) < 0.02, f"{filename}: match ratio {ratio:.3f} far from 40%"


def test_left_and_right_have_the_expected_fields() -> None:
    for filename in ["calib.jsonl", "test.jsonl"]:
        for record in _records(filename):
            assert set(record["left"]) == REQUIRED_FIELDS
            assert set(record["right"]) == REQUIRED_FIELDS


def test_calib_and_test_do_not_overlap_by_id() -> None:
    calib_ids = {r["id"] for r in _records("calib.jsonl")}
    test_ids = {r["id"] for r in _records("test.jsonl")}
    assert calib_ids.isdisjoint(test_ids)


def test_negative_pairs_are_not_trivially_identical() -> None:
    for filename in ["calib.jsonl", "test.jsonl"]:
        for record in _records(filename):
            if not record["match"]:
                assert record["left"] != record["right"], f"{record['id']}: identical records marked non-match"
