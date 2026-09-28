"""classification dataset: committed BANKING77 subset shape (offline, no network needed)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "src" / "local_enough" / "data" / "datasets" / "classification"


def _records(filename: str) -> list[dict]:
    return [json.loads(line) for line in (TASK_DIR / filename).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_calib_has_two_per_intent_and_test_has_four() -> None:
    calib = _records("calib.jsonl")
    test = _records("test.jsonl")
    assert len(calib) == 154
    assert len(test) == 308
    calib_counts = Counter(r["label"] for r in calib)
    test_counts = Counter(r["label"] for r in test)
    assert len(calib_counts) == 77
    assert len(test_counts) == 77
    assert set(calib_counts.values()) == {2}
    assert set(test_counts.values()) == {4}


def test_calib_and_test_are_disjoint() -> None:
    calib_texts = {r["text"] for r in _records("calib.jsonl")}
    test_texts = {r["text"] for r in _records("test.jsonl")}
    assert calib_texts.isdisjoint(test_texts)


def test_train_is_the_full_10003_row_split() -> None:
    train = _records("train.jsonl")
    assert len(train) == 10003
    assert len({r["label"] for r in train}) == 77


def test_license_file_is_the_full_cc_by_text() -> None:
    text = (TASK_DIR / "LICENSE-BANKING77.txt").read_text(encoding="utf-8")
    assert "Creative Commons Attribution 4.0 International" in text
    assert len(text.splitlines()) > 100


def test_task_yaml_declares_all_77_labels() -> None:
    import yaml

    data = yaml.safe_load((TASK_DIR / "task.yaml").read_text(encoding="utf-8"))
    assert len(data["labels"]) == 77
    assert data["kind"] == "classification"
    assert data["card"]["licence"] == "CC-BY-4.0"
