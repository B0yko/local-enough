"""Dataset hygiene: byte-identical regeneration, RFC 2606 domains, allowed phone ranges,
no blocklisted brand names, no "Andrii"/"Boiko", and split sizes across all bundled datasets.
"""

from __future__ import annotations

import filecmp
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "src" / "local_enough" / "data" / "datasets"
EXAMPLES_DIR = REPO_ROOT / "examples" / "custom-task"
SYNTHETIC_TASKS = ["extraction", "pii_redaction", "summarisation", "entity_matching"]


def _load_build_datasets() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_datasets", REPO_ROOT / "scripts" / "build_datasets.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_json_records(directory: Path) -> list[tuple[Path, dict]]:
    records = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append((path, json.loads(line)))
    return records


def _all_strings(obj: object) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        out: list[str] = []
        for v in obj.values():
            out.extend(_all_strings(v))
        return out
    if isinstance(obj, list):
        out = []
        for v in obj:
            out.extend(_all_strings(v))
        return out
    return []


def _dir_tree_equal(a: Path, b: Path) -> list[str]:
    """Return a list of mismatch descriptions; empty means the trees are byte-identical."""
    mismatches = []
    a_files = {p.relative_to(a) for p in a.rglob("*") if p.is_file()}
    b_files = {p.relative_to(b) for p in b.rglob("*") if p.is_file()}
    if a_files != b_files:
        mismatches.append(f"file sets differ: only in {a}: {a_files - b_files}; only in {b}: {b_files - a_files}")
    for rel in sorted(a_files & b_files):
        if not filecmp.cmp(a / rel, b / rel, shallow=False):
            mismatches.append(f"{rel} differs")
    return mismatches


def test_regeneration_is_byte_identical_for_synthetic_tasks(tmp_path: Path) -> None:
    out_dir = tmp_path / "datasets"
    examples_out = tmp_path / "custom-task"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_datasets.py"),
            "--seed",
            "7",
            "--out",
            str(out_dir),
            "--examples-out",
            str(examples_out),
            "--skip-banking77",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    for task in SYNTHETIC_TASKS:
        mismatches = _dir_tree_equal(DATASETS_DIR / task, out_dir / task)
        assert mismatches == [], f"{task}: {mismatches}"
    mismatches = _dir_tree_equal(EXAMPLES_DIR, examples_out)
    assert mismatches == [], f"custom-task: {mismatches}"


@pytest.mark.skipif("LOCAL_ENOUGH_BANKING77_DIR" not in __import__("os").environ, reason="needs BANKING77 cache")
def test_regeneration_is_byte_identical_for_classification(tmp_path: Path) -> None:
    import os

    out_dir = tmp_path / "datasets"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_datasets.py"),
            "--seed",
            "7",
            "--out",
            str(out_dir),
            "--banking77-dir",
            os.environ["LOCAL_ENOUGH_BANKING77_DIR"],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    mismatches = _dir_tree_equal(DATASETS_DIR / "classification", out_dir / "classification")
    assert mismatches == []


def test_split_sizes() -> None:
    expected = {
        "extraction": {"calib.jsonl": 100, "test.jsonl": 200},
        "classification": {"calib.jsonl": 154, "test.jsonl": 308, "train.jsonl": 10003},
        "pii_redaction": {"calib.jsonl": 100, "test.jsonl": 200},
        "summarisation": {
            "calib.jsonl": 40,
            "test.jsonl": 80,
            "judge_calib.jsonl": 200,
            "judge_holdout.jsonl": 200,
            "judge_holdout_transcripts.jsonl": 40,
        },
        "entity_matching": {"calib.jsonl": 100, "test.jsonl": 200},
    }
    for task, files in expected.items():
        for filename, n in files.items():
            path = DATASETS_DIR / task / filename
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert len(lines) == n, f"{task}/{filename}: expected {n} rows, got {len(lines)}"

    custom_expected = {"calib.jsonl": 30, "test.jsonl": 60, "train.jsonl": 150}
    for filename, n in custom_expected.items():
        lines = [line for line in (EXAMPLES_DIR / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == n


_ALLOWED_DOMAIN_RE = re.compile(r"^([a-z0-9-]+\.)*example\.(com|org|net)$", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)*)")

_UK_PHONE_RE = re.compile(
    r"(\+44\s?7700\s?900\d{3}|0044\s?7700\s?900\d{3}|\(07700\)\s?900[- ]?\d{3}|07700[- ]?900[- ]?\d{3})"
)
_US_PHONE_RE = re.compile(
    r"(\+1[\s.-]?\d{3}[\s.-]?555[\s.-]?01\d{2}|1[\s.-]\d{3}[\s.-]555[\s.-]01\d{2}"
    r"|\(\d{3}\)\s?555[\s.-]?01\d{2}|\b\d{3}[.-]555[.-]01\d{2}\b)"
)
_PHONE_ANCHOR_RE = re.compile(r"(7700[ -]?900\d{3}|555[ .-]?01\d{2})")


def _check_domains_and_phones(text: str, source: str) -> None:
    for domain in _EMAIL_RE.findall(text):
        assert _ALLOWED_DOMAIN_RE.match(domain), f"{source}: disallowed email domain {domain!r}"
    for anchor_match in _PHONE_ANCHOR_RE.finditer(text):
        window = text[max(0, anchor_match.start() - 20) : anchor_match.end() + 5]
        assert _UK_PHONE_RE.search(window) or _US_PHONE_RE.search(window), (
            f"{source}: phone-like text near {anchor_match.group()!r} does not match an allowed format: {window!r}"
        )


def test_only_rfc2606_domains_and_allowed_phone_ranges() -> None:
    # classification (BANKING77) is real public data, not synthetic, so the fictional-value rules
    # (which apply to data this project generates) do not apply to it.
    for task_dir in [*[DATASETS_DIR / t for t in SYNTHETIC_TASKS], EXAMPLES_DIR]:
        for path, record in _all_json_records(task_dir):
            for s in _all_strings(record):
                _check_domains_and_phones(s, f"{path.relative_to(REPO_ROOT)}#{record.get('id')}")


def test_extraction_gold_phones_are_e164_in_allowed_ranges() -> None:
    uk = re.compile(r"^\+447700900\d{3}$")
    us = re.compile(r"^\+1\d{3}55501\d{2}$")
    for path, record in _all_json_records(DATASETS_DIR / "extraction"):
        phone = record["gold"]["phone"]
        assert uk.match(phone) or us.match(phone), f"{path.name}#{record['id']}: bad E.164 phone {phone!r}"


def test_no_blocklisted_brand_names_in_synthetic_data() -> None:
    # Again scoped to synthetic data: BANKING77 is real public text and may legitimately mention
    # real brands (e.g. the official "apple_pay_or_google_pay" intent label).
    bd = _load_build_datasets()
    failures = []
    for task_dir in [*[DATASETS_DIR / t for t in SYNTHETIC_TASKS], EXAMPLES_DIR]:
        for path, record in _all_json_records(task_dir):
            for s in _all_strings(record):
                try:
                    bd.check_no_blocklisted_brand(s)
                except AssertionError as exc:
                    failures.append(f"{path.relative_to(REPO_ROOT)}#{record.get('id')}: {exc}")
    assert failures == []


def test_no_owner_name_anywhere_including_banking77() -> None:
    for task_dir in [*[DATASETS_DIR / t for t in [*SYNTHETIC_TASKS, "classification"]], EXAMPLES_DIR]:
        for path, record in _all_json_records(task_dir):
            for s in _all_strings(record):
                lowered = s.lower()
                assert "andrii" not in lowered and "boiko" not in lowered, (
                    f"{path.relative_to(REPO_ROOT)}#{record.get('id')}: contains owner's name"
                )
