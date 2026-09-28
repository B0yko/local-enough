from __future__ import annotations

from pathlib import Path

import pytest

from local_enough import evaluate
from local_enough.bench.rundir import RunDir
from local_enough.tasks import registry


def _classification_spec():  # type: ignore[no-untyped-def]
    return next(t for t in registry.bundled_tasks() if t.name == "classification")


def _summarisation_spec():  # type: ignore[no-untyped-def]
    return next(t for t in registry.bundled_tasks() if t.name == "summarisation")


def test_score_run_and_summarise_are_deterministic(tmp_path: Path) -> None:
    spec = _classification_spec()
    items = registry.load_items(spec, "calib")[:5]
    rd = RunDir(tmp_path / "run")
    records = []
    for i, item in enumerate(items):
        wrong = next(lbl for lbl in (spec.labels or []) if lbl != item["label"])
        raw = item["label"] if i < 3 else wrong
        records.append(
            {
                "model_id": "m1",
                "task": "classification",
                "split": "calib",
                "item_id": item["id"],
                "pass": "A",
                "raw": raw,
            }
        )
    rd.append_records("predictions.jsonl.gz", records)

    specs = {"classification": spec}
    scores = evaluate.score_run(rd, specs, "calib", pass_="A")
    assert set(scores.keys()) == {("m1", "classification")}
    item_scores = scores[("m1", "classification")]
    assert len(item_scores) == 5

    summary1 = evaluate.summarise(spec, item_scores)
    summary2 = evaluate.summarise(spec, item_scores)
    assert summary1 == summary2
    assert summary1["n"] == 5
    assert summary1["metrics"]["accuracy"] == pytest.approx(3 / 5)
    assert summary1["metrics"]["invalid_output_rate"] == pytest.approx(0.0)
    assert set(summary1["ci"]) == set(summary1["metrics"])
    for lo, hi in summary1["ci"].values():
        assert lo <= hi or (lo != lo and hi != hi)  # allow (nan, nan)


def test_score_run_filters_by_split_and_pass(tmp_path: Path) -> None:
    spec = _classification_spec()
    calib_items = registry.load_items(spec, "calib")[:2]
    test_item = registry.load_items(spec, "test")[0]
    rd = RunDir(tmp_path / "run")
    records = [
        {
            "model_id": "m1",
            "task": "classification",
            "split": "calib",
            "item_id": calib_items[0]["id"],
            "pass": "A",
            "raw": calib_items[0]["label"],
        },
        {
            "model_id": "m1",
            "task": "classification",
            "split": "test",
            "item_id": test_item["id"],
            "pass": "A",
            "raw": test_item["label"],
        },
        {
            "model_id": "m1",
            "task": "classification",
            "split": "calib",
            "item_id": calib_items[1]["id"],
            "pass": "B",
            "raw": calib_items[1]["label"],
        },
    ]
    rd.append_records("predictions.jsonl.gz", records)

    specs = {"classification": spec}
    calib_a = evaluate.score_run(rd, specs, "calib", pass_="A")
    assert len(calib_a[("m1", "classification")]) == 1
    test_a = evaluate.score_run(rd, specs, "test", pass_="A")
    assert len(test_a[("m1", "classification")]) == 1
    calib_b = evaluate.score_run(rd, specs, "calib", pass_="B")
    assert len(calib_b[("m1", "classification")]) == 1


def test_summarisation_uses_judge_verdicts_when_present(tmp_path: Path) -> None:
    spec = _summarisation_spec()
    items = registry.load_items(spec, "calib")[:1]
    item = items[0]
    rd = RunDir(tmp_path / "run")
    rd.append_records(
        "predictions.jsonl.gz",
        [
            {
                "model_id": "m1",
                "task": "summarisation",
                "split": "calib",
                "item_id": item["id"],
                "pass": "A",
                "raw": "A short summary.",
            }
        ],
    )
    n_facts = len(item.get("required_facts", []))
    verdict = {
        "facts": [{"index": i, "present": True, "correct": True} for i in range(n_facts)],
        "unsupported_claims": [],
    }
    rd.append_records(
        "judge_scores.jsonl.gz",
        [{"model_id": "m1", "split": "calib", "item_id": item["id"], "verdict": verdict, "passed": True}],
    )

    specs = {"summarisation": spec}
    scores = evaluate.score_run(rd, specs, "calib", pass_="A")
    item_scores = scores[("m1", "summarisation")]
    assert item_scores[0].stats["judged"] is True
    summary = evaluate.summarise(spec, item_scores)
    assert summary["metrics"]["pass_rate"] == pytest.approx(1.0)


def test_task_specs_for_run_includes_bundled_and_custom(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    custom_dir = tmp_path / "my-task"
    custom_dir.mkdir()
    (custom_dir / "task.yaml").write_text(
        "name: my_custom\nkind: classification\ncalib: calib.jsonl\ntest: test.jsonl\nlabels: [a, b]\n"
    )
    (custom_dir / "calib.jsonl").write_text('{"id": "1", "text": "x", "label": "a"}\n')
    (custom_dir / "test.jsonl").write_text('{"id": "1", "text": "x", "label": "a"}\n')

    rd = RunDir(tmp_path / "run")
    rd.write_json(
        "tasks.json",
        {
            "my_custom": {
                "kind": "classification",
                "source": "custom",
                "task_yaml_path": "my-task/task.yaml",
                "sha256": {},
            }
        },
    )

    specs = evaluate.task_specs_for_run(rd)
    assert "classification" in specs  # bundled
    assert "my_custom" in specs
    assert specs["my_custom"].labels == ["a", "b"]


def test_task_specs_for_run_extra_tasks_wins(tmp_path: Path) -> None:
    rd = RunDir(tmp_path / "run")
    override = _classification_spec()
    specs = evaluate.task_specs_for_run(rd, extra_tasks={"classification": override})
    assert specs["classification"] is override


def test_latency_summary_passthrough(tmp_path: Path) -> None:
    rd = RunDir(tmp_path / "run")
    rd.write_json("latency.json", {"m1": {"classification": {"calib": {"A": {"n": 5}}}}})
    assert evaluate.latency_summary(rd) == {"m1": {"classification": {"calib": {"A": {"n": 5}}}}}
    assert evaluate.latency_summary(tmp_path / "nonexistent") == {}
