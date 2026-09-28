"""registry: get_kind, load_task, bundled_tasks, resolve_tasks, load_items, stratified_sample, templates."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from local_enough.tasks import registry
from local_enough.tasks.base import TaskSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_TASK = REPO_ROOT / "examples" / "custom-task" / "task.yaml"
ALL_KINDS = ["extraction", "classification", "pii_redaction", "summarisation", "entity_matching"]


def test_get_kind_returns_a_module_per_kind() -> None:
    for k in ALL_KINDS:
        module = registry.get_kind(k)  # type: ignore[arg-type]
        assert module.kind == k
        assert module.primary_metric


def test_get_kind_rejects_unknown_kind() -> None:
    with pytest.raises(KeyError):
        registry.get_kind("not_a_kind")  # type: ignore[arg-type]


def test_bundled_tasks_has_the_five_kinds() -> None:
    specs = registry.bundled_tasks()
    kinds = {s.kind for s in specs}
    assert kinds == set(ALL_KINDS)
    for s in specs:
        assert s.root is not None and s.root.exists()


def test_load_task_sets_root_and_accepts_a_directory() -> None:
    spec = registry.load_task(EXAMPLE_TASK.parent)
    assert spec.root == EXAMPLE_TASK.parent.resolve()
    spec2 = registry.load_task(EXAMPLE_TASK)
    assert spec2.root == EXAMPLE_TASK.parent.resolve()


def test_resolve_tasks_all_plus_custom_path() -> None:
    specs = registry.resolve_tasks(f"all,{EXAMPLE_TASK}")
    names = [s.name for s in specs]
    assert names[:5] != []  # the five bundled tasks come first
    assert "custom_support_tickets" in names
    assert len(specs) == 6


def test_resolve_tasks_custom_only() -> None:
    specs = registry.resolve_tasks(str(EXAMPLE_TASK))
    assert len(specs) == 1
    assert specs[0].name == "custom_support_tickets"


def test_load_items_reads_jsonl() -> None:
    spec = registry.load_task(EXAMPLE_TASK)
    items = registry.load_items(spec, "calib")
    assert len(items) == 30
    assert all("id" in it and "text" in it and "label" in it for it in items)


def test_load_items_missing_split_raises() -> None:
    spec = registry.load_task(EXAMPLE_TASK)
    with pytest.raises(FileNotFoundError):
        registry.load_items(spec, "does_not_exist")


def test_stratified_sample_is_deterministic() -> None:
    items = [{"id": str(i), "v": i % 4} for i in range(40)]
    a = registry.stratified_sample(items, 8, seed=7, key=lambda it: it["v"])
    b = registry.stratified_sample(items, 8, seed=7, key=lambda it: it["v"])
    assert a == b


def test_stratified_sample_covers_groups_proportionally() -> None:
    items = [{"id": str(i), "v": "x" if i < 80 else "y"} for i in range(100)]  # 80/20 split
    sample = registry.stratified_sample(items, 20, seed=1, key=lambda it: it["v"])
    counts = Counter(it["v"] for it in sample)
    assert counts["x"] > counts["y"]
    assert sum(counts.values()) == 20


def test_stratified_sample_keeps_original_order() -> None:
    items = [{"id": str(i)} for i in range(50)]
    sample = registry.stratified_sample(items, 10, seed=3)
    ids = [int(it["id"]) for it in sample]
    assert ids == sorted(ids)


def test_stratified_sample_returns_all_when_n_exceeds_len() -> None:
    items = [{"id": "1"}, {"id": "2"}]
    assert registry.stratified_sample(items, 10, seed=1) == items


def test_stratified_sample_zero_returns_empty() -> None:
    assert registry.stratified_sample([{"id": "1"}], 0, seed=1) == []


def test_template_files_lists_all_five_kinds() -> None:
    files = registry.template_files()
    assert set(files) == set(ALL_KINDS)
    for path in files.values():
        assert path.exists()


def test_render_prompt_produces_system_and_user_messages() -> None:
    messages = registry.render_prompt("classification", labels=["a", "b"], text="hello")
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]
    assert "hello" in messages[1]["content"]


def test_render_prompt_entity_matching_embeds_json_records() -> None:
    messages = registry.render_prompt("entity_matching", left={"name": "A"}, right={"name": "B"})
    content = messages[1]["content"]
    assert '"name": "A"' in content
    assert '"name": "B"' in content


def test_stratified_sample_group_shrinkage_never_exceeds_group_size() -> None:
    rng = random.Random(0)
    items = [{"id": str(i), "v": rng.choice(["a", "b", "c"])} for i in range(30)]
    sample = registry.stratified_sample(items, 25, seed=5, key=lambda it: it["v"])
    assert len(sample) == 25


def test_spec_split_path_resolves_relative_to_root() -> None:
    spec: TaskSpec = registry.load_task(EXAMPLE_TASK)
    assert spec.split_path("calib") == spec.root / "calib.jsonl"
