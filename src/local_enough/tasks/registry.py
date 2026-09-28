"""Task-kind lookup, task.yaml loading, item loading and stratified sampling."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jinja2
import yaml

from local_enough import paths
from local_enough.config import TASK_KINDS, TaskKind
from local_enough.tasks.base import Item, Message, TaskKindModule, TaskSpec

_TEMPLATE_MARKER = "===USER==="


def get_kind(kind: TaskKind) -> TaskKindModule:
    """Import and return the module implementing this task kind."""
    from local_enough.tasks import (
        classification,
        entity_matching,
        extraction,
        pii_redaction,
        summarisation,
    )

    modules: dict[str, TaskKindModule] = {
        "classification": classification.MODULE,
        "extraction": extraction.MODULE,
        "pii_redaction": pii_redaction.MODULE,
        "summarisation": summarisation.MODULE,
        "entity_matching": entity_matching.MODULE,
    }
    if kind not in modules:
        raise KeyError(f"unknown task kind {kind!r}; expected one of {TASK_KINDS}")
    return modules[kind]


def load_task(path_to_task_yaml: str | Path) -> TaskSpec:
    """Load and validate a ``task.yaml``; ``.root`` is set to its containing directory."""
    yaml_path = Path(path_to_task_yaml)
    if yaml_path.is_dir():
        yaml_path = yaml_path / "task.yaml"
    with yaml_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    spec = TaskSpec.model_validate(data)
    spec.root = yaml_path.resolve().parent
    return spec


def bundled_tasks() -> list[TaskSpec]:
    """The five tasks shipped in ``data/datasets/<name>/task.yaml``, one per kind."""
    root = paths.datasets_dir()
    specs = []
    for child in sorted(root.iterdir()) if root.exists() else []:
        task_yaml = child / "task.yaml"
        if task_yaml.exists():
            specs.append(load_task(task_yaml))
    return specs


def resolve_tasks(selector: str) -> list[TaskSpec]:
    """Parse ``"all,path/to/task.yaml"`` into a list of task specs."""
    specs: list[TaskSpec] = []
    for part in (p.strip() for p in selector.split(",")):
        if not part:
            continue
        if part == "all":
            specs.extend(bundled_tasks())
        else:
            specs.append(load_task(part))
    return specs


def load_items(spec: TaskSpec, split: str) -> list[Item]:
    """Read one JSONL split of a task's dataset."""
    path = spec.split_path(split)
    items: list[Item] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def stratified_sample(items: list[Item], n: int, seed: int, key: Callable[[Item], Any] | None = None) -> list[Item]:
    """Deterministic sample of ``n`` items, stratified by ``key`` when given; keeps original order."""
    if n <= 0:
        return []
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    if key is None:
        chosen = set(rng.sample(range(len(items)), n))
        return [items[i] for i in sorted(chosen)]

    groups: dict[Any, list[int]] = {}
    for i, item in enumerate(items):
        groups.setdefault(key(item), []).append(i)

    total = len(items)
    alloc = {k: min(len(idxs), round(n * len(idxs) / total)) for k, idxs in groups.items()}
    diff = n - sum(alloc.values())
    ordered_keys = sorted(groups, key=lambda k: -len(groups[k]))
    i = 0
    guard = 0
    while diff != 0 and ordered_keys and guard < 10 * n + 100:
        k = ordered_keys[i % len(ordered_keys)]
        if diff > 0 and alloc[k] < len(groups[k]):
            alloc[k] += 1
            diff -= 1
        elif diff < 0 and alloc[k] > 0:
            alloc[k] -= 1
            diff += 1
        i += 1
        guard += 1

    chosen_idx: set[int] = set()
    for k, idxs in groups.items():
        chosen_idx.update(rng.sample(idxs, min(alloc[k], len(idxs))))
    return [items[i] for i in sorted(chosen_idx)]


def template_files() -> dict[str, Path]:
    """Path to each kind's Jinja prompt template, for sha256 recording in ``env.json``."""
    base = Path(__file__).resolve().parent / "templates"
    return {kind: base / f"{kind}.j2" for kind in TASK_KINDS}


def load_template_parts(kind: TaskKind) -> tuple[str, str]:
    """(system_template, user_template) split on the ``===USER===`` marker."""
    raw = template_files()[kind].read_text(encoding="utf-8")
    system_part, marker, user_part = raw.partition(_TEMPLATE_MARKER)
    if not marker:
        return "", raw.strip()
    return system_part.strip(), user_part.strip()


def render_prompt(kind: TaskKind, **context: Any) -> list[Message]:
    """Render the kind's template into system+user chat messages."""
    system_tmpl, user_tmpl = load_template_parts(kind)
    env = jinja2.Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda value, indent=None: json.dumps(
        value, indent=indent, sort_keys=True, ensure_ascii=False
    )
    messages: list[Message] = []
    if system_tmpl:
        messages.append({"role": "system", "content": env.from_string(system_tmpl).render(**context)})
    messages.append({"role": "user", "content": env.from_string(user_tmpl).render(**context)})
    return messages
