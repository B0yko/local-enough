"""Contracts shared by every task kind: task specs, items, parsed outputs and the per-kind module protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from local_enough.config import TaskKind

Item = dict[str, Any]
"""One JSONL record. Always has ``id``; the other fields depend on the kind (see docs/add-your-task.md)."""

Message = dict[str, str]
"""An OpenAI chat message: ``{"role": ..., "content": ...}``."""

DEFAULT_MAX_TOKENS: dict[TaskKind, int] = {
    "classification": 16,
    "entity_matching": 16,
    "extraction": 400,
    "pii_redaction": 500,
    "summarisation": 350,
}

FieldType = Literal["string", "email", "phone", "country", "date", "enum", "int", "amount", "currency"]


class FieldSpec(BaseModel):
    """One typed extraction field declared in ``task.yaml``."""

    model_config = ConfigDict(extra="forbid")

    type: FieldType
    values: list[str] | None = Field(default=None, description="Allowed values for enum/currency fields.")
    required: bool = Field(default=False, description="Gate requires a non-null value.")
    grounded: bool = Field(default=False, description="Free text that must occur in the source (gate).")
    nullable: bool = True


class DatasetCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    licence: str
    seed: int | None = None
    metric: str
    distinct_templates: int | None = None
    notes: str | None = None


class TaskSpec(BaseModel):
    """A task definition loaded from ``task.yaml``; bundled tasks use the same format."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Task name used in --tasks, route.yaml keys and the router alias.")
    kind: TaskKind
    description: str = ""
    calib: str = "calib.jsonl"
    test: str = "test.jsonl"
    train: str | None = Field(default=None, description="Optional train split (classification baseline only).")
    labels: list[str] | None = Field(default=None, description="Classification label set.")
    fields: dict[str, FieldSpec] | None = Field(default=None, description="Extraction schema, in output order.")
    pii_types: list[str] | None = Field(default=None, description="PII span types the policy covers.")
    max_words: int | None = Field(default=None, description="Summarisation default word limit.")
    max_tokens: int | None = Field(default=None, description="Override of the kind's visible output cap.")
    card: DatasetCard | None = None

    # Set by the loader, never serialised into run files (it is an absolute path).
    root: Path | None = Field(default=None, exclude=True)

    def split_path(self, split: str) -> Path:
        name = {"calib": self.calib, "test": self.test, "train": self.train}.get(split)
        if name is None or self.root is None:
            raise FileNotFoundError(f"task {self.name!r} has no {split!r} split")
        return self.root / name

    def output_cap(self) -> int:
        return self.max_tokens or DEFAULT_MAX_TOKENS[self.kind]


@dataclass(frozen=True)
class Parsed:
    """Result of the shared lenient parser plus kind-specific schema validation.

    ``value`` is the canonical parsed output (a label string, a dict or a list). ``content`` is the exact string the
    router returns as the message content. ``ok`` is False for unparseable or schema-invalid output, which scores as
    wrong and counts in ``invalid_output_rate``.
    """

    ok: bool
    value: Any = None
    content: str | None = None
    error: str | None = None


@dataclass
class ItemScore:
    """Per-item statistics; ``aggregate`` recomputes metrics from any resample of these (bootstrap)."""

    item_id: str
    valid: bool
    stats: dict[str, Any] = field(default_factory=dict)


class TaskKindModule(Protocol):
    """Implemented once per kind in ``local_enough.tasks.<kind>``."""

    kind: TaskKind
    primary_metric: str

    def render_messages(self, spec: TaskSpec, item: Item) -> list[Message]:
        """The benchmarked prompt: identical for every model, rendered from the kind's Jinja template."""
        ...

    def user_input(self, spec: TaskSpec, item: Item) -> str:
        """The raw input a router client sends as the user message for this item."""
        ...

    def item_from_input(self, spec: TaskSpec, raw: str) -> Item:
        """Rebuild an item (without gold) from a router client's user message."""
        ...

    def parse(self, spec: TaskSpec, raw_text: str) -> Parsed:
        """Lenient parse (strip fences and <think>, first JSON value / first line) plus schema validation."""
        ...

    def score(self, spec: TaskSpec, item: Item, parsed: Parsed, extra: dict[str, Any] | None = None) -> ItemScore:
        """Score one prediction against gold. ``extra`` carries judge verdicts for summarisation."""
        ...

    def aggregate(self, spec: TaskSpec, scores: list[ItemScore]) -> dict[str, float]:
        """Metrics over a list of item scores. Must include ``primary`` and ``invalid_output_rate``."""
        ...

    def validate_item(self, spec: TaskSpec, item: Item) -> list[str]:
        """Schema errors for a user's dataset record (empty list = valid)."""
        ...
