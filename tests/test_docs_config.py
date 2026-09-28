"""docs/configuration.md must document every field of every config and task.yaml model."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from local_enough.config import (
    BaselineModel,
    CloudModel,
    Config,
    Constraints,
    HardwareConfig,
    JudgeConfig,
    LocalModel,
    MlxLaunch,
    PowerConfig,
    QualityBar,
    RouteConfig,
    Timeouts,
)
from local_enough.tasks.base import DatasetCard, FieldSpec, TaskSpec

DOC = Path(__file__).resolve().parents[1] / "docs" / "configuration.md"

MODELS: list[type[BaseModel]] = [
    Config,
    LocalModel,
    MlxLaunch,
    CloudModel,
    BaselineModel,
    JudgeConfig,
    HardwareConfig,
    PowerConfig,
    RouteConfig,
    QualityBar,
    Constraints,
    Timeouts,
    TaskSpec,
    FieldSpec,
    DatasetCard,
]


def test_doc_exists() -> None:
    assert DOC.is_file(), "docs/configuration.md is missing"


def test_every_public_field_is_documented() -> None:
    text = DOC.read_text(encoding="utf-8")
    missing: list[str] = []
    for model in MODELS:
        for name, info in model.model_fields.items():
            if info.exclude:
                # Internal bookkeeping (e.g. TaskSpec.root), never written into or read from the YAML file.
                continue
            if f"`{name}`" not in text:
                missing.append(f"{model.__name__}.{name}")
    assert not missing, f"undocumented fields in docs/configuration.md: {missing}"


def test_env_vars_are_documented() -> None:
    text = DOC.read_text(encoding="utf-8")
    for var in ("OPENROUTER_API_KEY", "LOCAL_ENOUGH_BUDGET_USD", "LOCAL_ENOUGH_HOME"):
        assert var in text, f"{var} is not documented in docs/configuration.md"


def test_volume_semantics_point_to_cost_model_doc() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "V_t" in text
    assert "cost-model.md" in text
