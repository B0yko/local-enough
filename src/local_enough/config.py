"""Typed configuration for ``config.yaml`` (models, judge, hardware) and ``route.yaml`` (routing policy)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TaskKind = Literal["extraction", "classification", "pii_redaction", "summarisation", "entity_matching"]
TASK_KINDS: tuple[TaskKind, ...] = (
    "extraction",
    "classification",
    "pii_redaction",
    "summarisation",
    "entity_matching",
)
BaselineTask = Literal["classification", "pii_redaction", "entity_matching"]
CloudRole = Literal["frontier", "small-closed", "open-large", "open-same-family"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MlxLaunch(_Strict):
    """How the CLI starts ``mlx_lm.server`` for a local model."""

    runtime: Literal["mlx"] = "mlx"
    repo: str = Field(description="Hugging Face repo id, e.g. mlx-community/Qwen3-4B-Instruct-2507-4bit.")
    revision: str = Field(default="main", description="Commit sha (recommended) or branch of the repo.")
    port: int = Field(default=8081, ge=1024, le=65535, description="Port for the server on 127.0.0.1.")
    host: str = Field(default="127.0.0.1", description="Bind address. Anything but 127.0.0.1 exposes the server.")
    startup_timeout_s: float = Field(default=300.0, gt=0, description="Seconds to wait for server readiness.")


class LocalModel(_Strict):
    """A model on hardware you own: either launched by local-enough (``launch``) or already running (``base_url``)."""

    id: str
    kind: Literal["local"] = "local"
    launch: MlxLaunch | None = Field(default=None, description="Start and stop mlx_lm.server for this model.")
    base_url: str | None = Field(default=None, description="OpenAI-compatible endpoint you already run.")
    model: str | None = Field(default=None, description="Model name sent to base_url (ignored with launch).")
    api_key_env: str | None = Field(default=None, description="Env var holding a key for base_url, if needed.")
    concurrency: int = Field(default=1, ge=1, description="Default concurrent requests (Pass A uses 1).")
    timeout_s: float | None = Field(default=None, gt=0, description="Per-call timeout; route.yaml sets the default.")

    @model_validator(mode="after")
    def _one_endpoint(self) -> LocalModel:
        if (self.launch is None) == (self.base_url is None):
            raise ValueError(f"local model {self.id!r}: set exactly one of 'launch' or 'base_url'")
        return self


class CloudModel(_Strict):
    """A hosted model behind an OpenAI-compatible endpoint (OpenRouter by default)."""

    id: str
    kind: Literal["cloud"] = "cloud"
    role: CloudRole | None = Field(default=None, description="Lineup role used in the report.")
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    model: str = Field(description="Model id at the provider, e.g. deepseek/deepseek-chat-v3.1.")
    concurrency: int = Field(default=8, ge=1, description="Bounded concurrency per model.")
    reasoning: dict[str, Any] | None = Field(
        default=None,
        description="OpenRouter 'reasoning' request object, e.g. {effort: low} or {enabled: false}.",
    )
    reasoning_allowance_tokens: int = Field(
        default=0, ge=0, description="Added to max_tokens when reasoning cannot be switched off."
    )
    provider: dict[str, Any] | None = Field(
        default=None,
        description="OpenRouter provider routing object, e.g. {order: [x], allow_fallbacks: false}.",
    )
    extra_body: dict[str, Any] = Field(default_factory=dict, description="Extra request fields sent verbatim.")
    timeout_s: float | None = Field(default=None, gt=0, description="Per-call timeout; route.yaml sets the default.")


class BaselineModel(_Strict):
    """A non-LLM baseline served in-process at near-zero cost."""

    id: str
    kind: Literal["baseline"] = "baseline"
    task: BaselineTask = Field(description="Task kind the baseline solves.")


ModelConfig = Annotated[LocalModel | CloudModel | BaselineModel, Field(discriminator="kind")]


class JudgeConfig(_Strict):
    candidates: list[str] = Field(description="Judge model ids at base_url; calibration keeps the best one.")
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    concurrency: int = Field(default=8, ge=1)
    max_tokens: int = Field(default=600, ge=16, description="Visible output cap for a judge verdict.")
    reasoning: dict[str, Any] | None = None
    reasoning_allowance_tokens: int = Field(default=0, ge=0)
    provider: dict[str, Any] | None = None


class PowerConfig(_Strict):
    mode: Literal["measured", "configured"] = Field(
        default="measured", description="measured: use power-probe results; configured: use the watts below."
    )
    incremental_watts: float = Field(default=20.0, ge=0, description="Extra watts while serving (configured mode).")
    idle_watts: float = Field(default=5.0, ge=0, description="Watts with the model loaded and idle (configured).")


class HardwareConfig(_Strict):
    name: str = Field(description="Label for the hardware profile.")
    purchase_price_usd: float = Field(description="Purchase price of one machine in USD. 0 is rejected.")
    price_source_url: str | None = Field(default=None, description="Where the price comes from.")
    price_date: str | None = Field(default=None, description="Date the price was read (YYYY-MM-DD).")
    price_label: Literal["list price", "configured"] = Field(
        default="configured", description="'list price' only when verified at price_source_url on price_date."
    )
    lifetime_years: float = Field(default=3.0, gt=0, description="Amortisation period.")
    allocation: float = Field(default=1.0, gt=0, le=1.0, description="Share of the machine charged to this work.")
    busy_hours_per_day: float = Field(default=8.0, gt=0, le=24, description="Hours per day the machine serves.")
    electricity_usd_per_kwh: float = Field(default=0.30, ge=0, description="Electricity price (assumption).")
    ops_usd_per_month: float = Field(default=0.0, ge=0, description="Operations cost per machine per month.")
    power: PowerConfig = Field(default_factory=PowerConfig)

    @field_validator("purchase_price_usd")
    @classmethod
    def _price_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(
                "hardware.purchase_price_usd must be > 0: a missing price would make local look free. "
                "Enter the list price of your machine."
            )
        return v


class Config(_Strict):
    """Top-level ``config.yaml``."""

    project: str = Field(description="Ledger name: $LOCAL_ENOUGH_HOME/ledgers/<project>.jsonl.")
    budget_usd: float = Field(default=5.0, gt=0, description="Hard spend cap (lower of this and env var wins).")
    budget_warn_usd: float | None = Field(default=None, ge=0, description="Print a warning past this spend.")
    local_models_max_gb: float = Field(default=4.0, gt=0, description="Download cap for configured local models.")
    max_tokens: dict[str, int] = Field(default_factory=dict, description="Per-kind override of visible output caps.")
    models: list[ModelConfig]
    judge: JudgeConfig | None = None
    hardware: HardwareConfig | None = Field(default=None, description="Required for local cost and break-even.")

    @model_validator(mode="after")
    def _unique_ids(self) -> Config:
        ids = [m.id for m in self.models]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate model ids: {dupes}")
        return self

    def model(self, model_id: str) -> LocalModel | CloudModel | BaselineModel:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(model_id)


class QualityBar(_Strict):
    relative_to_best: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description="Fraction of the best candidate's calib score (baselines included).",
    )
    absolute: float | None = Field(default=None, ge=0, le=1, description="Absolute primary-metric threshold.")

    @model_validator(mode="after")
    def _exactly_one(self) -> QualityBar:
        if (self.relative_to_best is None) == (self.absolute is None):
            raise ValueError("quality bar: set exactly one of relative_to_best or absolute")
        return self


class Constraints(_Strict):
    data_must_stay_local: list[str] = Field(default_factory=list, description="Tasks never sent to cloud models.")


class Timeouts(_Strict):
    local: float = Field(default=60.0, gt=0)
    cloud: float = Field(default=60.0, gt=0)


_NO_CAP: float | None = None


class RouteConfig(_Strict):
    """Top-level ``route.yaml``. Task-keyed maps use task names (bundled names or a custom task's name)."""

    quality_bar: dict[str, QualityBar] = Field(
        default_factory=lambda: {"default": QualityBar(relative_to_best=0.95)},
        description="'default' plus per-task overrides.",
    )
    latency_p95_max_s: dict[str, float | None] = Field(
        default_factory=lambda: {"default": _NO_CAP}, description="Optional p95 latency ceiling per task."
    )
    constraints: Constraints = Field(default_factory=Constraints)
    allow_below_bar_local: bool = Field(
        default=False, description="Serve an unservable local-only task with its best local candidate anyway."
    )
    workload_mix: dict[str, float] = Field(description="Share of monthly volume per task; sums to 1.")
    reference_monthly_volume: float = Field(gt=0, description="Total mixed tasks per month.")
    expected_monthly_volume: dict[str, float] = Field(
        default_factory=dict, description="Optional per-task monthly volume overriding volume x mix."
    )
    timeouts_s: Timeouts = Field(default_factory=Timeouts)

    @field_validator("workload_mix")
    @classmethod
    def _mix_sums_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if v and abs(total - 1.0) > 1e-6:
            raise ValueError(f"workload_mix must sum to 1.0, got {total:.6f}")
        if any(x < 0 for x in v.values()):
            raise ValueError("workload_mix weights must be >= 0")
        return v

    def bar_for(self, task: str) -> QualityBar:
        return self.quality_bar.get(task) or self.quality_bar.get("default") or QualityBar(relative_to_best=0.95)

    def latency_cap_for(self, task: str) -> float | None:
        if task in self.latency_p95_max_s:
            return self.latency_p95_max_s[task]
        return self.latency_p95_max_s.get("default")

    def monthly_volume(self, task: str) -> float:
        if task in self.expected_monthly_volume:
            return self.expected_monthly_volume[task]
        return self.reference_monthly_volume * self.workload_mix.get(task, 0.0)

    def is_local_only(self, task: str) -> bool:
        return task in self.constraints.data_must_stay_local


def _read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | Path) -> Config:
    return Config.model_validate(_read_yaml(Path(path)))


def load_route(path: str | Path) -> RouteConfig:
    return RouteConfig.model_validate(_read_yaml(Path(path)))
