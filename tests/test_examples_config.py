"""examples/*.yaml load cleanly with the pydantic Config/RouteConfig models."""

from __future__ import annotations

from pathlib import Path

from local_enough.config import CloudModel, LocalModel, load_config, load_route

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_reference_config_loads_and_shapes_match() -> None:
    cfg = load_config(EXAMPLES / "config.yaml")
    assert cfg.project == "local-enough-reference"
    assert cfg.budget_usd == 14.5
    assert cfg.budget_warn_usd == 12
    ids = [m.id for m in cfg.models]
    assert ids == [
        "local-qwen3-4b",
        "local-qwen2.5-1.5b",
        "frontier",
        "small-closed",
        "open-large",
        "open-same-family",
        "tfidf-baseline",
        "regex-baseline",
        "rapidfuzz-baseline",
    ]
    cloud_models = {m.id: m.model for m in cfg.models if isinstance(m, CloudModel)}
    assert cloud_models == {
        "frontier": "x-ai/grok-4.7",
        "small-closed": "openai/gpt-5.4-nano",
        "open-large": "deepseek/deepseek-v3.2",
        "open-same-family": "qwen/qwen3-235b-a22b-2507",
    }
    assert cfg.judge is not None
    assert cfg.judge.candidates == ["openai/gpt-4.1-mini", "google/gemini-3.5-flash-lite"]
    assert cfg.hardware is not None
    assert cfg.hardware.name == "mac-studio-m4-max-128gb"
    assert cfg.hardware.purchase_price_usd == 4099
    assert cfg.hardware.price_label == "list price"
    assert cfg.hardware.power.mode == "configured"
    assert cfg.hardware.power.idle_watts == 6
    assert cfg.hardware.power.incremental_watts == 139


def test_quickstart_config_loads() -> None:
    cfg = load_config(EXAMPLES / "quickstart.yaml")
    assert cfg.budget_usd == 0.5
    assert len(cfg.models) == 4
    cloud = [m for m in cfg.models if isinstance(m, CloudModel)]
    assert len(cloud) == 1
    assert cloud[0].model == "qwen/qwen3-235b-a22b-2507"


def test_local_config_loads() -> None:
    cfg = load_config(EXAMPLES / "local.yaml")
    local_models = [m for m in cfg.models if isinstance(m, LocalModel)]
    assert len(local_models) == 1
    assert local_models[0].launch is not None
    assert local_models[0].launch.repo == "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    assert cfg.hardware is not None
    assert not any(isinstance(m, CloudModel) for m in cfg.models)


def test_route_config_loads_spec_shape() -> None:
    route_cfg = load_route(EXAMPLES / "route.yaml")
    assert route_cfg.workload_mix == {
        "classification": 0.30,
        "extraction": 0.25,
        "pii_redaction": 0.20,
        "entity_matching": 0.15,
        "summarisation": 0.10,
    }
    assert route_cfg.reference_monthly_volume == 50000
    assert route_cfg.constraints.data_must_stay_local == ["pii_redaction"]
    assert route_cfg.bar_for("pii_redaction").absolute == 0.95
    assert route_cfg.bar_for("classification").relative_to_best == 0.95
    assert route_cfg.latency_cap_for("classification") == 5
    assert route_cfg.latency_cap_for("extraction") is None
    assert route_cfg.timeouts_s.local == 60
    assert route_cfg.timeouts_s.cloud == 60
