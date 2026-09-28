from __future__ import annotations

import platform
from typing import Any

import pytest

from local_enough.bench.envinfo import (
    ALLOWED_HARDWARE_PRICE_KEYS,
    ALLOWED_POWER_KEYS,
    ALLOWED_TOP_LEVEL_KEYS,
    collect_env,
)

HARDWARE_PRICE = {
    "usd": 1299.0,
    "label": "list price",
    "source_url": "https://www.apple.com/shop/buy-mac/macbook-air",
    "date": "2026-09-28",
    "unexpected_field": "should be dropped",
}


def _collect() -> dict[str, Any]:
    return collect_env(
        models={"local-qwen": "mlx-community/Qwen3-4B-Instruct-2507-4bit@50d4277"},
        dataset_sha256={"extraction.test": "a" * 64},
        template_sha256={"extraction": "b" * 64},
        hardware_price=HARDWARE_PRICE,
        start_utc="2026-09-28T10:00:00Z",
        end_utc="2026-09-28T11:00:00Z",
    )


def test_collect_env_top_level_key_set_matches_allowlist() -> None:
    result = _collect()
    assert set(result.keys()) == ALLOWED_TOP_LEVEL_KEYS


def test_collect_env_power_keys_match_allowlist() -> None:
    result = _collect()
    assert set(result["power"].keys()) == ALLOWED_POWER_KEYS


def test_collect_env_hardware_price_drops_unexpected_fields() -> None:
    result = _collect()
    assert set(result["hardware_price"].keys()) <= ALLOWED_HARDWARE_PRICE_KEYS
    assert "unexpected_field" not in result["hardware_price"]
    assert result["hardware_price"]["usd"] == 1299.0


def test_collect_env_carries_through_caller_supplied_fields() -> None:
    result = _collect()
    assert result["models"] == {"local-qwen": "mlx-community/Qwen3-4B-Instruct-2507-4bit@50d4277"}
    assert result["datasets_sha256"] == {"extraction.test": "a" * 64}
    assert result["templates_sha256"] == {"extraction": "b" * 64}
    assert result["start_utc"] == "2026-09-28T10:00:00Z"
    assert result["end_utc"] == "2026-09-28T11:00:00Z"
    assert result["local_enough_version"] == "0.1.0"


def test_collect_env_no_value_looks_like_a_path_serial_or_hostname() -> None:
    result = _collect()
    hostname = platform.node()

    filesystem_path_markers = ("/Users/", "/home/", "\\Users\\")

    def check(value: object, path: str) -> None:
        if isinstance(value, str):
            assert value.startswith(("http://", "https://")) or not value.startswith("/"), (
                f"{path} looks like an absolute filesystem path: {value!r}"
            )
            assert not any(marker in value for marker in filesystem_path_markers), (
                f"{path} looks like a filesystem path: {value!r}"
            )
            if hostname:
                assert value != hostname, f"{path} matches this machine's hostname"
        elif isinstance(value, dict):
            for k, v in value.items():
                check(v, f"{path}.{k}")

    for key, value in result.items():
        check(value, key)


def test_collect_env_linux_darwin_only_fields_are_null(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_enough.bench.envinfo.sys.platform", "linux")
    result = _collect()
    assert result["machine_model"] is None
    assert result["chip"] is None
    assert result["memory_gb"] is None
    assert result["macos_version"] is None
    assert result["macos_build"] is None
    assert result["power"] == {"ac": None, "charge_pct": None, "charging": None, "low_power_mode": None}
