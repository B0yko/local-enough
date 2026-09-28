from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

from local_enough.bench.prices import load_snapshot, snapshot_prices

MODELS_RESPONSE = {
    "data": [
        {
            "id": "deepseek/deepseek-chat-v3.1",
            "pricing": {"prompt": "0.00000027", "completion": "0.0000011", "request": "0"},
            "supported_parameters": ["temperature", "max_tokens", "reasoning"],
        },
        {
            "id": "openai/gpt-4.1-mini",
            "pricing": {"prompt": "0.0000004", "completion": "0.0000016", "input_cache_read": "0.0000001"},
            "supported_parameters": ["temperature", "max_tokens"],
        },
    ]
}


async def test_snapshot_prices_builds_expected_shape() -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://openrouter.ai/api/v1/models").mock(return_value=httpx.Response(200, json=MODELS_RESPONSE))
        result = await snapshot_prices(
            "https://openrouter.ai/api/v1",
            {"open-large": "deepseek/deepseek-chat-v3.1", "judge": "openai/gpt-4.1-mini"},
        )

    assert result["source_url"] == "https://openrouter.ai/api/v1/models"
    assert "taken_at_utc" in result
    open_large = result["models"]["open-large"]
    assert open_large["provider_model"] == "deepseek/deepseek-chat-v3.1"
    assert open_large["prompt"] == 0.00000027
    assert open_large["completion"] == 0.0000011
    assert open_large["input_cache_read"] is None
    assert "reasoning" in open_large["supported_parameters"]

    judge = result["models"]["judge"]
    assert judge["input_cache_read"] == 0.0000001


async def test_snapshot_prices_missing_model_gets_null_fields() -> None:
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://openrouter.ai/api/v1/models").mock(return_value=httpx.Response(200, json=MODELS_RESPONSE))
        result = await snapshot_prices("https://openrouter.ai/api/v1", {"missing": "vendor/does-not-exist"})

    entry = result["models"]["missing"]
    assert entry["provider_model"] == "vendor/does-not-exist"
    assert entry["prompt"] is None
    assert entry["completion"] is None
    assert entry["supported_parameters"] == []


def test_load_snapshot_reads_written_file(tmp_path: Path) -> None:
    payload = {"taken_at_utc": "2026-09-28T10:00:00Z", "source_url": "https://x/models", "models": {}}
    path = tmp_path / "price_snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_snapshot(path) == payload
