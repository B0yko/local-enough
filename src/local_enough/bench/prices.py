"""A dated price snapshot from ``GET /models``, taken once at bench time (no key required).

Offline commands (``report``, ``route --simulate``) never refetch prices; they read the snapshot
written into the run directory by the live bench run instead (:func:`load_snapshot`).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


async def snapshot_prices(base_url: str, model_names: dict[str, str]) -> dict[str, Any]:
    """Fetch ``GET {base_url}/models`` and build the ``price_snapshot.json`` shape.

    ``model_names`` maps our config model id to the provider's model slug (for example
    ``{"open-large": "deepseek/deepseek-chat-v3.1"}``, judge models included). A slug missing from
    the response gets ``None`` prices and an empty ``supported_parameters`` list rather than
    raising, since a stale or renamed slug should not abort the whole bench run.
    """
    source_url = f"{base_url.rstrip('/')}/models"
    async with httpx.AsyncClient() as client:
        response = await client.get(source_url, timeout=30.0)
        response.raise_for_status()
        payload = response.json()

    by_id: dict[str, dict[str, Any]] = {}
    for entry in payload.get("data", []):
        entry_id = entry.get("id")
        if isinstance(entry_id, str):
            by_id[entry_id] = entry

    models: dict[str, Any] = {}
    for model_id, provider_model in model_names.items():
        entry = by_id.get(provider_model)
        pricing = (entry or {}).get("pricing") or {}
        models[model_id] = {
            "provider_model": provider_model,
            "prompt": _price(pricing.get("prompt")),
            "completion": _price(pricing.get("completion")),
            "input_cache_read": _price(pricing.get("input_cache_read")),
            "request": _price(pricing.get("request")),
            "supported_parameters": list((entry or {}).get("supported_parameters") or []),
        }

    return {"taken_at_utc": _now_utc(), "source_url": source_url, "models": models}


def _price(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_snapshot(path: str | Path) -> dict[str, Any]:
    """Load a previously written ``price_snapshot.json`` (offline; no network)."""
    with Path(path).open(encoding="utf-8") as fh:
        result: dict[str, Any] = json.load(fh)
        return result
