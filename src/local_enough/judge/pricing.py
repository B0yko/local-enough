"""Cost estimation and settlement for judge calls, sharing the bench ledger's reservation rule.

Judge candidates are never entries in ``config.yaml``'s ``models:`` list (a judge is not a benchmarked
model), so they get their own :class:`Endpoint` built straight from ``judge:`` config plus a candidate slug.
"""

from __future__ import annotations

import os
from typing import Any

from local_enough.bench.ledger import estimate_tokens
from local_enough.config import JudgeConfig
from local_enough.providers.openai_compat import ChatResult, Endpoint, Message, MissingApiKey


def judge_endpoint(judge_cfg: JudgeConfig, candidate: str) -> Endpoint:
    """Build the :class:`Endpoint` for one judge candidate model."""
    api_key = os.environ.get(judge_cfg.api_key_env)
    if not api_key:
        raise MissingApiKey(
            f"environment variable {judge_cfg.api_key_env!r} is not set (needed for judge model {candidate!r})"
        )
    return Endpoint(
        model_id=candidate,
        base_url=judge_cfg.base_url,
        model_name=candidate,
        api_key=api_key,
        kind="cloud",
        display_model=candidate,
        reasoning=judge_cfg.reasoning,
        provider_routing=judge_cfg.provider,
        reasoning_allowance_tokens=judge_cfg.reasoning_allowance_tokens,
    )


def reserve_estimate(
    messages: list[Message], max_tokens: int, reasoning_allowance: int, price: dict[str, Any]
) -> float:
    """Upper-bound reservation: ``(prompt_estimate * in_price + (max_tokens + allowance) * out_price) * 1.2``."""
    input_price = price.get("prompt") or 0.0
    output_price = price.get("completion") or 0.0
    prompt_estimate = estimate_tokens(messages)
    return (prompt_estimate * input_price + (max_tokens + reasoning_allowance) * output_price) * 1.2


def settle_cost(result: ChatResult, price: dict[str, Any]) -> tuple[float, str]:
    """Actual cost: ``usage.cost`` when the provider reports it, else tokens priced from the snapshot."""
    if result.cost_usd is not None:
        return result.cost_usd, "usage.cost"
    input_price = price.get("prompt") or 0.0
    output_price = price.get("completion") or 0.0
    cost = result.prompt_tokens * input_price + (result.completion_tokens + result.reasoning_tokens) * output_price
    return cost, "snapshot"
