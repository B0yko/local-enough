"""Cost estimation and settlement for judge calls, sharing the bench ledger's reservation rule.

Judge candidates are never entries in ``config.yaml``'s ``models:`` list (a judge is not a benchmarked
model), so they get their own :class:`Endpoint` built straight from ``judge:`` config plus a candidate slug.
"""

from __future__ import annotations

import os
from typing import Any

from local_enough.bench.ledger import reservation_usd, settled_cost
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
    """Upper-bound reservation (see ``bench.ledger.reservation_usd``)."""
    return reservation_usd(messages, max_tokens, reasoning_allowance, price)


def settle_cost(result: ChatResult, price: dict[str, Any]) -> tuple[float, str]:
    """Actual cost (see ``bench.ledger.settled_cost``)."""
    return settled_cost(result, price)
