"""Shared fake OpenAI-compatible server helpers for bench/evaluate tests.

Every fake speaks the same ``POST {base_url}/chat/completions`` shape the real ``ChatClient``
calls, registered through ``respx`` (which patches httpx at the transport level, so both a fake
cloud endpoint and a fake ``base_url`` local endpoint work without a real socket or process).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import respx

from local_enough.tasks.base import TaskSpec

Handler = Callable[[httpx.Request], httpx.Response]


def chat_response(
    content: str,
    *,
    prompt_tokens: int = 20,
    completion_tokens: int = 8,
    finish_reason: str = "stop",
    cost: float | None = None,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
    provider: str | None = None,
    status: int = 200,
    generation_id: str = "gen-fake",
) -> httpx.Response:
    """One canned OpenAI-compatible chat-completions response body."""
    usage: dict[str, Any] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    if cost is not None:
        usage["cost"] = cost
    if reasoning_tokens:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    if cached_tokens:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    body: dict[str, Any] = {
        "id": generation_id,
        # A real mlx_lm.server echoes an absolute cache path here; fakes do the same so the
        # model-path rewrite is exercised even against a fake local endpoint.
        "model": "/Users/x/.cache/huggingface/hub/models--fake/snapshots/deadbeef",
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": usage,
    }
    if provider:
        body["provider"] = provider
    return httpx.Response(status, json=body)


def sequence_handler(responses: list[httpx.Response]) -> Handler:
    """Return each response in order, repeating the last one once exhausted."""
    remaining: Iterator[httpx.Response] = iter(responses)

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal remaining
        try:
            return next(remaining)
        except StopIteration:
            return responses[-1]

    return _handler


def fixed_content_handler(content: str, **response_kwargs: Any) -> Handler:
    """Always answer with the same valid content, regardless of the request."""

    def _handler(_request: httpx.Request) -> httpx.Response:
        return chat_response(content, **response_kwargs)

    return _handler


def register_chat(mock: respx.MockRouter, base_url: str, handler: Handler) -> respx.Route:
    """Register a fake ``POST {base_url}/chat/completions`` route on an open ``respx.mock()``."""
    return mock.post(f"{base_url.rstrip('/')}/chat/completions").mock(side_effect=handler)


def default_valid_answer(spec: TaskSpec) -> str:
    """A canonically valid answer for ``spec.kind``, independent of any particular item.

    Used by end-to-end bench tests that care about harness mechanics (resume, ledger, model-path
    rewrite, retries) rather than about measuring accuracy against gold.
    """
    if spec.kind == "classification":
        labels = spec.labels or ["label"]
        return labels[0]
    if spec.kind == "entity_matching":
        return json.dumps({"match": True})
    if spec.kind == "pii_redaction":
        return "[]"
    if spec.kind == "summarisation":
        return "A short summary of the meeting and its decisions."
    if spec.kind == "extraction":
        fields = spec.fields or {}
        payload = {name: _dummy_field_value(fspec.type, fspec.values) for name, fspec in fields.items()}
        return json.dumps(payload)
    raise ValueError(f"unknown task kind {spec.kind!r}")


def _dummy_field_value(field_type: str, values: list[str] | None) -> Any:
    if field_type in ("enum", "currency") and values:
        return values[0]
    if field_type == "email":
        return "person@example.com"
    if field_type == "phone":
        return "+447700900123"
    if field_type == "country":
        return "GB"
    if field_type == "date":
        return "2026-01-01"
    if field_type == "int":
        return 1
    if field_type == "amount":
        return "100"
    return "Example"


def register_default_route(
    mock: respx.MockRouter, base_url: str, spec: TaskSpec, **response_kwargs: Any
) -> respx.Route:
    """Register a fake chat route that always answers correctly-shaped output for ``spec``."""
    return register_chat(mock, base_url, fixed_content_handler(default_valid_answer(spec), **response_kwargs))


def empty_length_response(**kwargs: Any) -> httpx.Response:
    """A response whose visible output is empty because the token cap was spent on reasoning."""
    return chat_response("", finish_reason="length", reasoning_tokens=200, completion_tokens=200, **kwargs)


def register_price_snapshot(
    mock: respx.MockRouter,
    base_url: str,
    model_names: list[str],
    *,
    prompt: float = 0.000001,
    completion: float = 0.000002,
) -> respx.Route:
    """Register a fake ``GET {base_url}/models`` price-list route for every given provider slug."""
    payload = {
        "data": [
            {"id": name, "pricing": {"prompt": str(prompt), "completion": str(completion)}} for name in model_names
        ]
    }
    return mock.get(f"{base_url.rstrip('/')}/models").mock(return_value=httpx.Response(200, json=payload))
