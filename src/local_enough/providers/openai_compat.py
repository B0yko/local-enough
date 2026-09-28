"""An OpenAI-compatible chat-completions client shared by local and cloud model runners.

Every provider (a launched ``mlx_lm.server``, a user's own local endpoint, or an OpenRouter-style
cloud endpoint) is called through the same ``ChatClient``. The response ``model`` field is always
rewritten to ``Endpoint.display_model`` before it reaches a caller: some local servers echo an
absolute cache path in that field, and it must never reach disk or a client.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import httpx

if TYPE_CHECKING:
    from local_enough.config import CloudModel, LocalModel

logger = logging.getLogger(__name__)

Message = dict[str, str]
"""An OpenAI chat message: ``{"role": ..., "content": ...}``."""

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class MissingApiKey(RuntimeError):
    """Raised when a cloud model's ``api_key_env`` is not set."""


@dataclass(frozen=True)
class Endpoint:
    """Everything ``ChatClient`` needs to call one model. Never serialise or log ``api_key``."""

    model_id: str
    base_url: str
    model_name: str
    api_key: str = field(repr=False)
    kind: Literal["local", "cloud"]
    display_model: str
    extra_body: dict[str, Any] = field(default_factory=dict)
    reasoning: dict[str, Any] | None = None
    provider_routing: dict[str, Any] | None = None
    reasoning_allowance_tokens: int = 0
    timeout_s: float = 60.0


@dataclass(frozen=True)
class ChatResult:
    """One completed (or failed) chat call, with the ``model`` field already rewritten."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    cost_usd: float | None
    provider: str | None
    finish_reason: str | None
    latency_s: float
    retries: int
    temperature_sent: bool
    error: str | None
    status: int | None
    generation_id: str | None = None


def endpoint_for(
    model_cfg: LocalModel | CloudModel,
    *,
    base_url: str | None = None,
    model_name: str | None = None,
    display_model: str | None = None,
    timeout_s: float | None = None,
) -> Endpoint:
    """Build an :class:`Endpoint` from a config entry.

    ``base_url``/``model_name``/``display_model`` override the config when given -- a launched
    ``mlx_lm.server`` supplies its own ``127.0.0.1:<port>`` base URL, ``"default_model"`` request
    name and ``<repo>@<sha>`` display model, none of which live in ``config.yaml``.
    """
    from local_enough.config import CloudModel as _CloudModel
    from local_enough.config import LocalModel as _LocalModel

    if isinstance(model_cfg, _CloudModel):
        api_key = os.environ.get(model_cfg.api_key_env)
        if not api_key:
            raise MissingApiKey(
                f"environment variable {model_cfg.api_key_env!r} is not set (needed for model {model_cfg.id!r})"
            )
        return Endpoint(
            model_id=model_cfg.id,
            base_url=base_url or model_cfg.base_url,
            model_name=model_name or model_cfg.model,
            api_key=api_key,
            kind="cloud",
            display_model=display_model or model_cfg.model,
            extra_body=dict(model_cfg.extra_body),
            reasoning=model_cfg.reasoning,
            provider_routing=model_cfg.provider,
            reasoning_allowance_tokens=model_cfg.reasoning_allowance_tokens,
            timeout_s=timeout_s or model_cfg.timeout_s or 60.0,
        )
    if isinstance(model_cfg, _LocalModel):
        resolved_base_url = base_url or model_cfg.base_url
        if resolved_base_url is None:
            raise ValueError(
                f"local model {model_cfg.id!r} has no base_url; launched mlx models must pass one from MlxServer"
            )
        key = ""
        if model_cfg.api_key_env:
            key = os.environ.get(model_cfg.api_key_env, "")
        return Endpoint(
            model_id=model_cfg.id,
            base_url=resolved_base_url,
            model_name=model_name or model_cfg.model or "default_model",
            api_key=key,
            kind="local",
            display_model=display_model or model_cfg.model or model_cfg.id,
            extra_body={},
            reasoning=None,
            provider_routing=None,
            reasoning_allowance_tokens=0,
            timeout_s=timeout_s or model_cfg.timeout_s or 60.0,
        )
    raise TypeError(f"endpoint_for: unsupported model config type {type(model_cfg)!r}")


class ChatClient:
    """Async chat-completions client with bounded retries. Never logs request or response bodies."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
    ) -> None:
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._sleep_fn = sleep_fn
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s

    async def __aenter__(self) -> ChatClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _build_body(
        self, endpoint: Endpoint, messages: list[Message], max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": endpoint.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if endpoint.kind == "cloud":
            if endpoint.reasoning:
                body["reasoning"] = endpoint.reasoning
            if endpoint.provider_routing:
                body["provider"] = endpoint.provider_routing
        if endpoint.extra_body:
            body.update(endpoint.extra_body)
        return body

    async def complete(
        self,
        endpoint: Endpoint,
        messages: Sequence[Message],
        max_tokens: int,
        *,
        temperature: float = 0.0,
    ) -> ChatResult:
        """Send one chat-completions call, retrying on 429/5xx/timeout/connect errors.

        ``max_tokens`` is the visible-output cap; the wire ``max_tokens`` is raised by
        ``endpoint.reasoning_allowance_tokens`` so a model that cannot switch off reasoning still
        returns visible content.
        """
        url = f"{endpoint.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
        sent_max_tokens = max_tokens + endpoint.reasoning_allowance_tokens
        body = self._build_body(endpoint, list(messages), sent_max_tokens, temperature)

        t0 = time.monotonic()
        retries = 0
        response: httpx.Response | None = None
        error: str | None = None
        status: int | None = None

        while True:
            try:
                response = await self._client.post(url, json=body, headers=headers, timeout=endpoint.timeout_s)
                status = response.status_code
                error = None
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                response = None
                status = None
                error = type(exc).__name__

            retryable = status is None or status in RETRYABLE_STATUS
            if not retryable or retries >= self._max_retries:
                break
            retries += 1
            logger.warning("retrying %s call: status=%s attempt=%d", endpoint.model_id, status, retries)
            await self._sleep_fn(self._backoff_base_s * (2 ** (retries - 1)))

        latency_s = time.monotonic() - t0

        if response is None:
            return ChatResult(
                text="",
                model=endpoint.display_model,
                prompt_tokens=0,
                completion_tokens=0,
                reasoning_tokens=0,
                cached_tokens=0,
                cost_usd=None,
                provider=None,
                finish_reason=None,
                latency_s=latency_s,
                retries=retries,
                temperature_sent=True,
                error=error or "connection_error",
                status=None,
            )
        if response.status_code >= 400:
            return ChatResult(
                text="",
                model=endpoint.display_model,
                prompt_tokens=0,
                completion_tokens=0,
                reasoning_tokens=0,
                cached_tokens=0,
                cost_usd=None,
                provider=None,
                finish_reason=None,
                latency_s=latency_s,
                retries=retries,
                temperature_sent=True,
                error=f"http_{response.status_code}",
                status=response.status_code,
            )

        data = response.json()
        choices = data.get("choices") or [{}]
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        raw_provider = data.get("provider")
        cost = usage.get("cost")
        raw_id = data.get("id")

        return ChatResult(
            text=str(message.get("content") or ""),
            model=endpoint.display_model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
            cached_tokens=int(prompt_details.get("cached_tokens") or 0),
            cost_usd=float(cost) if cost is not None else None,
            provider=raw_provider if isinstance(raw_provider, str) else None,
            finish_reason=choice.get("finish_reason"),
            latency_s=latency_s,
            retries=retries,
            temperature_sent=True,
            error=None,
            status=response.status_code,
            generation_id=raw_id if isinstance(raw_id, str) else None,
        )

    async def fetch_generation_provider(self, endpoint: Endpoint, generation_id: str) -> str | None:
        """Best-effort lookup of the upstream provider via ``GET /generation?id=``. Never raises."""
        url = f"{endpoint.base_url.rstrip('/')}/generation"
        headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
        for attempt in range(2):
            try:
                response = await self._client.get(url, params={"id": generation_id}, headers=headers, timeout=10.0)
                if response.status_code == 200:
                    payload = response.json()
                    record = payload.get("data", payload) if isinstance(payload, dict) else {}
                    provider = record.get("provider_name") if isinstance(record, dict) else None
                    return provider if isinstance(provider, str) else None
            except httpx.HTTPError:
                pass
            if attempt == 0:
                await self._sleep_fn(0.5)
        return None
