from __future__ import annotations

import httpx
import pytest
import respx

from local_enough.config import CloudModel
from local_enough.providers.openai_compat import (
    ChatClient,
    Endpoint,
    MissingApiKey,
    endpoint_for,
)


def _endpoint(**overrides: object) -> Endpoint:
    base = dict(
        model_id="m1",
        base_url="https://example.test/api/v1",
        model_name="vendor/model",
        api_key="secret-key",
        kind="cloud",
        display_model="vendor/model",
        timeout_s=5.0,
    )
    base.update(overrides)
    return Endpoint(**base)  # type: ignore[arg-type]


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_complete_parses_usage_and_rewrites_model() -> None:
    endpoint = _endpoint(display_model="local-qwen@abc123")
    payload = {
        "id": "gen-1",
        "model": "/Users/x/.cache/huggingface/hub/models--mlx-community--Qwen3-4B/snapshots/abc123",
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cost": 0.0012,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    }
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://example.test/api/v1/chat/completions").mock(return_value=httpx.Response(200, json=payload))
        async with ChatClient(sleep_fn=_no_sleep) as client:
            result = await client.complete(endpoint, [{"role": "user", "content": "hi"}], max_tokens=16)

    assert result.model == "local-qwen@abc123"
    assert "/Users/x" not in result.model
    assert result.text == "hello"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.cached_tokens == 2
    assert result.reasoning_tokens == 1
    assert result.cost_usd == pytest.approx(0.0012)
    assert result.finish_reason == "stop"
    assert result.retries == 0
    assert result.error is None
    assert result.generation_id == "gen-1"


@pytest.mark.asyncio
async def test_retries_on_429_then_succeeds() -> None:
    endpoint = _endpoint()
    route = respx.post("https://example.test/api/v1/chat/completions")
    with respx.mock(assert_all_called=True) as mock:
        mock.route(url="https://example.test/api/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(429, json={"error": "rate limited"}),
                httpx.Response(500, json={"error": "boom"}),
                httpx.Response(
                    200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}
                ),
            ]
        )
        async with ChatClient(sleep_fn=_no_sleep) as client:
            result = await client.complete(endpoint, [{"role": "user", "content": "hi"}], max_tokens=16)
    assert result.retries == 2
    assert result.error is None
    assert result.text == "ok"
    del route


@pytest.mark.asyncio
async def test_gives_up_after_max_retries() -> None:
    endpoint = _endpoint()
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://example.test/api/v1/chat/completions").mock(return_value=httpx.Response(503))
        async with ChatClient(sleep_fn=_no_sleep, max_retries=3) as client:
            result = await client.complete(endpoint, [{"role": "user", "content": "hi"}], max_tokens=16)
    assert result.retries == 3
    assert result.error == "http_503"
    assert result.status == 503


@pytest.mark.asyncio
async def test_connect_error_is_retried_and_recorded() -> None:
    endpoint = _endpoint()
    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://example.test/api/v1/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
        async with ChatClient(sleep_fn=_no_sleep, max_retries=2) as client:
            result = await client.complete(endpoint, [{"role": "user", "content": "hi"}], max_tokens=16)
    assert result.retries == 2
    assert result.error == "ConnectError"
    assert result.status is None


@pytest.mark.asyncio
async def test_max_tokens_raised_by_reasoning_allowance_and_body_shape() -> None:
    endpoint = _endpoint(
        reasoning={"effort": "low"},
        provider_routing={"order": ["vendor"]},
        reasoning_allowance_tokens=100,
        extra_body={"seed": 7},
    )
    captured: dict[str, object] = {}

    def _responder(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}], "usage": {}})

    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://example.test/api/v1/chat/completions").mock(side_effect=_responder)
        async with ChatClient(sleep_fn=_no_sleep) as client:
            await client.complete(endpoint, [{"role": "user", "content": "hi"}], max_tokens=16, temperature=0.0)

    assert captured["max_tokens"] == 116
    assert captured["reasoning"] == {"effort": "low"}
    assert captured["provider"] == {"order": ["vendor"]}
    assert captured["seed"] == 7
    assert captured["temperature"] == 0.0


@pytest.mark.asyncio
async def test_local_endpoint_never_sends_reasoning_or_provider() -> None:
    endpoint = _endpoint(kind="local", reasoning={"effort": "low"}, provider_routing={"order": ["x"]}, api_key="")
    captured: dict[str, object] = {}

    def _responder(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        assert "authorization" not in {k.lower() for k in request.headers}
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}], "usage": {}})

    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://example.test/api/v1/chat/completions").mock(side_effect=_responder)
        async with ChatClient(sleep_fn=_no_sleep) as client:
            await client.complete(endpoint, [{"role": "user", "content": "hi"}], max_tokens=16)

    assert "reasoning" not in captured
    assert "provider" not in captured


def test_endpoint_repr_hides_api_key() -> None:
    endpoint = _endpoint(api_key="super-secret-value")
    assert "super-secret-value" not in repr(endpoint)


def test_endpoint_for_cloud_reads_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-abc")
    cfg = CloudModel(id="open-large", model="deepseek/deepseek-chat-v3.1")
    endpoint = endpoint_for(cfg)
    assert endpoint.api_key == "sk-abc"
    assert endpoint.kind == "cloud"
    assert endpoint.display_model == "deepseek/deepseek-chat-v3.1"


def test_endpoint_for_cloud_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = CloudModel(id="open-large", model="deepseek/deepseek-chat-v3.1")
    with pytest.raises(MissingApiKey):
        endpoint_for(cfg)


@pytest.mark.asyncio
async def test_fetch_generation_provider_best_effort() -> None:
    endpoint = _endpoint()
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://example.test/api/v1/generation").mock(
            return_value=httpx.Response(200, json={"data": {"provider_name": "Infermatic"}})
        )
        async with ChatClient(sleep_fn=_no_sleep) as client:
            provider = await client.fetch_generation_provider(endpoint, "gen-1")
    assert provider == "Infermatic"


@pytest.mark.asyncio
async def test_fetch_generation_provider_never_raises_on_failure() -> None:
    endpoint = _endpoint()
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://example.test/api/v1/generation").mock(side_effect=httpx.ConnectError("down"))
        async with ChatClient(sleep_fn=_no_sleep) as client:
            provider = await client.fetch_generation_provider(endpoint, "gen-1")
    assert provider is None
