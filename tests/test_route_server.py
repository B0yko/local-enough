"""The router: local-only guarantees, 503 error codes, gate-triggered escalation, no-body-in-logs, and
compatibility with the official ``openai`` client."""

from __future__ import annotations

import logging

import httpx
import openai
import pytest
import respx

from local_enough.bench.ledger import Ledger
from local_enough.providers.baselines import RegexPiiBaseline, TfidfBaseline
from local_enough.providers.openai_compat import Endpoint
from local_enough.route.gates import GateContext, GateThresholds
from local_enough.route.planner import CandidateRef, Plan, TaskPlan
from local_enough.route.server import create_app
from local_enough.tasks.base import TaskSpec

CLOUD_A = "https://cloud-a.test/v1"
CLOUD_B = "https://cloud-b.test/v1"
LOCAL_A = "https://local-a.test/v1"
LOCAL_B = "https://local-b.test/v1"


def _spec() -> TaskSpec:
    return TaskSpec(name="classification", kind="classification", labels=["a", "b"])


def _tfidf() -> TfidfBaseline:
    train = [{"text": f"apple {i}", "label": "a"} for i in range(10)] + [
        {"text": f"banana {i}", "label": "b"} for i in range(10, 20)
    ]
    return TfidfBaseline(train)


def _ref(model_id: str, kind: str, score: float = 1.0, usd: float = 0.001) -> CandidateRef:
    return CandidateRef(model_id=model_id, kind=kind, provider=kind, role=None, score=score, usd_per_task=usd)


def _task_plan(
    *, status="served", local_only=False, primary=None, fallbacks=None, best_quality_model_id=None, format_only=False
):
    fallbacks = fallbacks or []
    return TaskPlan(
        task="classification",
        status=status,
        quality="primary",
        bar_kind="relative_to_best",
        bar_value=0.95,
        best_score=1.0,
        local_only=local_only,
        latency_p95_max_s=None,
        primary=primary,
        fallbacks=fallbacks,
        survivors=[primary, *fallbacks] if primary else [],
        dropped=[],
        format_only_gate=format_only,
        below_bar=False,
        best_quality_model_id=best_quality_model_id,
        extraction_grounding_ratio=None,
        entity_matching_band=None,
        gate_label="passed",
    )


def _endpoint(model_id: str, base_url: str, kind: str = "cloud") -> Endpoint:
    return Endpoint(
        model_id=model_id,
        base_url=base_url,
        model_name=model_id,
        api_key="key" if kind == "cloud" else "",
        kind=kind,  # type: ignore[arg-type]
        display_model=model_id,
    )


def _ledger(tmp_path, cap_usd: float = 5.0) -> Ledger:
    return Ledger("test-proj", cap_usd, None, run_dir=None)


def _plan(tp: TaskPlan) -> Plan:
    return Plan(
        run="test",
        gates_enabled=True,
        local_only_tasks=["classification"] if tp.local_only else [],
        tasks={"classification": tp},
    )


def _gate_ctx(*, tfidf: TfidfBaseline | None = None) -> GateContext:
    return GateContext(
        tfidf={"classification": tfidf} if tfidf else {},
        fuzzy={},
        regex_pii=RegexPiiBaseline(),
        thresholds=GateThresholds(),
    )


async def _post(app, body):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/v1/chat/completions", json=body)


@pytest.mark.asyncio
async def test_unservable_task_returns_503_local_only_below_bar(tmp_path):
    tp = _task_plan(status="unservable")
    app = create_app(_plan(tp), {"classification": _spec()}, {}, _gate_ctx(), _ledger(tmp_path), None)
    with respx.mock(assert_all_called=False):
        resp = await _post(
            app, {"model": "local-enough/classification", "messages": [{"role": "user", "content": "apple"}]}
        )
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "local_only_below_bar"


@pytest.mark.asyncio
async def test_local_only_exhaustion_returns_503_and_never_calls_cloud(tmp_path):
    primary = _ref("local-a", "local")
    fallback = _ref("local-b", "local")
    tp = _task_plan(local_only=True, primary=primary, fallbacks=[fallback], best_quality_model_id="local-b")
    endpoints = {
        "local-a": _endpoint("local-a", LOCAL_A, kind="local"),
        "local-b": _endpoint("local-b", LOCAL_B, kind="local"),
    }
    app = create_app(_plan(tp), {"classification": _spec()}, endpoints, _gate_ctx(), _ledger(tmp_path), None)

    with respx.mock(assert_all_called=False) as mock:
        mock.post(f"{LOCAL_A}/chat/completions").mock(return_value=httpx.Response(500))
        mock.post(f"{LOCAL_B}/chat/completions").mock(return_value=httpx.Response(500))
        cloud_route = mock.post(f"{CLOUD_A}/chat/completions").mock(
            side_effect=AssertionError("a local-only request must never reach a cloud endpoint")
        )
        resp = await _post(
            app, {"model": "local-enough/classification", "messages": [{"role": "user", "content": "apple"}]}
        )

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "local_only_unavailable"
    assert not cloud_route.called


@pytest.mark.asyncio
async def test_local_only_gate_failure_on_last_fallback_returns_503_not_a_degraded_answer(tmp_path):
    """The "serve the best-quality candidate's answer with gate: failed" fallback is only for
    a task that *isn't* local-only. A local-only task whose gate fails on every candidate (not just an HTTP
    error) must still 503 ``local_only_unavailable`` -- it must not silently return a gate-failing answer."""
    tfidf = _tfidf()
    primary = _ref("local-a", "local")
    fallback = _ref("local-b", "local")
    tp = _task_plan(local_only=True, primary=primary, fallbacks=[fallback], best_quality_model_id="local-b")
    endpoints = {
        "local-a": _endpoint("local-a", LOCAL_A, kind="local"),
        "local-b": _endpoint("local-b", LOCAL_B, kind="local"),
    }
    app = create_app(_plan(tp), {"classification": _spec()}, endpoints, _gate_ctx(tfidf=tfidf), _ledger(tmp_path), None)

    with respx.mock(assert_all_called=True) as mock:
        # both answer with a valid, parseable label that disagrees with the baseline -> gate fails on both.
        mock.post(f"{LOCAL_A}/chat/completions").mock(return_value=httpx.Response(200, json=_completion_body("b")))
        mock.post(f"{LOCAL_B}/chat/completions").mock(return_value=httpx.Response(200, json=_completion_body("b")))
        resp = await _post(
            app, {"model": "local-enough/classification", "messages": [{"role": "user", "content": "apple report"}]}
        )

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "local_only_unavailable"


@pytest.mark.asyncio
async def test_gate_failure_escalates_to_the_agreeing_fallback(tmp_path):
    tfidf = _tfidf()
    primary = _ref("cloud-p", "cloud", usd=0.001)
    fallback = _ref("cloud-f", "cloud", usd=0.002)
    tp = _task_plan(primary=primary, fallbacks=[fallback], best_quality_model_id="cloud-f")
    endpoints = {"cloud-p": _endpoint("cloud-p", CLOUD_A), "cloud-f": _endpoint("cloud-f", CLOUD_B)}
    app = create_app(_plan(tp), {"classification": _spec()}, endpoints, _gate_ctx(tfidf=tfidf), _ledger(tmp_path), None)

    with respx.mock(assert_all_called=True) as mock:
        mock.post(f"{CLOUD_A}/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion_body("b"))  # disagrees with the "apple" -> "a" baseline
        )
        mock.post(f"{CLOUD_B}/chat/completions").mock(return_value=httpx.Response(200, json=_completion_body("a")))
        resp = await _post(
            app, {"model": "local-enough/classification", "messages": [{"role": "user", "content": "apple report"}]}
        )

    assert resp.status_code == 200
    assert resp.headers["x-local-enough-escalated"] == "true"
    assert resp.headers["x-local-enough-gate"] == "passed"
    assert resp.headers["x-local-enough-model"] == "cloud-f"
    assert resp.json()["choices"][0]["message"]["content"] == "a"


@pytest.mark.asyncio
async def test_all_candidates_fail_gate_serves_best_quality_with_gate_failed(tmp_path):
    tfidf = _tfidf()
    primary = _ref("cloud-p", "cloud", usd=0.001)
    fallback = _ref("cloud-f", "cloud", usd=0.002)
    tp = _task_plan(primary=primary, fallbacks=[fallback], best_quality_model_id="cloud-f")
    endpoints = {"cloud-p": _endpoint("cloud-p", CLOUD_A), "cloud-f": _endpoint("cloud-f", CLOUD_B)}
    app = create_app(_plan(tp), {"classification": _spec()}, endpoints, _gate_ctx(tfidf=tfidf), _ledger(tmp_path), None)

    with respx.mock(assert_all_called=True) as mock:
        mock.post(f"{CLOUD_A}/chat/completions").mock(return_value=httpx.Response(200, json=_completion_body("b")))
        mock.post(f"{CLOUD_B}/chat/completions").mock(return_value=httpx.Response(200, json=_completion_body("b")))
        resp = await _post(
            app, {"model": "local-enough/classification", "messages": [{"role": "user", "content": "apple report"}]}
        )

    assert resp.status_code == 200
    assert resp.headers["x-local-enough-gate"] == "failed"
    assert resp.headers["x-local-enough-model"] == "cloud-f"


@pytest.mark.asyncio
async def test_budget_cap_skips_a_cloud_candidate_and_escalates(tmp_path):
    primary = _ref("cloud-expensive", "cloud", usd=1.0)
    fallback = _ref("tfidf", "baseline", usd=0.0)
    tp = _task_plan(primary=primary, fallbacks=[fallback], best_quality_model_id="tfidf", format_only=True)
    endpoints = {"cloud-expensive": _endpoint("cloud-expensive", CLOUD_A)}
    ledger = Ledger("test-proj", 0.0000001, None, run_dir=None)  # too small for any cloud reservation
    app = create_app(_plan(tp), {"classification": _spec()}, endpoints, _gate_ctx(tfidf=_tfidf()), ledger, None)

    with respx.mock(assert_all_called=False) as mock:
        cloud_route = mock.post(f"{CLOUD_A}/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion_body("a"))
        )
        resp = await _post(
            app, {"model": "local-enough/classification", "messages": [{"role": "user", "content": "apple"}]}
        )

    assert resp.status_code == 200
    assert not cloud_route.called
    assert resp.headers["x-local-enough-model"] == "tfidf"


@pytest.mark.asyncio
async def test_stream_true_is_rejected(tmp_path):
    tp = _task_plan(primary=_ref("tfidf", "baseline"), best_quality_model_id="tfidf", format_only=True)
    app = create_app(_plan(tp), {"classification": _spec()}, {}, _gate_ctx(tfidf=_tfidf()), _ledger(tmp_path), None)
    resp = await _post(
        app,
        {"model": "local-enough/classification", "messages": [{"role": "user", "content": "apple"}], "stream": True},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_unknown_task_returns_404(tmp_path):
    tp = _task_plan(primary=_ref("tfidf", "baseline"), best_quality_model_id="tfidf")
    app = create_app(_plan(tp), {"classification": _spec()}, {}, _gate_ctx(tfidf=_tfidf()), _ledger(tmp_path), None)
    resp = await _post(app, {"model": "local-enough/nonexistent", "messages": [{"role": "user", "content": "x"}]})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_healthz_and_models_and_stats(tmp_path):
    tp = _task_plan(primary=_ref("tfidf", "baseline"), best_quality_model_id="tfidf", format_only=True)
    app = create_app(_plan(tp), {"classification": _spec()}, {}, _gate_ctx(tfidf=_tfidf()), _ledger(tmp_path), None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/healthz")).json() == {"status": "ok"}
        models = (await client.get("/v1/models")).json()
        assert models["data"][0]["id"] == "local-enough/classification"

        await client.post(
            "/v1/chat/completions",
            json={"model": "local-enough/classification", "messages": [{"role": "user", "content": "apple"}]},
        )
        stats = (await client.get("/stats")).json()
    assert stats["tasks"]["classification"]["count"] == 1


@pytest.mark.asyncio
async def test_request_and_response_bodies_never_reach_logs(tmp_path, caplog):
    marker = "MARKER_SECRET_DO_NOT_LOG_9f3c"
    tp = _task_plan(primary=_ref("tfidf", "baseline"), best_quality_model_id="tfidf", format_only=True)
    app = create_app(_plan(tp), {"classification": _spec()}, {}, _gate_ctx(tfidf=_tfidf()), _ledger(tmp_path), None)
    with caplog.at_level(logging.DEBUG, logger="local_enough.route.server"):
        resp = await _post(
            app, {"model": "local-enough/classification", "messages": [{"role": "user", "content": marker}]}
        )
    assert resp.status_code == 200
    assert marker not in caplog.text


@pytest.mark.asyncio
async def test_official_openai_client_round_trips_against_the_router(tmp_path):
    tp = _task_plan(primary=_ref("tfidf", "baseline"), best_quality_model_id="tfidf", format_only=True)
    app = create_app(_plan(tp), {"classification": _spec()}, {}, _gate_ctx(tfidf=_tfidf()), _ledger(tmp_path), None)
    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    client = openai.AsyncOpenAI(base_url="http://test/v1", api_key="unused", http_client=http_client)
    try:
        response = await client.chat.completions.create(
            model="local-enough/classification", messages=[{"role": "user", "content": "apple report please"}]
        )
    finally:
        await http_client.aclose()
    assert response.choices[0].message.content in {"a", "b"}
    assert response.model == "tfidf"


def _completion_body(content: str) -> dict:
    return {
        "id": "gen-1",
        "model": "ignored",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
