"""The OpenAI-compatible router (spec item 15): ``POST /v1/chat/completions`` plus ``/v1/models``,
``/healthz`` and ``/stats``.

The router only ever serves the benchmarked task prompts (ADR 5): the client calls model
``local-enough/<task>`` with the raw input as the last user message, and the server renders the same prompt
template ``bench`` used, calls the plan's fallback chain with :mod:`local_enough.route.gates` (the same gate
code ``route.simulate`` uses), and returns the canonical parsed answer. Request and response bodies are never
logged; the module logger records only task, model, status and latency.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from local_enough.bench.ledger import BudgetExceeded, Ledger, reservation_usd, settled_cost, usage_meta
from local_enough.config import RouteConfig
from local_enough.providers.openai_compat import ChatClient, Endpoint
from local_enough.route import gates as gates_module
from local_enough.route.planner import STATUS_UNSERVABLE, Plan, TaskPlan
from local_enough.stats import p50_p95
from local_enough.tasks import registry
from local_enough.tasks.base import Item, Parsed, TaskSpec

logger = logging.getLogger("local_enough.route.server")

MODEL_PREFIX = "local-enough/"

# Used only when the run has no price snapshot for a cloud model (e.g. a hand-made plan): a deliberately high
# per-token price so the reservation over-reserves rather than letting the cap be crossed.
_FALLBACK_PRICE = {"prompt": 2e-5, "completion": 2e-5}


def _task_from_model(model: str) -> str:
    return model[len(MODEL_PREFIX) :] if model.startswith(MODEL_PREFIX) else model


def _openai_error(message: str, error_type: str, code: str | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"message": message, "type": error_type}
    if code is not None:
        error["code"] = code
    return {"error": error}


def _last_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


@dataclass
class _TaskStats:
    count: int = 0
    escalations: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    calls_by_class: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def to_dict(self) -> dict[str, Any]:
        p50, p95 = p50_p95(self.latencies_ms)
        return {
            "count": self.count,
            "escalations": self.escalations,
            "p50_ms": p50,
            "p95_ms": p95,
            "calls_by_class": dict(self.calls_by_class),
        }


class _RouterStats:
    """In-process counters for ``/stats``. Never holds request/response text."""

    def __init__(self) -> None:
        self.tasks: dict[str, _TaskStats] = defaultdict(_TaskStats)
        self.spend_usd = 0.0

    def record(self, task: str, *, escalated: bool, latency_ms: float, calls_by_class: dict[str, int]) -> None:
        stat = self.tasks[task]
        stat.count += 1
        if escalated:
            stat.escalations += 1
        stat.latencies_ms.append(latency_ms)
        for cls, n in calls_by_class.items():
            stat.calls_by_class[cls] += n

    def to_dict(self) -> dict[str, Any]:
        return {"spend_usd": self.spend_usd, "tasks": {name: s.to_dict() for name, s in self.tasks.items()}}


@dataclass
class _Attempt:
    raw: str
    ok: bool
    latency_ms: float
    usage: dict[str, int]


@dataclass
class _ChainOutcome:
    served_model_id: str | None
    parsed: Parsed | None
    gate_label: str
    escalated: bool
    latency_ms: float
    usage: dict[str, int]
    calls_by_class: dict[str, int]


def _baseline_answer(gate_ctx: gates_module.GateContext, spec: TaskSpec, user_input: str) -> str | None:
    if spec.kind == "classification":
        tfidf = gate_ctx.tfidf.get(spec.name)
        return tfidf.respond(user_input) if tfidf else None
    if spec.kind == "pii_redaction":
        return gate_ctx.regex_pii.respond(user_input)
    if spec.kind == "entity_matching":
        fuzzy = gate_ctx.fuzzy.get(spec.name)
        return fuzzy.respond(user_input) if fuzzy else None
    return None


_EMPTY_USAGE: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


async def _attempt_cloud(
    model_id: str,
    task: str,
    endpoint: Endpoint,
    messages: list[dict[str, str]],
    max_tokens: int,
    client: ChatClient,
    ledger: Ledger,
    price: dict[str, Any],
) -> _Attempt | None:
    """One cloud call, reserved and settled in the ledger; ``None`` if the budget cap would be exceeded."""
    reserve_usd = reservation_usd(messages, max_tokens, endpoint.reasoning_allowance_tokens, price)
    try:
        meta = {"model_id": model_id, "command": "route", "task": task}
        async with ledger.reserve(reserve_usd, meta) as reservation:
            t0 = time.perf_counter()
            result = await client.complete(endpoint, messages, max_tokens, temperature=0.0)
            latency_ms = (time.perf_counter() - t0) * 1000
            actual, source = settled_cost(result, price)
            reservation.settle(actual, source, usage_meta(result))
    except BudgetExceeded:
        return None
    usage = {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }
    return _Attempt(raw=result.text, ok=result.error is None, latency_ms=latency_ms, usage=usage)


async def _attempt_local(
    endpoint: Endpoint, messages: list[dict[str, str]], max_tokens: int, client: ChatClient
) -> _Attempt:
    t0 = time.perf_counter()
    result = await client.complete(endpoint, messages, max_tokens, temperature=0.0)
    latency_ms = (time.perf_counter() - t0) * 1000
    usage = {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }
    return _Attempt(raw=result.text, ok=result.error is None, latency_ms=latency_ms, usage=usage)


def _attempt_baseline(gate_ctx: gates_module.GateContext, spec: TaskSpec, user_input: str) -> _Attempt:
    t0 = time.perf_counter()
    raw = _baseline_answer(gate_ctx, spec, user_input)
    latency_ms = (time.perf_counter() - t0) * 1000
    return _Attempt(raw=raw or "", ok=raw is not None, latency_ms=latency_ms, usage=dict(_EMPTY_USAGE))


async def _call_chain(
    *,
    tp: TaskPlan,
    spec: TaskSpec,
    item: Item,
    endpoints: dict[str, Endpoint],
    gate_ctx: gates_module.GateContext,
    ledger: Ledger,
    client: ChatClient,
    prices: dict[str, dict[str, Any]],
) -> _ChainOutcome:
    module = registry.get_kind(spec.kind)
    messages = module.render_messages(spec, item)
    user_input = module.user_input(spec, item)
    max_tokens = spec.output_cap()

    total_latency_ms = 0.0
    escalated = False
    usage = dict(_EMPTY_USAGE)
    calls_by_class: dict[str, int] = defaultdict(int)
    served_model_id: str | None = None
    served_parsed: Parsed | None = None
    gate_label = "n/a"
    fallback_parsed: Parsed | None = None

    for model_id in tp.chain_model_ids():
        kind = tp.candidate_kind(model_id) or "cloud"

        if kind == "baseline":
            attempt: _Attempt | None = _attempt_baseline(gate_ctx, spec, user_input)
        else:
            endpoint = endpoints.get(model_id)
            if endpoint is None:
                escalated = True
                continue
            if kind == "cloud":
                price = prices.get(model_id, _FALLBACK_PRICE)
                attempt = await _attempt_cloud(
                    model_id, spec.name, endpoint, messages, max_tokens, client, ledger, price
                )
            else:
                attempt = await _attempt_local(endpoint, messages, max_tokens, client)

        calls_by_class[kind] += 1
        if attempt is None:  # budget cap reached: never falls through to a further cloud candidate silently
            escalated = True
            continue

        total_latency_ms += attempt.latency_ms
        for key in usage:
            usage[key] += attempt.usage[key]
        if not attempt.ok:
            escalated = True
            continue

        parsed = module.parse(spec, attempt.raw)
        if not parsed.ok:
            escalated = True
            continue
        if model_id == tp.best_quality_model_id:
            fallback_parsed = parsed

        gr = gates_module.gate(spec, item, parsed, gate_ctx, format_only=kind == "baseline")
        if gr.passed:
            served_model_id = model_id
            gate_label = "format-only" if kind == "baseline" else "passed"
            served_parsed = parsed
            break
        escalated = True

    # Spec item 15: this "serve the best-quality answer anyway" fallback is only for a task that isn't
    # local-only. A local-only task whose chain is exhausted (every candidate errored or failed the gate,
    # including a gate failure on the last fallback) must 503 instead -- never serve a degraded answer.
    if served_model_id is None and fallback_parsed is not None and not tp.local_only:
        served_model_id = tp.best_quality_model_id
        served_parsed = fallback_parsed
        gate_label = "failed"

    return _ChainOutcome(
        served_model_id, served_parsed, gate_label, escalated, total_latency_ms, usage, dict(calls_by_class)
    )


def create_app(
    plan: Plan,
    specs: dict[str, TaskSpec],
    endpoints: dict[str, Endpoint],
    gate_ctx: gates_module.GateContext,
    ledger: Ledger,
    route_cfg: RouteConfig,
    prices: dict[str, dict[str, Any]] | None = None,
) -> FastAPI:
    """Build the FastAPI router app. ``endpoints`` covers every local/cloud model in the plan's chains;
    baseline candidates are served in-process from ``gate_ctx``'s own baseline objects. ``prices`` maps cloud
    model ids to their price-snapshot entry and sizes ledger reservations exactly as in ``bench``."""
    del route_cfg  # timeouts and other policy are already baked into `endpoints` and `plan` by the caller
    app = FastAPI(title="local-enough router", docs_url=None, redoc_url=None)
    stats = _RouterStats()
    price_table = prices or {}
    client = ChatClient()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()
        if body.get("stream"):
            return JSONResponse(
                status_code=400,
                content=_openai_error(
                    "streaming responses are not supported", "invalid_request_error", "stream_unsupported"
                ),
            )
        task_name = _task_from_model(str(body.get("model") or ""))
        tp = plan.tasks.get(task_name)
        spec = specs.get(task_name)
        if tp is None or spec is None:
            return JSONResponse(
                status_code=404,
                content=_openai_error(f"unknown task {task_name!r}", "invalid_request_error", "unknown_task"),
            )

        if tp.status == STATUS_UNSERVABLE:
            logger.info("task=%s model=- status=503 latency_ms=0", task_name)
            return JSONResponse(
                status_code=503,
                content=_openai_error(
                    f"task {task_name!r} has no candidate meeting the quality bar",
                    "server_error",
                    "local_only_below_bar",
                ),
            )

        raw_input = _last_user_message(list(body.get("messages") or []))
        module = registry.get_kind(spec.kind)
        item = module.item_from_input(spec, raw_input)

        outcome = await _call_chain(
            tp=tp,
            spec=spec,
            item=item,
            endpoints=endpoints,
            gate_ctx=gate_ctx,
            ledger=ledger,
            client=client,
            prices=price_table,
        )
        stats.record(
            task_name, escalated=outcome.escalated, latency_ms=outcome.latency_ms, calls_by_class=outcome.calls_by_class
        )
        stats.spend_usd = ledger.total_spent()

        if outcome.served_model_id is None or outcome.parsed is None:
            logger.info("task=%s model=- status=503 latency_ms=%.1f", task_name, outcome.latency_ms)
            code = "local_only_unavailable" if tp.local_only else "upstream_unavailable"
            return JSONResponse(
                status_code=503,
                content=_openai_error(f"no candidate for task {task_name!r} produced an answer", "server_error", code),
            )

        display_model = _display_model(outcome.served_model_id, tp, endpoints)
        logger.info("task=%s model=%s status=200 latency_ms=%.1f", task_name, display_model, outcome.latency_ms)
        response_body = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": display_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": outcome.parsed.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": outcome.usage,
        }
        return JSONResponse(
            content=response_body,
            headers={
                "x-local-enough-model": display_model,
                "x-local-enough-escalated": "true" if outcome.escalated else "false",
                "x-local-enough-gate": outcome.gate_label,
                "x-local-enough-upstream-ms": f"{outcome.latency_ms:.1f}",
            },
        )

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": f"{MODEL_PREFIX}{name}", "object": "model", "owned_by": "local-enough"}
                for name in sorted(plan.tasks)
            ],
        }

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats")
    async def get_stats() -> dict[str, Any]:
        return stats.to_dict()

    return app


def _display_model(model_id: str, tp: TaskPlan, endpoints: dict[str, Endpoint]) -> str:
    kind = tp.candidate_kind(model_id)
    if kind == "baseline":
        return model_id
    endpoint = endpoints.get(model_id)
    return endpoint.display_model if endpoint is not None else model_id
