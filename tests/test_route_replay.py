"""route replay: sampling, comparison with the offline simulate() run, and the local-only cloud-call
guarantee, exercised against a live uvicorn server on a free port with fake backends."""

from __future__ import annotations

import json
import socket
import threading
import time

import httpx
import respx
import uvicorn

from local_enough.bench.ledger import Ledger
from local_enough.bench.rundir import PREDICTIONS, RunDir
from local_enough.config import Constraints, QualityBar, RouteConfig
from local_enough.providers.openai_compat import Endpoint
from local_enough.route import gates, planner, replay
from local_enough.route.server import create_app
from local_enough.tasks.base import TaskSpec

CLOUD_URL = "https://cloud.test/v1"


def _write_jsonl(path, items) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _LiveServer:
    def __init__(self, app):
        port = _free_port()
        self.url = f"http://127.0.0.1:{port}"
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
        self.server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.02)

    def stop(self) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=5.0)


def _build_scenario(tmp_path, *, pii_unservable=False):
    # classification: served by a (fake) cloud model.
    cls_root = tmp_path / "cls"
    cls_root.mkdir()
    cls_spec = TaskSpec(name="classification", kind="classification", labels=["a", "b"], train="train.jsonl")
    cls_spec.root = cls_root
    _write_jsonl(
        cls_root / "train.jsonl",
        [{"id": f"tr{i}", "text": f"apple {i}", "label": "a"} for i in range(10)]
        + [{"id": f"tr{i}", "text": f"banana {i}", "label": "b"} for i in range(10, 20)],
    )
    cls_test = [
        {"id": "ct1", "text": "apple test one", "label": "a"},
        {"id": "ct2", "text": "apple test two", "label": "a"},
    ]
    _write_jsonl(cls_root / "test.jsonl", cls_test)
    _write_jsonl(cls_root / "calib.jsonl", cls_test)

    # pii_redaction: local-only, served entirely by the in-process regex baseline.
    pii_root = tmp_path / "pii"
    pii_root.mkdir()
    pii_spec = TaskSpec(name="pii_redaction", kind="pii_redaction")
    pii_spec.root = pii_root
    pii_text = "Contact john.smith@example.com about it."
    email = "john.smith@example.com"
    pt1_spans = [{"start": pii_text.index(email), "end": pii_text.index(email) + len(email), "type": "EMAIL"}]
    pii_test = [
        {"id": "pt1", "text": pii_text, "spans": pt1_spans if pii_unservable else []},
        {"id": "pt2", "text": "No personal data appears in this note.", "spans": []},
    ]
    _write_jsonl(pii_root / "test.jsonl", pii_test)
    _write_jsonl(pii_root / "calib.jsonl", pii_test)

    specs = {"classification": cls_spec, "pii_redaction": pii_spec}
    run = RunDir(tmp_path / "run")
    run.write_json(
        "config.json",
        {
            "project": "p",
            "models": [
                {"id": "cloud-p", "kind": "cloud", "model": "vendor/cloud-p"},
                {"id": "regex", "kind": "baseline", "task": "pii_redaction"},
            ],
        },
    )

    base = {"pass": "A", "valid": True, "latency_s": 0.01}
    records = []
    for split, items in (("calib", cls_test), ("test", cls_test)):
        for item in items:
            records.append(
                {
                    **base,
                    "task": "classification",
                    "split": split,
                    "model_id": "cloud-p",
                    "item_id": item["id"],
                    "raw": "a",
                    "cost_usd": 0.001,
                    "model": "vendor/cloud-p",
                }
            )
    for split, items in (("calib", pii_test), ("test", pii_test)):
        for item in items:
            records.append(
                {
                    **base,
                    "task": "pii_redaction",
                    "split": split,
                    "model_id": "regex",
                    "item_id": item["id"],
                    "raw": "[]",
                    "cost_usd": 0.0,
                }
            )
    run.append_records(PREDICTIONS, records)

    route_cfg = RouteConfig(
        workload_mix={"classification": 0.5, "pii_redaction": 0.5},
        reference_monthly_volume=1000,
        constraints=Constraints(data_must_stay_local=["pii_redaction"]),
        quality_bar={
            "default": QualityBar(relative_to_best=0.95),
            **({"pii_redaction": QualityBar(absolute=0.95)} if pii_unservable else {}),
        },
    )
    plan = planner.build_plan(run, specs, route_cfg, gates=True)
    gate_ctx = gates.build_gate_context(run, specs)
    endpoints = {
        "cloud-p": Endpoint(
            model_id="cloud-p",
            base_url=CLOUD_URL,
            model_name="cloud-p",
            api_key="k",
            kind="cloud",
            display_model="vendor/cloud-p",
        )
    }
    ledger = Ledger("test-proj", 5.0, None, run_dir=None)
    app = create_app(plan, specs, endpoints, gate_ctx, ledger, route_cfg)
    return run, specs, route_cfg, plan, app


def test_replay_against_live_uvicorn_with_fake_backends(tmp_path):
    run, specs, route_cfg, plan, app = _build_scenario(tmp_path)
    server = _LiveServer(app)
    server.start()
    try:
        with respx.mock(assert_all_called=False) as mock:
            mock.route(host="127.0.0.1").pass_through()
            mock.post(f"{CLOUD_URL}/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "gen-1",
                        "model": "ignored",
                        "choices": [{"message": {"content": "a"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                    },
                )
            )
            live_check = replay.run_replay(server.url, run, specs, route_cfg, plan, n=8, seed=7)
    finally:
        server.stop()

    assert live_check.n == 8
    assert live_check.router_p50_ms >= 0
    assert 0.0 <= live_check.decision_match_rate <= 1.0
    # every request the router made for pii_redaction (local-only) stayed off cloud entirely.
    assert live_check.local_only_cloud_calls == 0
    assert "classification" in live_check.per_task and "pii_redaction" in live_check.per_task
    # nothing in the report carries request/response text.
    dumped = json.dumps(live_check.to_dict())
    assert "apple" not in dumped and "banana" not in dumped and "john.smith" not in dumped


def test_replay_decisions_match_the_offline_simulate_run(tmp_path):
    run, specs, route_cfg, plan, app = _build_scenario(tmp_path)
    server = _LiveServer(app)
    server.start()
    try:
        with respx.mock(assert_all_called=False) as mock:
            mock.route(host="127.0.0.1").pass_through()
            mock.post(f"{CLOUD_URL}/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "gen-1",
                        "model": "ignored",
                        "choices": [{"message": {"content": "a"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                    },
                )
            )
            live_check = replay.run_replay(server.url, run, specs, route_cfg, plan, n=8, seed=7)
    finally:
        server.stop()

    # the fake backend always answers identically to what bench recorded, so the live router's decisions
    # should match the offline replay on every sampled item.
    assert live_check.decision_match_rate == 1.0
    assert live_check.mismatches == []


def test_a_refusal_the_simulation_also_predicts_counts_as_a_matching_decision(tmp_path):
    # The regex baseline misses the gold email, so no local candidate reaches the absolute PII bar: the plan marks
    # pii_redaction unservable, the router answers 503 and the offline replay serves nothing either.
    run, specs, route_cfg, plan, app = _build_scenario(tmp_path, pii_unservable=True)
    assert plan.tasks["pii_redaction"].status == planner.STATUS_UNSERVABLE
    server = _LiveServer(app)
    server.start()
    try:
        with respx.mock(assert_all_called=False) as mock:
            mock.route(host="127.0.0.1").pass_through()
            mock.post(f"{CLOUD_URL}/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "id": "gen-1",
                        "model": "ignored",
                        "choices": [{"message": {"content": "a"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                    },
                )
            )
            live_check = replay.run_replay(server.url, run, specs, route_cfg, plan, n=8, seed=7)
    finally:
        server.stop()

    assert live_check.mismatches == []
    assert live_check.decision_match_rate == 1.0
    assert live_check.per_task["pii_redaction"]["refused_503"] >= 1
