from __future__ import annotations

import os
from pathlib import Path

import pytest
import respx

import fakes
from local_enough.bench import memcheck
from local_enough.bench.rundir import RunDir
from local_enough.config import Config, LocalModel, MlxLaunch, RouteConfig
from local_enough.providers.openai_compat import endpoint_for

LOCAL_URL = "https://local.test/v1"
CLOUD_URL = "https://cloud.test/v1"


class _FakeServer:
    pid = os.getpid()

    def stop(self) -> None:
        self.stopped = True


def test_memory_check_records_models_router_and_system(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = LocalModel(id="local-a", launch=MlxLaunch(repo="org/model", revision="abc", port=8123))
    cfg = Config(project="p", budget_usd=1.0, models=[model])
    route_cfg = RouteConfig(workload_mix={"classification": 1.0}, reference_monthly_volume=1000)
    server = _FakeServer()

    def fake_local_endpoint(model_cfg: LocalModel, _route: RouteConfig):  # type: ignore[no-untyped-def]
        plain = LocalModel(id=model_cfg.id, base_url=LOCAL_URL, model="default_model")
        return server, endpoint_for(plain, display_model="org/model@abc")

    used = iter([10 * 2**30, 14 * 2**30])
    monkeypatch.setattr(memcheck, "local_endpoint", fake_local_endpoint)
    monkeypatch.setattr(memcheck.memory, "footprint_bytes", lambda pid: 2**30)
    monkeypatch.setattr(memcheck.memory, "system_used_bytes", lambda: next(used))

    run = RunDir(tmp_path / "run")
    with respx.mock(assert_all_called=False) as mock:
        fakes.register_chat(mock, LOCAL_URL, fakes.fixed_content_handler("card_swallowed"))
        cloud = mock.post(f"{CLOUD_URL}/chat/completions")
        entry = memcheck.run_memory_check(cfg, route_cfg, run)
        assert not cloud.called

    assert entry["models_bytes"] == {"local-a": 2**30}
    assert entry["ours_bytes"] == 2 * 2**30
    assert (entry["system_used_before_bytes"], entry["system_used_bytes"]) == (10 * 2**30, 14 * 2**30)
    assert run.read_json("memory.json")["combined"]["router_bytes"] == 2**30
    assert server.stopped
