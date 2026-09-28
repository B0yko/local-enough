"""The bench runner: drives baselines, cloud models and local models over selected tasks/splits.

Every call becomes one prediction record appended to ``predictions.jsonl.gz``; ``--resume`` (or
simply reusing a run directory) skips whatever ``(model_id, task, split, item_id, pass)`` keys are
already recorded, so an interrupted run neither repeats work nor re-bills.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from local_enough import evaluate, paths
from local_enough.bench import envinfo, prices
from local_enough.bench.ledger import BudgetExceeded, Ledger, estimate_tokens, reservation_usd, settled_cost, usage_meta
from local_enough.bench.load import LoadSampler
from local_enough.bench.memory import sample_footprint, sample_system_delta, system_used_bytes
from local_enough.bench.rundir import PREDICTIONS, RunDir
from local_enough.config import (
    BaselineModel,
    CloudModel,
    Config,
    Constraints,
    LocalModel,
    RouteConfig,
)
from local_enough.providers.baselines import baseline_for
from local_enough.providers.caffeinate import spawn_caffeinate
from local_enough.providers.mlx import MlxServer
from local_enough.providers.openai_compat import ChatClient, ChatResult, Endpoint, endpoint_for
from local_enough.stats import p50_p95
from local_enough.tasks import registry
from local_enough.tasks.base import Item, Parsed, TaskKindModule, TaskSpec

PASS_A = "A"
PASS_B = "B"
_PREFLIGHT_MAX_TASKS = 5
_MEMORY_SAMPLE_INTERVAL_S = 5.0

DEFAULT_ROUTE_CONFIG = RouteConfig(
    workload_mix={
        "classification": 0.30,
        "extraction": 0.25,
        "pii_redaction": 0.20,
        "entity_matching": 0.15,
        "summarisation": 0.10,
    },
    reference_monthly_volume=50_000,
    constraints=Constraints(data_must_stay_local=["pii_redaction"]),
)


class PreflightFailed(RuntimeError):
    """A cloud model returned empty or reasoning-truncated output on a preflight call: a harness/config bug."""


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stratify_key(spec: TaskSpec) -> Callable[[Item], Any] | None:
    if spec.kind == "classification":
        return lambda item: item.get("label")
    if spec.kind == "entity_matching":
        return lambda item: bool(item.get("match"))
    if spec.kind == "pii_redaction":
        return lambda item: len(item.get("spans") or []) > 0
    return None


def _max_tokens_for(cfg: Config, spec: TaskSpec) -> int:
    """The visible-output cap: ``config.yaml``'s per-kind override, else the task's own default."""
    return cfg.max_tokens.get(spec.kind, spec.output_cap())


def _is_bundled(spec: TaskSpec) -> bool:
    if spec.root is None:
        return False
    try:
        spec.root.resolve().relative_to(paths.datasets_dir().resolve())
        return True
    except ValueError:
        return False


def _task_entry(spec: TaskSpec) -> dict[str, Any]:
    sha256: dict[str, str] = {}
    for split_name, filename in (("calib", spec.calib), ("test", spec.test), ("train", spec.train)):
        if filename is None or spec.root is None:
            continue
        path = spec.root / filename
        if path.exists():
            sha256[split_name] = _sha256_file(path)
    bundled = _is_bundled(spec)
    entry: dict[str, Any] = {"kind": spec.kind, "source": "bundled" if bundled else "custom", "sha256": sha256}
    if not bundled and spec.root is not None:
        rel = os.path.relpath(spec.root / "task.yaml", Path.cwd())
        if not rel.startswith("/"):
            entry["task_yaml_path"] = rel
    return entry


def _sanitise_config(cfg: Config) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for m in cfg.models:
        if isinstance(m, LocalModel):
            entry: dict[str, Any] = {"id": m.id, "kind": "local"}
            if m.launch is not None:
                entry["repo"] = m.launch.repo
                entry["revision"] = m.launch.revision
            models.append(entry)
        elif isinstance(m, CloudModel):
            entry = {"id": m.id, "kind": "cloud", "model": m.model}
            if m.role:
                entry["role"] = m.role
            models.append(entry)
        else:
            models.append({"id": m.id, "kind": "baseline", "task": m.task})
    out: dict[str, Any] = {"project": cfg.project, "models": models}
    if cfg.hardware is not None:
        hw = cfg.hardware
        out["hardware"] = {
            "name": hw.name,
            "purchase_price_usd": hw.purchase_price_usd,
            "price_label": hw.price_label,
            "price_source_url": hw.price_source_url,
            "price_date": hw.price_date,
            "lifetime_years": hw.lifetime_years,
            "allocation": hw.allocation,
            "busy_hours_per_day": hw.busy_hours_per_day,
            "electricity_usd_per_kwh": hw.electricity_usd_per_kwh,
            "ops_usd_per_month": hw.ops_usd_per_month,
            "power": {
                "mode": hw.power.mode,
                "incremental_watts": hw.power.incremental_watts,
                "idle_watts": hw.power.idle_watts,
            },
        }
    if cfg.judge is not None:
        out["judge"] = {"candidates": list(cfg.judge.candidates)}
    return out


def _merge_config(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    """Later passes may use a subset of the lineup (e.g. ``--local-only`` with a local config): keep the models
    earlier passes recorded, let the current config win for ids it defines."""
    if not previous:
        return current
    models = {m["id"]: m for m in previous.get("models", [])}
    models.update({m["id"]: m for m in current.get("models", [])})
    merged = {**previous, **current}
    merged["models"] = list(models.values())
    return merged


def _baseline_result(text: str, latency_s: float) -> ChatResult:
    return ChatResult(
        text=text,
        model="baseline",
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        cached_tokens=0,
        cost_usd=0.0,
        provider="baseline",
        finish_reason="stop",
        latency_s=latency_s,
        retries=0,
        temperature_sent=False,
        error=None,
        status=200,
    )


def _preflight_bad(result: ChatResult) -> bool:
    """Output cut by the cap: empty, or truncated at the length cap while reasoning tokens used part of it."""
    if not result.text.strip() and (result.finish_reason == "length" or result.reasoning_tokens > 0):
        return True
    return result.finish_reason == "length" and result.reasoning_tokens > 0


class _Runner:
    """Mutable state for one ``run_bench`` invocation."""

    def __init__(
        self,
        cfg: Config,
        tasks: list[TaskSpec],
        splits: list[str],
        run_dir: RunDir,
        *,
        limit: int | None,
        seed: int,
        local_only: bool,
        concurrency: int,
        route_cfg: RouteConfig | None,
        command: str,
    ) -> None:
        self.cfg = cfg
        self.tasks = tasks
        self.splits = splits
        self.run_dir = run_dir
        self.limit = limit
        self.seed = seed
        self.local_only = local_only
        self.concurrency = max(1, concurrency)
        self.route = route_cfg or DEFAULT_ROUTE_CONFIG
        self.route_cfg = route_cfg
        self.command = command

        self.baselines = [m for m in cfg.models if isinstance(m, BaselineModel)] if not local_only else []
        self.clouds = [m for m in cfg.models if isinstance(m, CloudModel)] if not local_only else []
        self.locals = [m for m in cfg.models if isinstance(m, LocalModel)]

        self.work = self._build_work()
        self.completed = run_dir.completed_keys()
        self.start_utc = _now_utc()

        self.buffer: list[dict[str, Any]] = []
        self._last_flush = time.monotonic()

        self.models_seen: dict[str, str] = {}
        self.new_segments: dict[tuple[str, str, str, str], dict[str, float]] = {}
        self.group_concurrency: dict[tuple[str, str, str, str], int] = {}
        self.memory_peaks: dict[str, dict[str, Any]] = {}
        self.load_windows: list[dict[str, Any]] = []

        self.ledger: Ledger | None = None
        self.snapshot: dict[str, Any] = {}
        self.caffeinate_pid: int | None = None
        self.budget_exceeded: BudgetExceeded | None = None

    def _build_work(self) -> dict[tuple[str, str], list[Item]]:
        work: dict[tuple[str, str], list[Item]] = {}
        for spec in self.tasks:
            key_fn = _stratify_key(spec)
            for split in self.splits:
                items = registry.load_items(spec, split)
                if self.limit is not None:
                    items = registry.stratified_sample(items, self.limit, self.seed, key_fn)
                work[(spec.name, split)] = items
        return work

    # -- ledger / pricing -------------------------------------------------

    def start_ledger(self) -> Ledger:
        self.ledger = Ledger(self.cfg.project, self.cfg.budget_usd, self.cfg.budget_warn_usd, run_dir=self.run_dir.path)
        return self.ledger

    async def ensure_price_snapshot(self) -> None:
        if not self.clouds:
            return
        if not self.run_dir.exists("price_snapshot.json"):
            snapshot = await _fetch_snapshot(self.clouds, self.cfg)
            self.run_dir.write_json("price_snapshot.json", snapshot)
        self.snapshot = self.run_dir.read_json("price_snapshot.json", {}) or {}

    def _price_for(self, model_id: str) -> dict[str, float | None]:
        entry = (self.snapshot.get("models") or {}).get(model_id) or {}
        return {"prompt": entry.get("prompt"), "completion": entry.get("completion")}

    # -- buffering ----------------------------------------------------------

    def _maybe_flush(self, force: bool = False) -> None:
        if not self.buffer:
            return
        if force or len(self.buffer) >= 25 or (time.monotonic() - self._last_flush) >= 10:
            self.run_dir.append_records(PREDICTIONS, self.buffer)
            self.buffer.clear()
            self._last_flush = time.monotonic()

    def _make_record(
        self,
        *,
        model_id: str,
        task: str,
        split: str,
        pass_: str,
        item_id: Any,
        result: ChatResult,
        parsed: Parsed,
        t_start: float,
        t_end: float,
    ) -> dict[str, Any]:
        return {
            "model_id": model_id,
            "task": task,
            "split": split,
            "item_id": str(item_id),
            "pass": pass_,
            "raw": result.text,
            "content": parsed.content,
            "valid": bool(parsed.ok and result.error is None),
            "error": result.error or parsed.error,
            "latency_s": result.latency_s,
            "t_start": t_start,
            "t_end": t_end,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "reasoning_tokens": result.reasoning_tokens,
            "cached_tokens": result.cached_tokens,
            "cost_usd": result.cost_usd,
            "provider": result.provider,
            "model": result.model,
            "retries": result.retries,
            "finish_reason": result.finish_reason,
            "temperature_sent": result.temperature_sent,
        }

    def _record(
        self,
        *,
        model_id: str,
        task: str,
        split: str,
        pass_: str,
        item_id: Any,
        result: ChatResult,
        parsed: Parsed,
        t_start: float,
        t_end: float,
    ) -> bool:
        record = self._make_record(
            model_id=model_id,
            task=task,
            split=split,
            pass_=pass_,
            item_id=item_id,
            result=result,
            parsed=parsed,
            t_start=t_start,
            t_end=t_end,
        )
        self.buffer.append(record)
        self.completed.add((model_id, task, split, str(item_id), pass_))
        self._maybe_flush()
        return bool(record["valid"])

    def _todo_items(self, model_id: str, spec: TaskSpec, split: str, pass_: str) -> list[Item]:
        items = self.work.get((spec.name, split), [])
        return [it for it in items if (model_id, spec.name, split, str(it["id"]), pass_) not in self.completed]

    def _set_segment(
        self, model_id: str, task: str, split: str, pass_: str, n: int, wall_s: float, concurrency: int
    ) -> None:
        if n <= 0:
            return
        key = (model_id, task, split, pass_)
        self.new_segments[key] = {"n": n, "wall_s": wall_s}
        self.group_concurrency[key] = concurrency

    def _progress(self, model_id: str, task: str, split: str, n: int, invalid_n: int) -> None:
        if n <= 0:
            return
        invalid_pct = invalid_n / n * 100.0
        spend = self.ledger.total_spent() if self.ledger is not None else 0.0
        print(f"{model_id} {task} {split}: n={n} invalid={invalid_pct:.1f}% spend=${spend:.4f}")

    # -- baselines ------------------------------------------------------

    async def run_baselines(self) -> None:
        for model_cfg in self.baselines:
            applicable = [spec for spec in self.tasks if spec.kind == model_cfg.task]
            for spec in applicable:
                baseline = baseline_for(model_cfg, spec)
                if baseline is None:
                    print(f"{model_cfg.id}: baseline unavailable for task {spec.name!r} (no train split); skipping")
                    continue
                module = registry.get_kind(spec.kind)
                self.models_seen[model_cfg.id] = model_cfg.id
                for split in self.splits:
                    todo = self._todo_items(model_cfg.id, spec, split, PASS_A)
                    if not todo:
                        continue
                    t0 = time.monotonic()
                    invalid = 0
                    for item in todo:
                        t_start = time.time()
                        call_t0 = time.perf_counter()
                        raw_text = baseline.respond(module.user_input(spec, item))
                        latency_s = time.perf_counter() - call_t0
                        t_end = time.time()
                        result = _baseline_result(raw_text, latency_s)
                        parsed = module.parse(spec, raw_text)
                        valid = self._record(
                            model_id=model_cfg.id,
                            task=spec.name,
                            split=split,
                            pass_=PASS_A,
                            item_id=item["id"],
                            result=result,
                            parsed=parsed,
                            t_start=t_start,
                            t_end=t_end,
                        )
                        invalid += 0 if valid else 1
                    wall_s = time.monotonic() - t0
                    self._set_segment(model_cfg.id, spec.name, split, PASS_A, len(todo), wall_s, 1)
                    self._progress(model_cfg.id, spec.name, split, len(todo), invalid)

    # -- cloud ------------------------------------------------------------

    async def run_cloud(self, client: ChatClient) -> None:
        for model_cfg in self.clouds:
            await self._preflight(client, model_cfg)
            endpoint = endpoint_for(model_cfg, timeout_s=model_cfg.timeout_s or self.route.timeouts_s.cloud)
            self.models_seen[model_cfg.id] = endpoint.display_model
            sem = asyncio.Semaphore(model_cfg.concurrency)
            for spec in self.tasks:
                module = registry.get_kind(spec.kind)
                for split in self.splits:
                    todo = self._todo_items(model_cfg.id, spec, split, PASS_A)
                    if not todo:
                        continue
                    t0 = time.monotonic()

                    async def _one(
                        item: Item,
                        spec: TaskSpec = spec,
                        split: str = split,
                        module: TaskKindModule = module,
                        sem: asyncio.Semaphore = sem,
                        model_cfg: CloudModel = model_cfg,
                        endpoint: Endpoint = endpoint,
                    ) -> bool | None:
                        async with sem:
                            return await self._call_cloud_item(client, model_cfg, endpoint, spec, module, split, item)

                    outcomes = await asyncio.gather(*(_one(it) for it in todo))
                    attempted = [o for o in outcomes if o is not None]
                    wall_s = time.monotonic() - t0
                    conc = model_cfg.concurrency
                    self._set_segment(model_cfg.id, spec.name, split, PASS_A, len(attempted), wall_s, conc)
                    invalid = sum(1 for o in attempted if not o)
                    self._progress(model_cfg.id, spec.name, split, len(attempted), invalid)
                    if self.budget_exceeded is not None:
                        raise self.budget_exceeded

    def _preflight_item(self, spec: TaskSpec) -> Item | None:
        items = self.work.get((spec.name, "calib"))
        if not items:
            try:
                items = registry.load_items(spec, "calib")
            except FileNotFoundError:
                return None
        return items[0] if items else None

    async def _preflight(self, client: ChatClient, model_cfg: CloudModel) -> None:
        endpoint = endpoint_for(model_cfg, timeout_s=model_cfg.timeout_s or self.route.timeouts_s.cloud)
        price = self._price_for(model_cfg.id)
        for spec in self.tasks[:_PREFLIGHT_MAX_TASKS]:
            item = self._preflight_item(spec)
            if item is None:
                continue
            key = (model_cfg.id, spec.name, "calib", str(item["id"]), PASS_A)
            if key in self.completed:
                continue
            module = registry.get_kind(spec.kind)
            messages = module.render_messages(spec, item)
            max_tokens = _max_tokens_for(self.cfg, spec)
            reserve_usd = self._reserve_estimate(messages, max_tokens, endpoint, price)
            meta = {"model_id": model_cfg.id, "command": self.command, "task": spec.name, "item_id": str(item["id"])}
            ledger = self._require_ledger()
            async with ledger.reserve(reserve_usd, meta) as reservation:
                t_start = time.time()
                result = await client.complete(endpoint, messages, max_tokens, temperature=0.0)
                t_end = time.time()
                self._settle(reservation, result, price)
            if _preflight_bad(result):
                raise PreflightFailed(
                    f"{model_cfg.id}: preflight on task {spec.name!r} returned no or truncated visible output "
                    f"(finish_reason={result.finish_reason!r}, reasoning_tokens={result.reasoning_tokens}). "
                    "This is a harness/config bug, not an invalid output: set reasoning_allowance_tokens "
                    f"(and 'reasoning' if needed) for model {model_cfg.id!r} in config.yaml and rerun."
                )
            parsed = module.parse(spec, result.text)
            self._record(
                model_id=model_cfg.id,
                task=spec.name,
                split="calib",
                pass_=PASS_A,
                item_id=item["id"],
                result=result,
                parsed=parsed,
                t_start=t_start,
                t_end=t_end,
            )

    def _reserve_estimate(
        self, messages: list[dict[str, str]], max_tokens: int, endpoint: Endpoint, price: dict[str, float | None]
    ) -> float:
        return reservation_usd(messages, max_tokens, endpoint.reasoning_allowance_tokens, price)

    def _settle(self, reservation: Any, result: ChatResult, price: dict[str, float | None]) -> None:
        actual, source = settled_cost(result, price)
        reservation.settle(actual, source, usage_meta(result))

    def _require_ledger(self) -> Ledger:
        if self.ledger is None:
            raise RuntimeError("ledger not started")
        return self.ledger

    async def _call_cloud_item(
        self,
        client: ChatClient,
        model_cfg: CloudModel,
        endpoint: Endpoint,
        spec: TaskSpec,
        module: TaskKindModule,
        split: str,
        item: Item,
    ) -> bool | None:
        if self.budget_exceeded is not None:
            return None
        messages = module.render_messages(spec, item)
        max_tokens = _max_tokens_for(self.cfg, spec)
        price = self._price_for(model_cfg.id)
        reserve_usd = self._reserve_estimate(messages, max_tokens, endpoint, price)
        meta = {"model_id": model_cfg.id, "command": self.command, "task": spec.name, "item_id": str(item["id"])}
        ledger = self._require_ledger()
        try:
            async with ledger.reserve(reserve_usd, meta) as reservation:
                t_start = time.time()
                result = await client.complete(endpoint, messages, max_tokens, temperature=0.0)
                t_end = time.time()
                self._settle(reservation, result, price)
        except BudgetExceeded as exc:
            if self.budget_exceeded is None:
                self.budget_exceeded = exc
            return None
        parsed = module.parse(spec, result.text)
        return self._record(
            model_id=model_cfg.id,
            task=spec.name,
            split=split,
            pass_=PASS_A,
            item_id=item["id"],
            result=result,
            parsed=parsed,
            t_start=t_start,
            t_end=t_end,
        )

    # -- local --------------------------------------------------------------

    async def run_local(self, client: ChatClient) -> None:
        for model_cfg in self.locals:
            pass_ = PASS_B if self.concurrency > 1 else PASS_A
            conc = self.concurrency if pass_ == PASS_B else 1
            if model_cfg.launch is not None:
                launch = model_cfg.launch
                server = MlxServer(
                    launch.repo,
                    launch.revision,
                    host=launch.host,
                    port=launch.port,
                    startup_timeout_s=launch.startup_timeout_s,
                )
                server.start()
                try:
                    endpoint = endpoint_for(
                        model_cfg,
                        base_url=server.base_url,
                        model_name="default_model",
                        display_model=server.display_model,
                        timeout_s=model_cfg.timeout_s or self.route.timeouts_s.local,
                    )
                    await self._run_one_local(client, model_cfg, endpoint, pass_, conc, server_pid=server.pid)
                finally:
                    server.stop()
            else:
                endpoint = endpoint_for(model_cfg, timeout_s=model_cfg.timeout_s or self.route.timeouts_s.local)
                await self._run_one_local(client, model_cfg, endpoint, pass_, conc, server_pid=None)

    async def _run_one_local(
        self,
        client: ChatClient,
        model_cfg: LocalModel,
        endpoint: Endpoint,
        pass_: str,
        conc: int,
        server_pid: int | None,
    ) -> None:
        self.models_seen[model_cfg.id] = endpoint.display_model
        own_pids = {os.getpid()}
        if server_pid is not None:
            own_pids.add(server_pid)
        if self.caffeinate_pid is not None:
            own_pids.add(self.caffeinate_pid)
        sampler = LoadSampler(own_pids_fn=lambda: own_pids)
        sampler.sample_once()
        await sampler.start()

        stop_mem = asyncio.Event()
        mem_task: asyncio.Task[tuple[int, str]] | None = None
        if server_pid is not None:
            idle_baseline = system_used_bytes()
            mem_task = asyncio.create_task(_sample_memory_peak(server_pid, idle_baseline, stop_mem))

        try:
            sem = asyncio.Semaphore(conc)
            for spec in self.tasks:
                module = registry.get_kind(spec.kind)
                for split in self.splits:
                    todo = self._todo_items(model_cfg.id, spec, split, pass_)
                    if not todo:
                        continue
                    t0 = time.monotonic()

                    async def _one(
                        item: Item, spec: TaskSpec = spec, split: str = split, module: TaskKindModule = module
                    ) -> bool:
                        async with sem:
                            return await self._call_local_item(
                                client, model_cfg, endpoint, spec, module, split, pass_, item
                            )

                    outcomes = await asyncio.gather(*(_one(it) for it in todo))
                    wall_s = time.monotonic() - t0
                    self._set_segment(model_cfg.id, spec.name, split, pass_, len(todo), wall_s, conc)
                    self._progress(model_cfg.id, spec.name, split, len(todo), sum(1 for o in outcomes if not o))
        finally:
            await sampler.stop()
            self._record_load_window(f"{pass_}:{model_cfg.id}", sampler)
            if mem_task is not None:
                stop_mem.set()
                peak_bytes, method = await mem_task
                disk_bytes = _snapshot_disk_bytes(model_cfg)
                self.memory_peaks[model_cfg.id] = {"peak_bytes": peak_bytes, "method": method, "disk_bytes": disk_bytes}

    async def _call_local_item(
        self,
        client: ChatClient,
        model_cfg: LocalModel,
        endpoint: Endpoint,
        spec: TaskSpec,
        module: TaskKindModule,
        split: str,
        pass_: str,
        item: Item,
    ) -> bool:
        messages = module.render_messages(spec, item)
        max_tokens = _max_tokens_for(self.cfg, spec)
        t_start = time.time()
        result = await client.complete(endpoint, messages, max_tokens, temperature=0.0)
        t_end = time.time()
        parsed = module.parse(spec, result.text)
        return self._record(
            model_id=model_cfg.id,
            task=spec.name,
            split=split,
            pass_=pass_,
            item_id=item["id"],
            result=result,
            parsed=parsed,
            t_start=t_start,
            t_end=t_end,
        )

    def _record_load_window(self, name: str, sampler: LoadSampler) -> None:
        if not sampler.samples:
            return
        self.load_windows.append(
            {
                "name": name,
                "start_utc": _iso(sampler.samples[0].t),
                "samples": [
                    {"t": s.t, "load1": s.load1, "other_cpu_pct": s.other_cpu_pct, "other_mem_gb": s.other_mem_gb}
                    for s in sampler.samples
                ],
                "contaminated": sampler.contaminated,
            }
        )

    # -- finalisation --------------------------------------------------------

    def _dataset_sha256(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for spec in self.tasks:
            for split_name, filename in (("calib", spec.calib), ("test", spec.test), ("train", spec.train)):
                if filename is None or spec.root is None:
                    continue
                path = spec.root / filename
                if path.exists():
                    out[f"{spec.name}:{split_name}"] = _sha256_file(path)
        return out

    def _template_sha256(self) -> dict[str, str]:
        out: dict[str, str] = {}
        seen_kinds = {spec.kind for spec in self.tasks}
        for kind, path in registry.template_files().items():
            if kind in seen_kinds and path.exists():
                out[kind] = _sha256_file(path)
        return out

    def _hardware_price(self) -> dict[str, Any]:
        if self.cfg.hardware is None:
            return {}
        hw = self.cfg.hardware
        return {
            "usd": hw.purchase_price_usd,
            "label": hw.price_label,
            "source_url": hw.price_source_url,
            "date": hw.price_date,
        }

    def _finalize_latency(self) -> None:
        if not self.new_segments:
            return
        existing = self.run_dir.read_json("latency.json", {}) or {}
        for (model_id, task, split, pass_), segment in self.new_segments.items():
            by_model = existing.setdefault(model_id, {})
            by_task = by_model.setdefault(task, {})
            by_split = by_task.setdefault(split, {})
            node = by_split.setdefault(pass_, {"segments": []})
            node["segments"].append(segment)

            group = (model_id, task, split, pass_)

            def _matches(r: dict[str, Any], group: tuple[str, str, str, str] = group) -> bool:
                return (r.get("model_id"), r.get("task"), r.get("split"), r.get("pass")) == group

            latencies = [float(r["latency_s"]) for r in self.run_dir.iter_records(PREDICTIONS) if _matches(r)]
            n_total = len(latencies)
            p50, p95 = p50_p95(latencies) if latencies else (float("nan"), float("nan"))
            mean_s = sum(latencies) / n_total if n_total else float("nan")
            sum_n = sum(s["n"] for s in node["segments"])
            sum_wall = sum(s["wall_s"] for s in node["segments"])
            tasks_per_hour = (sum_n / sum_wall * 3600.0) if sum_wall > 0 else 0.0
            node.update(
                {
                    "n": n_total,
                    "p50_s": p50,
                    "p95_s": p95,
                    "mean_s": mean_s,
                    "tasks_per_hour": tasks_per_hour,
                    "concurrency": self.group_concurrency[(model_id, task, split, pass_)],
                }
            )
        self.run_dir.write_json("latency.json", existing)

    def _finalize_memory(self) -> None:
        """Merge peaks into memory.json, keeping the highest peak seen for a model across passes."""
        if not self.memory_peaks:
            return
        existing = self.run_dir.read_json("memory.json", {}) or {}
        updates = {}
        for model_id, entry in self.memory_peaks.items():
            previous = existing.get(model_id) or {}
            keep_previous = int(previous.get("peak_bytes") or 0) > int(entry.get("peak_bytes") or 0)
            updates[model_id] = previous if keep_previous else entry
        self.run_dir.merge_json("memory.json", updates)

    def _finalize_load(self) -> None:
        if not self.load_windows:
            return
        existing = self.run_dir.read_json("load_samples.json", {}) or {}
        windows = list(existing.get("windows") or [])
        windows.extend(self.load_windows)
        existing["windows"] = windows
        self.run_dir.write_json("load_samples.json", existing)

    def _compute_metrics(self) -> dict[str, Any]:
        specs = {spec.name: spec for spec in self.tasks}
        metrics: dict[str, Any] = self.run_dir.read_json("metrics.json", {}) or {}
        for split in self.splits:
            scores_by_key = evaluate.score_run(self.run_dir, specs, split, pass_=PASS_A)
            for (model_id, task_name), scores in scores_by_key.items():
                spec = specs[task_name]
                summary = evaluate.summarise(spec, scores)
                metrics.setdefault(model_id, {}).setdefault(task_name, {})[split] = summary
        return metrics

    def finalize(self) -> None:
        self._maybe_flush(force=True)
        self._finalize_latency()
        self._finalize_memory()
        self._finalize_load()
        env_payload = envinfo.collect_env(
            models=self.models_seen,
            dataset_sha256=self._dataset_sha256(),
            template_sha256=self._template_sha256(),
            hardware_price=self._hardware_price(),
            start_utc=self.start_utc,
            end_utc=_now_utc(),
        )
        self.run_dir.merge_json("env.json", env_payload)
        self.run_dir.write_json(
            "config.json", _merge_config(self.run_dir.read_json("config.json"), _sanitise_config(self.cfg))
        )
        self.run_dir.write_json("route.json", self.route.model_dump())
        self.run_dir.write_json("tasks.json", {spec.name: _task_entry(spec) for spec in self.tasks})
        self.run_dir.write_json("metrics.json", self._compute_metrics())


async def _sample_memory_peak(pid: int, idle_baseline_bytes: int, stop_event: asyncio.Event) -> tuple[int, str]:
    peak = 0
    method = "footprint"
    use_footprint = True
    while True:
        try:
            sample = sample_footprint(pid) if use_footprint else sample_system_delta(idle_baseline_bytes)
            peak = max(peak, sample.bytes)
            method = sample.method
        except Exception:
            if use_footprint:
                use_footprint = False
            # keep the last known peak/method; the process may be between polls
        if stop_event.is_set():
            return peak, method
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=_MEMORY_SAMPLE_INTERVAL_S)


def _snapshot_disk_bytes(model_cfg: LocalModel) -> int | None:
    if model_cfg.launch is None:
        return None
    try:
        from huggingface_hub import snapshot_download

        resolved = snapshot_download(model_cfg.launch.repo, revision=model_cfg.launch.revision, local_files_only=True)
        snapshot_dir = Path(resolved)
    except Exception:
        return None
    total = 0
    for path in snapshot_dir.rglob("*"):
        if path.is_file():
            with contextlib.suppress(OSError):
                total += path.stat().st_size
    return total


async def _fetch_snapshot(clouds: list[CloudModel], cfg: Config) -> dict[str, Any]:
    base_urls = sorted({m.base_url for m in clouds} | ({cfg.judge.base_url} if cfg.judge else set()))
    combined_models: dict[str, Any] = {}
    taken_at = ""
    source_urls: list[str] = []
    for base_url in base_urls:
        names = {m.id: m.model for m in clouds if m.base_url == base_url}
        if cfg.judge is not None and cfg.judge.base_url == base_url:
            for slug in cfg.judge.candidates:
                names[slug] = slug
        if not names:
            continue
        snap = await prices.snapshot_prices(base_url, names)
        combined_models.update(snap["models"])
        taken_at = snap["taken_at_utc"]
        source_urls.append(snap["source_url"])
    source_url: Any = source_urls[0] if len(source_urls) == 1 else source_urls
    return {"taken_at_utc": taken_at, "source_url": source_url, "models": combined_models}


async def _dry_run(state: _Runner) -> None:
    priced_models: list[CloudModel] = list(state.clouds)
    snapshot_models: dict[str, Any] = {}
    if priced_models:
        snapshot = await _fetch_snapshot(priced_models, state.cfg)
        snapshot_models = snapshot["models"]

    report_models: list[LocalModel | CloudModel] = [*priced_models, *state.locals]
    total = 0.0
    for model_cfg in report_models:
        pass_ = PASS_A
        allowance = 0
        if isinstance(model_cfg, CloudModel):
            allowance = model_cfg.reasoning_allowance_tokens
        elif state.concurrency > 1:
            pass_ = PASS_B

        calls = 0
        input_tokens = 0
        output_tokens = 0
        for spec in state.tasks:
            module = registry.get_kind(spec.kind)
            max_tokens = _max_tokens_for(state.cfg, spec)
            for split in state.splits:
                todo = state._todo_items(model_cfg.id, spec, split, pass_)
                for item in todo:
                    input_tokens += estimate_tokens(module.render_messages(spec, item))
                calls += len(todo)
                output_tokens += (max_tokens + allowance) * len(todo)

        usd = 0.0
        if isinstance(model_cfg, CloudModel):
            price = snapshot_models.get(model_cfg.id, {})
            usd = input_tokens * (price.get("prompt") or 0.0) + output_tokens * (price.get("completion") or 0.0)
        total += usd
        print(f"{model_cfg.id}: calls={calls} input_tokens={input_tokens} output_tokens={output_tokens} usd=${usd:.4f}")
    print(f"total: ${total:.4f}")


async def run_bench(
    cfg: Config,
    tasks: list[TaskSpec],
    splits: list[str],
    run: Path,
    *,
    limit: int | None = None,
    seed: int = 7,
    local_only: bool = False,
    concurrency: int = 1,
    dry_run: bool = False,
    route_cfg: RouteConfig | None = None,
    command: str = "bench",
) -> None:
    """Run every selected model over every selected task/split and write the run directory.

    ``run`` must be pre-validated by the caller (the CLI): ``--resume`` requires it to already
    exist; a fresh ``--run``/default run dir does not. Completed ``(model_id, task, split, item_id,
    pass)`` keys already in ``run`` are always skipped, so this also implements resume.
    """
    state = _Runner(
        cfg,
        tasks,
        splits,
        RunDir(run),
        limit=limit,
        seed=seed,
        local_only=local_only,
        concurrency=concurrency,
        route_cfg=route_cfg,
        command=command,
    )
    if dry_run:
        await _dry_run(state)
        return

    caffeinate_proc = spawn_caffeinate(os.getpid())
    state.caffeinate_pid = caffeinate_proc.pid if caffeinate_proc is not None else None
    state.start_ledger()
    try:
        await state.ensure_price_snapshot()
        await state.run_baselines()
        async with ChatClient() as client:
            await state.run_cloud(client)
            await state.run_local(client)
    finally:
        state.finalize()
