# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-28

### Added

- `local-enough init` to write a starter `config.yaml`, `quickstart.yaml`, `local.yaml`, `route.yaml`,
  `.env.example` and a bring-your-own-task example into a new project directory.
- Five bundled tasks (`extraction`, `classification`, `pii_redaction`, `summarisation`, `entity_matching`),
  each with gold labels, a `calib` and a `test` split and a dataset card; `local-enough datasets list` and
  `local-enough datasets validate` to inspect and check them.
- Bring-your-own-task support: any of the five task kinds can be benchmarked, reported and routed against a
  user's own `task.yaml`, `calib.jsonl` and `test.jsonl`, documented in `docs/add-your-task.md`.
- One OpenAI-compatible interface for every model: an MLX launcher that starts and stops `mlx_lm.server` for a
  configured Hugging Face repo, any other OpenAI-compatible local or cloud endpoint, and three non-LLM
  baselines (TF-IDF classification, regex PII redaction, `rapidfuzz` entity matching).
- `local-enough models pull` to download configured local models into the Hugging Face cache, capped at
  `local_models_max_gb` and recorded in `downloads.json`.
- `local-enough bench` with a resumable, budget-capped run over every configured model, task and split,
  writing predictions, metrics, latency and an allow-listed `env.json` per run; `--dry-run` to estimate cost
  first and `--limit` for smoke runs.
- A cost ledger with a hard spend cap shared across `bench`, `judge` and `route`, backed by
  `$LOCAL_ENOUGH_HOME`/`LOCAL_ENOUGH_BUDGET_USD`.
- Local measurement commands: `local-enough soak` for sustained throughput and throttling, and
  `local-enough power-probe` for incremental watts without sudo, with a `configured`-watts fallback when
  telemetry is unavailable (for example, a desktop with no battery).
- A cost model (`docs/cost-model.md`, `local_enough.costmodel`) computing fixed and energy cost, break-even
  monthly volume, capacity and a dedicated-vs-shared-machine view of the mixed workload.
- A calibrated summarisation judge (`local-enough judge calibrate`/`judge score`) with Rogan-Gladen
  bias-corrected pass rates against a construction-labelled holdout set.
- `local-enough report` producing a self-contained HTML report, a Markdown report and a filled-in
  local-vs-cloud ADR, with accuracy-vs-cost frontier charts per task and deterministic per-task verdicts.
- `local-enough route`, a router that plans, simulates and serves an OpenAI-compatible
  `POST /v1/chat/completions` endpoint, sending each task to the cheapest candidate that meets its quality bar,
  applying deterministic confidence gates, and falling back along a provider-diverse chain.
- A committed, gzip-compressed reference run so `local-enough report --run reference` and
  `local-enough route --simulate --run reference` reproduce every README table fully offline.
- `docs/configuration.md`, a full field-by-field reference for `config.yaml`, `route.yaml` and `task.yaml`.
- GitHub Actions CI running lint, formatting, strict type checks and the test suite on every push and pull
  request, with no calls to paid APIs.
