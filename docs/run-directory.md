# Run directory

Every measured number comes from files in a run directory (`./runs/<UTC stamp>` by default, `--run DIR` to choose
one). The bundled reference run in `src/local_enough/data/reference/` has the same layout, so
`local-enough report --run reference` and `local-enough route --simulate --run reference` work offline. JSON files
use sorted keys; record files are gzip JSONL, appended one gzip member per flush. No file contains request headers,
API keys, host names, user names or filesystem paths.

| file | written by | content |
|---|---|---|
| `predictions.jsonl.gz` | `bench` | one record per call: `model_id, task, split, item_id, pass` (`A` = scored pass, `B` = concurrency-4 throughput pass), `raw` output, parsed `content`, `valid`, `error`, `latency_s`, `t_start/t_end`, token counts (`prompt, completion, reasoning, cached`), `cost_usd`, upstream `provider`, `model` (`<repo>@<sha>` for local), `retries`, `finish_reason`, `temperature_sent` |
| `metrics.json` | `bench` | per model, task and split: `n`, every metric and its 95% bootstrap CI (1,000 resamples, seed 7), Pass A only |
| `latency.json` | `bench` | per model, task, split and pass: `p50_s`, `p95_s`, `mean_s`, `tasks_per_hour`, `concurrency`, and the wall-clock `segments` (a resumed run adds a segment) |
| `cost_ledger.jsonl` | every paid command | copy of the ledger lines for this run: reserved and actual USD, cost source, token counts, provider |
| `price_snapshot.json` | first `bench` with a cloud model | dated per-token prices and supported parameters for the cloud and judge models |
| `env.json` | `bench` | allowlisted environment: machine model, chip, memory, macOS version and build, Python/mlx/mlx-lm/local-enough versions, model `repo@sha`, AC/battery state, charge, Low Power Mode, sha256 of every dataset file and prompt template, hardware price with source and date, UTC start and end per bench pass |
| `config.json`, `route.json`, `tasks.json` | `bench` | the sanitised model lineup and hardware block, the `route.yaml` used, and the tasks with file hashes |
| `load_samples.json` | local passes | load average and aggregate CPU %/memory of other processes (numbers only), sampled before and every 60 s; `contaminated` flag per window |
| `memory.json` | `bench` (local), `memory` | peak physical footprint per model server and method; `combined`: both models + router footprints and system memory in use |
| `soak.json` | `soak` | per-minute throughput, first-5 vs last-5-minute throughput and the throttle factor |
| `power.json` | `power-probe` | idle and incremental watts, or `mode: unavailable` with the reason |
| `downloads.json` | `models pull` | repo, revision sha, bytes, bytes newly downloaded, licence |
| `judge_calibration.json`, `judge_raw.jsonl.gz` | `judge calibrate` | per-candidate judge-calib metrics, the selected judge, its judge-holdout TPR/TNR/balanced accuracy/kappa and per-item labels; cached raw verdicts |
| `judge_scores.jsonl.gz` | `judge score` | per model and item: verdict, pass, fact recall, key-token agreement |
| `live_check.json` | `route replay` | router overhead, decision match rate against the simulation, mismatches, local-only cloud-call count |

`--resume DIR` (or reusing `--run DIR`) skips every `(model_id, task, split, item_id, pass)` already recorded, so an
interrupted run neither repeats work nor re-bills.

**The committed reference run** (`src/local_enough/data/reference/`) is the measured run as recorded, with two
changes made before committing: the ledger copy is stored gzip-compressed (`cost_ledger.jsonl.gz`), and in 5 of
11,930 prediction records the model output contained an email address outside the reserved `example.com/.org/.net`
domains (a model misspelling or inventing an address), which was replaced with `redacted@removed.example.com`. Neither
the original nor the replacement occurs in any source document, so every score, gate decision and table is unchanged;
`scripts/check_readme.py` regenerates the README tables from these files.
