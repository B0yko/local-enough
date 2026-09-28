# Configuration reference

`local-enough` reads three kinds of YAML file, each backed by a typed pydantic model in
[`src/local_enough/config.py`](../src/local_enough/config.py) (`config.yaml`, `route.yaml`) or
[`src/local_enough/tasks/base.py`](../src/local_enough/tasks/base.py) (`task.yaml`). Every field on every model
listed here is documented below; `tests/test_docs_config.py` walks `model_fields` on each model and fails the
build if a field goes undocumented.

Run `local-enough init` to write a working `config.yaml`, `quickstart.yaml`, `local.yaml`, `route.yaml` and
`.env.example` into the current directory (see `examples/` for the same files, read on GitHub).

## Environment variables

Secrets and machine-local overrides never go into a committed YAML file; they come from the environment (see
`.env.example`):

| Variable | Used by | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | any model or judge candidate whose `api_key_env` is left at its default | API key for `https://openrouter.ai/api/v1`. A `CloudModel` or `JudgeConfig` entry can point `api_key_env` at a different variable name to use another OpenAI-compatible provider. |
| `LOCAL_ENOUGH_BUDGET_USD` | the cost ledger | Hard spend cap in USD. The effective cap is `min(config.yaml's budget_usd, LOCAL_ENOUGH_BUDGET_USD)`; unset means only `budget_usd` applies. |
| `LOCAL_ENOUGH_HOME` | the cost ledger, `models pull` | State directory for `ledgers/<project>.jsonl` and downloads. Defaults to `~/.local/share/local-enough`. Tests set it to a temporary directory so runs never touch a real one. |

## `config.yaml`

Loaded by `load_config(path) -> Config`. Lists the models to bench, the judge candidates and the hardware profile
used for local cost.

### `Config`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `project` | string, required | - | Ledger file name: spend is appended to `$LOCAL_ENOUGH_HOME/ledgers/<project>.jsonl`. |
| `budget_usd` | float > 0 | `5.0` | Hard spend cap in USD for this config, before the `LOCAL_ENOUGH_BUDGET_USD` override. |
| `budget_warn_usd` | float ≥ 0 or omitted | `null` | Print a warning once total spend crosses this amount; does not stop the run. |
| `local_models_max_gb` | float > 0 | `4.0` | `models pull` refuses to download configured local models past this combined size, checked through the Hub API before downloading. |
| `max_tokens` | mapping of task kind to int | `{}` | Per-kind override of the visible output cap (default per kind: classification 16, entity_matching 16, extraction 400, pii_redaction 500, summarisation 350). |
| `models` | list of `LocalModel` \| `CloudModel` \| `BaselineModel`, required | - | Every model or baseline this config benches or routes over; discriminated by `kind`. Ids must be unique. |
| `judge` | `JudgeConfig` or omitted | `null` | Required to run `judge calibrate` / `judge score` on the summarisation task. |
| `hardware` | `HardwareConfig` or omitted | `null` | Required for any local cost, break-even or capacity number. |

```yaml
project: local-enough-reference
budget_usd: 14.5
budget_warn_usd: 12
local_models_max_gb: 4.0
max_tokens: {classification: 24}
models: [ ... ]        # see below
```

### Models: `LocalModel`, `CloudModel`, `BaselineModel`

`models` is a list discriminated by `kind: local | cloud | baseline`. Every entry is an OpenAI-compatible chat
endpoint (`local`, `cloud`) or an in-process non-LLM baseline (`baseline`).

#### `LocalModel` (`kind: local`)

A model on hardware you own: either started by `local-enough` (`launch`) or an already-running OpenAI-compatible
server (`base_url`). Exactly one of `launch` or `base_url` must be set.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | string, required | - | Unique model id used in `--models`, reports and the route plan. |
| `kind` | literal `"local"` | `"local"` | Discriminator; always `local` for this shape. |
| `launch` | `MlxLaunch` or omitted | `null` | Set to have `local-enough` start and stop `mlx_lm.server` for this model. Mutually exclusive with `base_url`. |
| `base_url` | string or omitted | `null` | Base URL of an OpenAI-compatible server you already run (llama.cpp, Ollama, LM Studio). Mutually exclusive with `launch`. |
| `model` | string or omitted | `null` | Model name sent to `base_url`. Ignored when `launch` is set. |
| `api_key_env` | string or omitted | `null` | Env var holding a key for `base_url`, if that server needs one. |
| `concurrency` | int ≥ 1 | `1` | Default concurrent requests; Pass A always benches local models at concurrency 1 regardless of this value. |
| `timeout_s` | float > 0 or omitted | `null` | Per-call timeout; falls back to `route.yaml`'s `timeouts_s.local`. |

```yaml
- id: local-qwen3-4b
  kind: local
  launch: {runtime: mlx, repo: mlx-community/Qwen3-4B-Instruct-2507-4bit, revision: main, port: 8081}
- id: llama-cpp-local
  kind: local
  base_url: http://127.0.0.1:8080/v1
  model: default
```

#### `MlxLaunch`

How the CLI starts `mlx_lm.server` for a `LocalModel` with `launch` set.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `runtime` | literal `"mlx"` | `"mlx"` | Only runtime supported in v0.1. |
| `repo` | string, required | - | Hugging Face repo id, e.g. `mlx-community/Qwen3-4B-Instruct-2507-4bit`. |
| `revision` | string | `"main"` | Commit sha (recommended for reproducibility) or branch of the repo. |
| `port` | int, 1024-65535 | `8081` | Port the server listens on, on `host`. |
| `host` | string | `"127.0.0.1"` | Bind address. Anything other than `127.0.0.1` exposes the model server beyond this machine. |
| `startup_timeout_s` | float > 0 | `300.0` | Seconds to wait for the server to answer `GET /v1/models` before giving up. |

#### `CloudModel` (`kind: cloud`)

A hosted model behind an OpenAI-compatible endpoint, OpenRouter by default.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | string, required | - | Unique model id. |
| `kind` | literal `"cloud"` | `"cloud"` | Discriminator. |
| `role` | one of `frontier`, `small-closed`, `open-large`, `open-same-family`, or omitted | `null` | Lineup role shown in the report; informational only. |
| `base_url` | string | `"https://openrouter.ai/api/v1"` | OpenAI-compatible endpoint base URL. |
| `api_key_env` | string | `"OPENROUTER_API_KEY"` | Env var holding the API key. |
| `model` | string, required | - | Model id at the provider, e.g. `deepseek/deepseek-chat-v3.1`. |
| `concurrency` | int ≥ 1 | `8` | Bounded concurrent requests to this model. |
| `reasoning` | object or omitted | `null` | OpenRouter `reasoning` request object, e.g. `{effort: low}` or `{enabled: false}`, for models that cannot fully disable reasoning. |
| `reasoning_allowance_tokens` | int ≥ 0 | `0` | Added to the task's `max_tokens` cap when reasoning tokens would otherwise starve the visible answer. |
| `provider` | object or omitted | `null` | OpenRouter provider-routing object, e.g. `{order: [...], allow_fallbacks: false}`, to pin a provider when price or quantisation differs between them. |
| `extra_body` | object | `{}` | Extra fields sent verbatim in the request body. |
| `timeout_s` | float > 0 or omitted | `null` | Per-call timeout; falls back to `route.yaml`'s `timeouts_s.cloud`. |

```yaml
- id: frontier
  kind: cloud
  role: frontier
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY
  model: deepseek/deepseek-chat-v3.1
  reasoning: {effort: minimal}
  reasoning_allowance_tokens: 128
```

#### `BaselineModel` (`kind: baseline`)

A non-LLM baseline served in-process at zero cost: TF-IDF plus logistic regression for classification, regex for
PII, `rapidfuzz` field similarity for entity matching.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | string, required | - | Unique model id. |
| `kind` | literal `"baseline"` | `"baseline"` | Discriminator. |
| `task` | one of `classification`, `pii_redaction`, `entity_matching`, required | - | Task kind this baseline solves; it is only a candidate for that kind. |

```yaml
- {id: tfidf-baseline, kind: baseline, task: classification}
```

### `JudgeConfig`

Judge candidates for the summarisation rubric. `judge calibrate` scores every candidate and keeps the one with
the highest balanced accuracy; judges are never one of the benchmarked models.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `candidates` | list of strings, required | - | Model ids at `base_url` to calibrate and choose between. |
| `base_url` | string | `"https://openrouter.ai/api/v1"` | OpenAI-compatible endpoint base URL. |
| `api_key_env` | string | `"OPENROUTER_API_KEY"` | Env var holding the API key. |
| `concurrency` | int ≥ 1 | `8` | Bounded concurrent requests while calibrating or scoring. |
| `max_tokens` | int ≥ 16 | `600` | Visible output cap for one judge verdict. |
| `reasoning` | object or omitted | `null` | Same shape as `CloudModel.reasoning`. |
| `reasoning_allowance_tokens` | int ≥ 0 | `0` | Same meaning as `CloudModel.reasoning_allowance_tokens`. |
| `provider` | object or omitted | `null` | Same shape as `CloudModel.provider`. |

```yaml
judge:
  candidates: [openai/gpt-4.1-mini]
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY
```

### `HardwareConfig` and `PowerConfig`

The machine local cost, break-even and capacity are computed for. Required to get any local number in the report.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | string, required | - | Label for the hardware profile, shown in reports. |
| `purchase_price_usd` | float, required, must be > 0 | - | Purchase price of one machine in USD. `0` is rejected on purpose, so a forgotten price can't make local look free. |
| `price_source_url` | string or omitted | `null` | Where the price was read. Recorded alongside `price_date` in the report. |
| `price_date` | string (`YYYY-MM-DD`) or omitted | `null` | Date the price was read. |
| `price_label` | `"list price"` or `"configured"` | `"configured"` | `"list price"` only when the price is verified at `price_source_url` on `price_date`; otherwise it is shown as an assumption. |
| `lifetime_years` | float > 0 | `3.0` | Amortisation period for `purchase_price_usd` in `fixed_usd_per_month`. |
| `allocation` | float, 0 < x ≤ 1 | `1.0` | Share of the machine charged to this workload; `1.0` means a machine dedicated to `local-enough`. |
| `busy_hours_per_day` | float, 0 < x ≤ 24 | `8.0` | Hours per day the machine serves traffic, used for `capacity_per_month`. |
| `electricity_usd_per_kwh` | float ≥ 0 | `0.30` | Electricity price, stated as an assumption in the report. |
| `ops_usd_per_month` | float ≥ 0 | `0.0` | Any other recurring cost per machine per month (for example, none in the reference setup). |
| `power` | `PowerConfig` | `{mode: measured, incremental_watts: 20, idle_watts: 5}` | Power inputs to the cost model. |

`PowerConfig`:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `"measured"` or `"configured"` | `"measured"` | `measured`: use the committed `power-probe` result for this hardware. `configured`: always use the watts below, labelled "configured" everywhere they appear (for example, a desktop Mac Studio with no battery telemetry). |
| `incremental_watts` | float ≥ 0 | `20.0` | Extra watts while a local model serves requests, over idle. Used only in `configured` mode. |
| `idle_watts` | float ≥ 0 | `5.0` | Watts with the model loaded and idle. Used in `configured` mode and always in `fixed_usd_per_month`'s idle-energy term. |

```yaml
hardware:
  name: macbook-air-m5-24gb
  purchase_price_usd: 1499
  price_label: list price
  price_source_url: https://www.apple.com/shop/buy-mac/macbook-air
  price_date: "2026-09-01"
  lifetime_years: 3
  allocation: 1.0
  busy_hours_per_day: 8
  electricity_usd_per_kwh: 0.30
  ops_usd_per_month: 0
  power: {mode: measured, incremental_watts: 20, idle_watts: 5}
```

## `route.yaml`

Loaded by `load_route(path) -> RouteConfig`. Drives `local-enough route`'s plan, gates and cost tables; the same
file is used by `--simulate` and by the live server.

### `RouteConfig`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `quality_bar` | mapping of task name to `QualityBar` | `{default: {relative_to_best: 0.95}}` | Per-task quality bar; `"default"` applies to any task without its own entry. |
| `latency_p95_max_s` | mapping of task name to float or `null` | `{default: null}` | Optional p95 latency ceiling per task; a candidate over the cap is dropped from the plan. |
| `constraints` | `Constraints` | `{data_must_stay_local: []}` | Tasks that may never be routed to a cloud model. |
| `allow_below_bar_local` | bool | `false` | For a local-only task with no candidate meeting the bar: `true` serves it with the best local candidate anyway (`x-local-enough-gate: below-bar`); `false` makes the plan mark it `unservable` (HTTP 503 `local_only_below_bar`). |
| `workload_mix` | mapping of task name to float, required | - | Share of monthly volume per task; must sum to 1.0 (empty mapping is allowed only when unused). |
| `reference_monthly_volume` | float > 0, required | - | Total mixed tasks per month across the whole workload. |
| `expected_monthly_volume` | mapping of task name to float | `{}` | Optional per-task override of `V_t`, in place of `reference_monthly_volume × workload_mix[t]`. |
| `timeouts_s` | `Timeouts` | `{local: 60, cloud: 60}` | Default per-call timeout by model kind, used when a model doesn't set its own `timeout_s`. |

`QualityBar` (exactly one of the two fields must be set):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `relative_to_best` | float, 0 < x ≤ 1, or omitted | `null` | Fraction of the best candidate's `calib` score on that task (baselines included) a candidate must reach. |
| `absolute` | float, 0 ≤ x ≤ 1, or omitted | `null` | Absolute primary-metric threshold, used instead of a relative one (for example, PII recall must clear a fixed floor regardless of how other candidates do). |

`Constraints`:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `data_must_stay_local` | list of task names | `[]` | Tasks the router and the report never send to a cloud model, whatever the cost or break-even numbers say. |

`Timeouts`:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `local` | float > 0 | `60.0` | Default per-call timeout in seconds for local models. |
| `cloud` | float > 0 | `60.0` | Default per-call timeout in seconds for cloud models. |

```yaml
quality_bar:
  default: {relative_to_best: 0.95}
  pii_redaction: {absolute: 0.95}
latency_p95_max_s: {default: null, classification: 5}
constraints: {data_must_stay_local: [pii_redaction]}
allow_below_bar_local: false
workload_mix: {classification: 0.30, extraction: 0.25, pii_redaction: 0.20, entity_matching: 0.15, summarisation: 0.10}
reference_monthly_volume: 50000
expected_monthly_volume: {}
timeouts_s: {local: 60, cloud: 60}
```

### Volume semantics

Per-task monthly volume is

```
V_t = reference_monthly_volume × workload_mix[t]
```

unless `expected_monthly_volume.<t>` is set, in which case that value is used for `V_t` directly instead of the
computed one. `V_t` is the volume the **dedicated** cost scenario (per-task tables, frontier charts, break-even
table) assumes a machine serves for task `t` alone. The router's mixed-workload table uses the **shared**
scenario instead, where one machine's fixed cost is spread across every task it serves locally. Both scenarios,
the exact formulas and the break-even and capacity rules are defined in
[`docs/cost-model.md`](cost-model.md).

## `task.yaml`

Loaded by `local_enough.tasks.registry.load_task(path) -> TaskSpec`; the five bundled tasks and any
bring-your-own task (see [`docs/add-your-task.md`](add-your-task.md)) use this same shape.

### `TaskSpec`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `name` | string, required | - | Task name used in `--tasks`, `route.yaml` keys and the router alias `local-enough/<name>`. |
| `kind` | one of `extraction`, `classification`, `pii_redaction`, `summarisation`, `entity_matching`, required | - | Selects which task module renders prompts, parses and scores this dataset. |
| `description` | string | `""` | Free text, shown in reports. |
| `calib` | string | `"calib.jsonl"` | Relative path to the calib split JSONL. Used for routing decisions and gate tuning. |
| `test` | string | `"test.jsonl"` | Relative path to the test split JSONL. Every reported number comes from this split. |
| `train` | string or omitted | `null` | Relative path to an optional train split; only meaningful for `classification` (trains the TF-IDF baseline). |
| `labels` | list of strings or omitted | `null` | Required for `classification`: the full label set. |
| `fields` | mapping of field name to `FieldSpec`, or omitted | `null` | Required for `extraction`: the output schema, in output order. |
| `pii_types` | list of strings or omitted | `null` | Required for `pii_redaction`: the PII span types the policy covers. |
| `max_words` | int or omitted | `null` | Required for `summarisation`: the default summary word limit. |
| `max_tokens` | int or omitted | `null` | Overrides the kind's default visible-output token cap for this task. |
| `card` | `DatasetCard` or omitted | `null` | Source, licence and generation metadata for this dataset. |

`root` is set by the loader to the directory containing `task.yaml` and is never written into or read from
`task.yaml` itself (it is excluded from serialisation because it would be an absolute path).

```yaml
name: custom_support_tickets
kind: classification
description: Example bring-your-own task: a 3-label support-ticket classifier.
calib: calib.jsonl
test: test.jsonl
train: train.jsonl
labels: [billing, bug_report, how_to]
card: {source: synthetic example, licence: Apache-2.0, seed: 7, metric: accuracy}
```

### `FieldSpec`

One typed extraction field declared under `TaskSpec.fields`, in the order fields should appear in the model's
JSON output.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | one of `string`, `email`, `phone`, `country`, `date`, `enum`, `int`, `amount`, `currency`, required | - | Field type; selects the type-aware normaliser and gate used to score and ground this field. |
| `values` | list of strings or omitted | `null` | Allowed values for `enum` or `currency` fields. |
| `required` | bool | `false` | The router's extraction gate requires a non-null value for this field. |
| `grounded` | bool | `false` | Free text that must occur in the source document (normalised containment, or a fuzzy ratio above a calib-tuned threshold). |
| `nullable` | bool | `true` | Whether gold and predicted values may be `null`. |

```yaml
fields:
  company_name: {type: string, grounded: true, required: true}
  contact_email: {type: email, grounded: true}
  product: {type: enum, values: [alpha, beta, gamma], required: true}
  seats: {type: int, nullable: true}
```

### `DatasetCard`

Source, licence and generation metadata for one dataset, shown in the README's Data table and in
`local-enough datasets list`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `source` | string, required | - | Where the data comes from, e.g. `"synthetic"` or a cited public dataset. |
| `licence` | string, required | - | Licence of this dataset, e.g. `"Apache-2.0"` or `"CC-BY-4.0"`. |
| `seed` | int or omitted | `null` | Seed used by `scripts/build_datasets.py` to generate this dataset, when synthetic. |
| `metric` | string, required | - | Name of the primary metric reported for this task. |
| `distinct_templates` | int or omitted | `null` | Number of distinct generation templates behind this dataset, when synthetic (see `DATASETS.md`). |
| `notes` | string or omitted | `null` | Free text, shown in `DATASETS.md`. |
