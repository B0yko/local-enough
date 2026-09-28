# Bring your own task

`local-enough` benchmarks, reports and routes any task that fits one of its five kinds
(`extraction`, `classification`, `pii_redaction`, `summarisation`, `entity_matching`), as long as you provide a
`task.yaml` plus a `calib.jsonl` and a `test.jsonl` in the shape that kind expects. A complete, working example
lives in `examples/custom-task/`: a tiny made-up 3-label support-ticket classifier.

## Selecting your task

Point any command at your `task.yaml` with `--tasks`, alongside the bundled tasks:

```
local-enough bench --tasks all,path/to/your/task.yaml --models config.yaml
```

`all` expands to the five bundled tasks; add as many extra `task.yaml` paths as you like, comma-separated.
Using only your own task (without `all`) works the same way: `--tasks path/to/your/task.yaml`.

Validate a dataset before benchmarking it:

```
local-enough datasets validate path/to/your/task.yaml
local-enough datasets list --tasks all,path/to/your/task.yaml
```

`validate` checks every record in every split against your kind's schema and exits 1 with a list of errors on
failure.

## `task.yaml` fields

| Field | Type | Meaning |
|---|---|---|
| `name` | string, required | Used in `--tasks`, `route.yaml` keys and the router alias `local-enough/<name>`. |
| `kind` | one of the five kinds, required | Selects which task module scores this dataset. |
| `description` | string | Free text, shown in reports. |
| `calib` | string | Relative path to the calib split JSONL (default `calib.jsonl`). |
| `test` | string | Relative path to the test split JSONL (default `test.jsonl`). |
| `train` | string or omitted | Relative path to an optional train split (classification baseline only). |
| `labels` | list of strings | Required for `classification`: the full label set. |
| `fields` | mapping name -> field spec | Required for `extraction`: see below, in output order. |
| `pii_types` | list of strings | For `pii_redaction`: the span types your policy covers. |
| `max_words` | integer | For `summarisation`: the default word limit (items may override per record). |
| `max_tokens` | integer | Overrides the kind's default visible-output token cap for this task. |
| `card` | object or omitted | Dataset-card metadata: `source`, `licence`, `seed`, `metric`, `distinct_templates`, `notes`. |

All paths are relative to the directory containing `task.yaml`; `local-enough` resolves them for you.

### Extraction field specs (`fields`)

Each entry under `fields` describes one output field:

| Field | Type | Meaning |
|---|---|---|
| `type` | one of `string`, `email`, `phone`, `country`, `date`, `enum`, `int`, `amount`, `currency` | Drives the type-aware normalisation used when scoring. |
| `values` | list of strings | Allowed values for `enum`/`currency` fields. |
| `required` | bool (default `false`) | The router gate requires a non-null value for this field. |
| `grounded` | bool (default `false`) | Free text that must occur in the source document (router gate). |
| `nullable` | bool (default `true`) | Whether `null` is a valid value for this field. |

## JSONL record schema per kind

Every record needs an `id` (string, unique within the split) plus the fields below.

**classification** — `{id, text, label}`. An optional `train.jsonl` of the same shape trains the TF-IDF
baseline; without it, the baseline and the classification agreement gate are disabled for your task (never
trained on `calib` or `test`).

```json
{"id": "t-0001", "text": "My card was declined again today.", "label": "bug_report"}
```

**extraction** — `{id, text, gold: {...}}`, where `gold` has exactly the keys declared in `task.yaml`'s
`fields`, typed and `null` when absent.

```json
{"id": "e-0001", "text": "Date: Monday, 5 Jan 2026\n...", "gold": {"company_name": "Acme Ltd", "seats": null}}
```

**pii_redaction** — `{id, text, spans: [{start, end, type}]}`. `start`/`end` are character offsets into
`text` (0-indexed, half-open, so `text[start:end]` is the exact gold span).

```json
{"id": "p-0001", "text": "Contact: Jane Doe.", "spans": [{"start": 9, "end": 17, "type": "PERSON"}]}
```

**summarisation** — `{id, text, required_facts: [...], max_words}`. Each entry in `required_facts` is
`{fact, kind, key_tokens}`, where `kind` is a free-text label for your own bookkeeping (the bundled tasks use
`decision`, `action_item`, `amount`) and `key_tokens` holds whichever of `owner`/`date`/`amount` apply to that
fact, as plain strings.

```json
{
  "id": "s-0001",
  "text": "Priya: ...",
  "max_words": 120,
  "required_facts": [{"fact": "Priya will send the proposal by Friday.", "kind": "action_item",
                       "key_tokens": {"owner": "Priya", "date": "Friday"}}]
}
```

**entity_matching** — `{id, left: {...}, right: {...}, match}`. `left`/`right` are free-form JSON objects
(the bundled task uses `name`, `street`, `postcode`, `city`, `country`, `domain`, `vat_id`, `phone`); `match`
is a boolean.

```json
{"id": "m-0001", "left": {"name": "Acme Ltd"}, "right": {"name": "Acme Limited"}, "match": true}
```

## Regenerating the bundled example

`examples/custom-task/` is generated by `build_custom_example()` in `scripts/build_datasets.py`, the same
script that builds the five bundled tasks, so it stays in sync:

```
uv run python scripts/build_datasets.py --seed 7 --examples-out examples/custom-task
```
