# Contributing

Thanks for looking at `local-enough`. This document covers development setup, the checks a change must pass,
and the shape of a few common contributions.

## Development setup

```
git clone https://github.com/B0yko/local-enough.git
cd local-enough
uv sync --locked
```

`uv sync --locked` installs the exact versions pinned in `uv.lock`, including the dev dependencies (pytest,
ruff, mypy). On Apple Silicon it also installs `mlx` and `mlx-lm`; on any other platform those two are skipped,
and only cloud models, remote OpenAI-compatible endpoints and the non-LLM baselines are available.

## Tests, lint and types

Run all three before opening a pull request; CI runs the same commands:

```
uv run pytest -q
uv run ruff check
uv run ruff format --check
uv run mypy
```

`mypy` runs in strict mode against `src/local_enough`. `ruff format --check` only checks formatting; run
`uv run ruff format` (without `--check`) to apply it.

Tests never make network calls. HTTP-based providers are tested against `respx`-mocked endpoints
(see `tests/fakes.py`); MLX and hub downloads are tested against fakes and fixtures, not real models. A test
suite that reaches the network, or that imports `mlx`/`mlx_lm` at collection time, breaks CI on Linux, which has
neither Apple Silicon nor MLX.

If your change touches live cloud calls or local model downloads, it belongs behind an explicit, manually run
command (`bench`, `judge calibrate`, `models pull`, ...), never in the default test suite or CI workflow.

## Adding a task kind

The five task kinds (`extraction`, `classification`, `pii_redaction`, `summarisation`, `entity_matching`) are
listed in `local_enough.config.TASK_KINDS`. Adding a sixth kind means:

1. Add the new kind to `TaskKind`/`TASK_KINDS` in `src/local_enough/config.py` and to
   `DEFAULT_MAX_TOKENS` in `src/local_enough/tasks/base.py`.
2. Add `src/local_enough/tasks/<kind>.py` exporting `MODULE: TaskKindModule` (see the `TaskKindModule` protocol
   in `tasks/base.py`): a prompt template under `tasks/templates/<kind>.j2`, a lenient parser built on
   `tasks/parsing.py`, a normaliser, a metric with a `primary` key and an `invalid_output_rate`, and the
   deterministic gate(s) this kind's router entries should use.
3. Register the module in `get_kind()` in `src/local_enough/tasks/registry.py`.
4. Document the bring-your-own JSONL shape for the new kind in `docs/add-your-task.md`.
5. If you're also bundling a dataset for it, add `src/local_enough/data/datasets/<kind>/` with a `task.yaml`,
   `calib.jsonl`, `test.jsonl` and a card entry in `DATASETS.md`; a hygiene test checks regeneration is
   byte-identical (see below).

A **bring-your-own task** of an existing kind needs none of this: point `--tasks` at your own `task.yaml`
(see `docs/add-your-task.md` and `examples/custom-task/`).

## Adding a provider

Any endpoint that speaks the OpenAI chat-completions API needs no code change: point a `cloud` model's
`base_url` and `api_key_env` at it, or run a `local` model with `base_url` set to a server you already run
(llama.cpp, Ollama, LM Studio). See `docs/configuration.md` for every `CloudModel`/`LocalModel` field.

Adding a new **non-LLM baseline** (the `kind: baseline` candidates: TF-IDF classification, regex PII, rapidfuzz
entity matching) means implementing it in `src/local_enough/providers/baselines.py` next to the existing three,
returning raw text so it goes through the same task-module parser and scorer as any LLM, and wiring it into
`baseline_for()` for a `BaselineTask` value.

## Regenerating datasets

The four synthetic datasets (`extraction`, `pii_redaction`, `summarisation`, `entity_matching`) and the
bring-your-own example (`examples/custom-task/`) are produced entirely offline by hand-written templates and
seeded value pools; `classification` samples the public BANKING77 CSVs. Regenerate all of them with:

```
uv run python scripts/build_datasets.py --seed 7
```

Regeneration with the same seed is byte-identical by construction (`tests/test_datasets_hygiene.py` asserts
this); never hand-edit a bundled `.jsonl` file. Update `DATASETS.md`'s per-task card (counts, distinct template
count, notes) if you change a generator.

## Commit style

Commits use [Conventional Commits](https://www.conventionalcommits.org/) (`feat(tasks): ...`,
`fix(bench): ...`, `test(ledger): ...`, `docs: ...`), kept small and scoped to one logical change.

## Reporting a bug or requesting a feature

Use the issue templates: **Bug report**, **Feature request**, or **Benchmark result on your hardware** if
you've run `local-enough bench`/`report` on your own machine and want to share the numbers. Security issues go
through [`SECURITY.md`](SECURITY.md) instead of a public issue.
