# ADR 7: Model lineup for the reference run

- Status: accepted
- Date: 2026-09-28 (written before the full run)

## Context

The reference run compares two local models with four cloud models, one per role, plus three non-LLM baselines.
The reference-run budget caps the `--dry-run` estimate of the full run (both splits, 1,482 calls per model) at $4.50 for the
frontier role and $6.50 for the four cloud models together. The dry-run estimate assumes every call uses its full
visible-output cap plus any reasoning allowance, so it is an upper bound on visible output. Model ids were checked
against `GET https://openrouter.ai/api/v1/models` on 2026-09-28; the snapshot taken at bench time is stored as
`price_snapshot.json` in the run.

Measurement hardware: the local runs are measured on a Mac Studio (2025, Apple M4 Max 16-core CPU / 40-core GPU,
128 GB, 2 TB). The development laptop is not used for local-model measurements.

## Decision

Local models (MLX, 4-bit, Apache-2.0, pinned revisions):

| id | repo | revision | size |
|---|---|---|---|
| `local-qwen3-4b` | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b` | 2.28 GB |
| `local-qwen2.5-1.5b` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | `8b403126fc14f14cfc99bb4cfa72ecbc129ea677` | 0.88 GB |

Cloud models (OpenRouter):

| role | model | request settings | dry-run estimate |
|---|---|---|---|
| frontier | `x-ai/grok-4.7` | `reasoning: {effort: minimal}`, `reasoning_allowance_tokens: 128` | $4.12 |
| small-closed | `openai/gpt-5.4-nano` | `reasoning: {effort: none}` | $0.61 |
| open-large | `deepseek/deepseek-v3.2` | `provider: {order: [gmicloud, novita], allow_fallbacks: false, quantizations: [fp8]}` | $0.43 |
| open-same-family | `qwen/qwen3-235b-a22b-2507` (non-thinking instruct) | same provider pin | $0.20 |

Total dry-run estimate: $5.36.

Judge candidates: `openai/gpt-4.1-mini` and `google/gemini-3.5-flash-lite`. Neither is benchmarked;
`local-enough judge calibrate` keeps the one with the higher balanced accuracy on judge-calib.

Why these:

- frontier: the current flagships of the major closed vendors are above the $4.50 cap on the dry-run formula:
  `openai/gpt-6-astra` $26.53, `anthropic/claude-opus-5.5` $10.61, `google/gemini-3.1-pro-preview` $5.95, and
  `openai/gpt-6-sol` (below Astra in OpenAI's line-up) $5.31. `x-ai/grok-4.7`, xAI's flagship, is the one that fits.
  Its endpoint rejects `reasoning: {enabled: false}` ("Reasoning is mandatory for this endpoint"), so it runs at the
  lowest effort. A 5-item preflight showed 60 to 550 reasoning tokens per call and that xAI does not count them
  against `max_tokens`, so visible output is never cut. Reasoning tokens are billed and recorded in the ledger.
- small-closed: `openai/gpt-5.4-mini` ($2.23) would put the four-model total at $6.98, above $6.50;
  `google/gemini-3.8-flash` has mandatory reasoning that consumed the 16-token label cap in the preflight
  (truncated labels) and needs an allowance that brings it to $2.70; `anthropic/claude-haiku-4.5` is $2.65.
  `openai/gpt-5.4-nano` runs with reasoning off. Its endpoints do not support `temperature`, so OpenRouter drops
  the `temperature: 0` we send; the report records this.
- open-large and open-same-family: several upstream providers serve these models with different prices and
  quantisations (fp4, fp8, unknown), so requests are pinned to fp8 providers with fallbacks disabled. The upstream
  provider of every call is recorded.
- Closed models are served by their own vendor only; the dated snapshot prices apply. Measured cost is
  `usage.cost` as reported by OpenRouter, which includes automatic prompt-cache discounts.

## Consequences

- Swapping any of these ids after the full run means rerunning every affected pass, and the README says so.
- The frontier role is a flagship with minimal but non-zero reasoning; its latency and cost include reasoning.
- If OpenRouter changes a price, the committed snapshot still reproduces the reported numbers offline.

## Outcome (recorded after the run)

- No id was swapped after the full run.
- Measured cost of Pass A (both splits): frontier $4.35, small-closed $0.25, open-large $0.14, open-same-family $0.07,
  $4.80 in total. The frontier model spent 475,967 reasoning tokens (321 per call on average, above the 128-token
  allowance), so its actual cost exceeded its $4.12 dry-run estimate; the four-model total stayed below $6.50.
- open-same-family needed 125 retries (HTTP 504 from the pinned providers) and one call still failed after three
  retries; it is scored as a wrong answer.
- The judge calibration kept `openai/gpt-4.1-mini` (balanced accuracy 0.953 on judge-calib against 0.952 for
  `google/gemini-3.5-flash-lite`; 0.932 on judge-holdout).
