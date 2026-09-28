# ADR 3: MLX as the measured runtime

- Status: accepted
- Date: 2026-09-28

## Context

The measured hardware is a fanless MacBook Air (Apple M5, 24 GB). Local runtimes on Apple Silicon include MLX
(`mlx_lm.server`), llama.cpp (`llama-server`), Ollama and LM Studio. Measuring several runtimes multiplies the
measurement windows on a machine that throttles, and v0.1 has one hardware profile.

## Decision

`mlx_lm.server` is the one measured runtime. local-enough starts it itself as `python -m mlx_lm.server` bound to
127.0.0.1 with `HF_HUB_OFFLINE=1`, passing the pinned snapshot directory as `--model` because the server has no
revision flag, and sends `model: "default_model"` because any other value makes the server load a different model.
The server batches concurrent requests, which is why Pass B (concurrency 4) measures a different throughput from
Pass A (concurrency 1). Other runtimes are supported through `kind: local, base_url: ...` (any OpenAI-compatible
server) and tested with a fake server, but not measured.

## Consequences

- Local numbers describe MLX on one machine; llama.cpp or Ollama on the same machine may differ.
- The server lists its `--model` value, an absolute cache path, as the model id in `GET /v1/models`, and echoes
  the request's `model` value in responses. Every `model` value is rewritten to `<repo>@<sha>` before it is written
  or returned, and `/v1/models` output of the model server is never stored.
- MLX needs Metal, so it cannot run inside Docker on macOS; the router container reaches it on the host.
