# Running the router in Docker

The image `ghcr.io/b0yko/local-enough` runs `local-enough route` on `0.0.0.0:8000`. MLX needs Metal and cannot
run inside a container on macOS, so the local models run on the host with `mlx_lm.server` and the container
reaches them through `host.docker.internal`. Cloud fallbacks need `OPENROUTER_API_KEY` in the container
environment.

## 1. Start the local models on the host

Use the same pinned snapshots the benchmark measured (`local-enough models pull --models config.yaml` downloads
them), one server per model:

```bash
REPO=mlx-community/Qwen3-4B-Instruct-2507-4bit
REV=50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b
SNAP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$REPO', revision='$REV', local_files_only=True))")
HF_HUB_OFFLINE=1 python -m mlx_lm.server --model "$SNAP" --host 127.0.0.1 --port 8081
```

Repeat for `mlx-community/Qwen2.5-1.5B-Instruct-4bit` at `8b403126fc14f14cfc99bb4cfa72ecbc129ea677` on port 8082.
Keep `--host 127.0.0.1` on macOS (see below).

## 2. Start the router

```bash
docker compose up
```

or, without compose:

```bash
docker run --rm -p 127.0.0.1:8000:8000 \
  -e OPENROUTER_API_KEY \
  -v "$PWD/examples/docker/config.yaml:/config/config.yaml:ro" \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/b0yko/local-enough:0.1.0
```

The default command serves the bundled reference plan (`--run reference`). To serve a plan from your own run,
mount the run directory and pass it: `... ghcr.io/b0yko/local-enough:0.1.0 --run /run --models
/config/config.yaml` with `-v "$PWD/runs/<stamp>:/run:ro"`.

`examples/docker/config.yaml` is the reference lineup with the two local models given as
`base_url: http://host.docker.internal:808x/v1` instead of `launch:`. With `base_url` the router does not know the
snapshot sha, so the response `model` field carries the model id from the config (`local-qwen3-4b`) instead of
`<repo>@<sha>`.

## Container-to-host networking, verified

Checked on 2026-09-28 on macOS with colima 0.10.3 (Docker engine in a Lima VM): inside the container
`host.docker.internal` resolves to the Lima host gateway (`192.168.5.2`), and an HTTP server on the Mac bound to
`127.0.0.1` answered from the container, as did one bound to `0.0.0.0`. A plain HTTP server stood in for
`mlx_lm.server`, since the question is only which bind address the container can reach. So on macOS with colima,
`mlx_lm.server --host 127.0.0.1` (its default) works and nothing is exposed on the LAN.

- **macOS, colima:** keep `--host 127.0.0.1`. `--add-host=host.docker.internal:host-gateway` is harmless and makes
  the name explicit.
- **Linux, Docker Engine:** `--add-host=host.docker.internal:host-gateway` maps the name to the Docker bridge
  gateway (usually `172.17.0.1`). A host server bound to `127.0.0.1` is **not** reachable from the bridge; it has to
  bind to the bridge address (`--host 172.17.0.1`) or to `0.0.0.0`. **Warning:** `--host 0.0.0.0` exposes the model
  server, which has no authentication, on every interface of the machine, including the LAN. Prefer the bridge
  address, or firewall the port.
- **Docker Desktop for Mac:** not verified here; Docker documents that `host.docker.internal` forwards to the host.

The router itself has no authentication either (see [SECURITY.md](../SECURITY.md)); the compose file and the
command above publish it on `127.0.0.1` only.
