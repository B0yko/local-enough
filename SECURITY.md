# Security policy

## The router has no authentication

`local-enough route` starts a plain HTTP server with no auth, no rate limiting and no multi-tenancy.
Anyone who can reach the port can call it and spend your configured
cloud budget or load your local models.

- By default it binds to `127.0.0.1`: only processes on the same machine can reach it.
- The Docker image binds to `0.0.0.0` inside its container, because a containerised router needs to accept
  connections proxied from outside the container. **Put it behind your own network controls** (a reverse proxy
  with auth, a firewall rule, a private network, or a VPN) before exposing it beyond `localhost` — do not run
  the container's published port directly on a public interface.
- If you need auth, rate limiting or multi-tenancy today, put a gateway with those features in front of
  `local-enough route`; it speaks plain OpenAI-compatible HTTP, so any such gateway can proxy it.

## Secrets

API keys are read only from environment variables (`OPENROUTER_API_KEY` by default; see
`docs/configuration.md` for how a model's `api_key_env` can point elsewhere). Never put a key into a committed
`config.yaml`, `route.yaml` or `.env` file — `.env.example` ships with empty values and `.env` is gitignored.

## Logging

Request and response bodies are never logged, by the router or by `bench`. `GET /stats` exposes only counts,
latency and spend, never the text of a request or response; `tests/` covers this so it cannot regress silently.

## Reporting a vulnerability

Please report security issues through
[GitHub private security advisories](https://github.com/B0yko/local-enough/security/advisories/new) for this
repository rather than a public issue. Include what you found, how to reproduce it and, if you have one, an
assessment of impact. We'll acknowledge the report and follow up with a fix or a mitigation plan.
