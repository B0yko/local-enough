# Router image: serves `local-enough route` on 0.0.0.0:8000. MLX needs Metal and cannot run in a container on
# macOS, so local models run on the host (mlx_lm.server) and the container reaches them via host.docker.internal.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 router
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH LOCAL_ENOUGH_HOME=/data PYTHONUNBUFFERED=1
RUN mkdir -p /data && chown router /data
USER router
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
ENTRYPOINT ["local-enough", "route", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--run", "reference", "--models", "/config/config.yaml"]
