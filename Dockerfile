# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.5.21-python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock* README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then uv sync --frozen --no-dev --no-install-project; \
    else uv sync --no-dev --no-install-project; fi

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then uv sync --frozen --no-dev; else uv sync --no-dev; fi

FROM python:3.11.11-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EDGESIGHT_CONFIG=/app/configs/production.yaml
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 edgesight \
    && useradd --uid 10001 --gid edgesight --create-home --shell /usr/sbin/nologin edgesight

WORKDIR /app
COPY --from=builder --chown=edgesight:edgesight /app/.venv /app/.venv
COPY --chown=edgesight:edgesight configs ./configs
COPY --chown=edgesight:edgesight docker/entrypoint.sh /usr/local/bin/entrypoint
RUN chmod 0555 /usr/local/bin/entrypoint && mkdir -p /app/models /app/results \
    && chown -R edgesight:edgesight /app

USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl --fail --silent http://127.0.0.1:8080/health/live || exit 1
ENTRYPOINT ["entrypoint"]
CMD ["uvicorn", "edgesight.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
