# syntax=docker/dockerfile:1.7
# ============================================
# Backend Containerfile - Multi-stage Build
# FastAPI + uvicorn — NO ML dependencies
# ML worker runs on Vast.ai serverless with GPU
# ============================================

FROM docker.io/library/python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------
# Build dependencies
# ---------------------------------------------------------
FROM base AS deps

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

COPY backend/pyproject.toml backend/uv.lock* ./backend/
WORKDIR /app/backend

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv \
    && uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------
# Runtime (no build tools)
# ---------------------------------------------------------
FROM docker.io/library/python:3.11-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Only runtime shared lib for asyncpg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home --shell /usr/sbin/nologin appuser

COPY --from=deps --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser backend/ .

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=2)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
