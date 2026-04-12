# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install bun
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:$PATH"

WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml uv.lock* ./

# Install Python deps
RUN uv sync --frozen 2>/dev/null || uv sync

# Copy frontend deps
COPY frontend/package.json frontend/bun.lock* frontend/
RUN cd frontend && bun install --frozen-lockfile 2>/dev/null || bun install

# Copy source
COPY backend/ backend/
COPY ml/ ml/
COPY ml/scripts/ ml/scripts/
COPY data/ data/

# Build frontend
RUN cd frontend && bun run build

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
