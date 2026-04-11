# infra/CLAUDE.md — Infrastructure & Deployment

## Architecture

```
Caddy (:3000)
  ├── /api/* → FastAPI (:8000)
  ├── WebSocket → Next.js dev (:5173)
  └── /* → frontend/dist (static files)
```

## Files

| File | Purpose |
|------|---------|
| `Containerfile` | Multi-stage build: Python 3.11 + uv + bun, builds frontend, runs uvicorn |
| `Caddyfile` | Reverse proxy: API routing, static files, WebSocket for HMR |
| `compose.yaml` | Local dev services: Valkey (task queue) + PostgreSQL (database) |
| `.containerignore` | Docker build exclusion rules |

## Local Development Services

```bash
podman compose up -d    # Start Valkey + PostgreSQL
podman compose down     # Stop services
```

**Valkey**: `localhost:6379` — arq task queue
**PostgreSQL**: `localhost:5432` — SQLAlchemy async (db: `skating_ml`, user: `skating`)

Defaults in `compose.yaml` use env vars with `:-` fallbacks (`VALKEY_HOST_PORT`, `POSTGRES_DB`, etc.).

## Container Build

```bash
podman build -t skating-ml -f infra/Containerfile .
podman run -p 8000:8000 --env-file .env skating-ml
```

Build copies `backend/`, `ml/`, `data/`, builds frontend from `frontend/`. Does **not** include `docs/`, `experiments/`, or `infra/`.

## GPU Worker (Vast.ai)

Separate container in `ml/gpu_server/Containerfile` — multi-stage, 4.9GB, no torch/timm/triton.
Image: `ghcr.io/xpos587/skating-ml-gpu:latest`

## Environment Variables

See `backend/app/config.py` for full list. Key ones:
- `DATABASE_URL` — PostgreSQL connection string
- `VALKEY_URL` — Valkey/Redis connection string
- `R2_ENDPOINT`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`, `R2_BUCKET` — Cloudflare R2
- `VASTAI_API_KEY` — enables remote GPU dispatch
- `JWT_SECRET` — JWT signing key
