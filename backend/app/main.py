"""FastAPI application for the figure skating biomechanics web UI."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import get_settings
from backend.app.logging_config import configure_logging
from backend.app.routes import (
    auth,
    detect,
    metrics,
    misc,
    models,
    process,
    relationships,
    sessions,
    uploads,
    users,
)

configure_logging()
logger = structlog.get_logger()

app = FastAPI(title="AI Тренер — Фигурное катание")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(auth.router, prefix="/auth")
api_v1.include_router(users.router, prefix="/users")
api_v1.include_router(detect.router)
api_v1.include_router(models.router)
api_v1.include_router(process.router)
api_v1.include_router(misc.router)
api_v1.include_router(sessions.router)
api_v1.include_router(metrics.router)
api_v1.include_router(relationships.router)
api_v1.include_router(uploads.router)
app.include_router(api_v1)
