"""FastAPI application for the figure skating biomechanics web UI."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.logging_config import configure_logging
from app.routes import (
    auth,
    choreography,
    connections,
    detect,
    metrics,
    misc,
    models,
    process,
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
api_v1.include_router(connections.router)
api_v1.include_router(uploads.router)
api_v1.include_router(choreography.router)
app.include_router(api_v1)
