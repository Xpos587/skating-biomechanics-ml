"""FastAPI application for the figure skating biomechanics web UI."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.logging_config import configure_logging
from src.backend.routes import auth, detect, metrics, misc, models, process, sessions, users
from src.config import get_settings

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
app.include_router(api_v1)
