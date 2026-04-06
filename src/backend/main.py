"""FastAPI application for the figure skating biomechanics web UI."""

from __future__ import annotations

from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.logging_config import configure_logging
from src.backend.routes import detect, process

configure_logging()
logger = structlog.get_logger()

app = FastAPI(title="AI Тренер — Фигурное катание")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router)
app.include_router(process.router)

# Serve output files (videos, NPY, CSV)
OUTPUTS_DIR = Path("data/uploads")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/outputs/{filename:path}")
async def serve_output(filename: str):
    from fastapi.responses import FileResponse

    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))


@app.get("/api/health")
async def health():
    return {"status": "ok"}
