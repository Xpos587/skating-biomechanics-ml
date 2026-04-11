"""Health check and static file serving routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.config import get_settings

router = APIRouter(tags=["misc"])

OUTPUTS_DIR = Path(get_settings().outputs_dir)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/outputs/{filename:path}")
async def serve_output(filename: str):
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))
