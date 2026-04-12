"""Health check and file serving routes (R2 streaming proxy)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.app.storage import object_exists, stream_object

router = APIRouter(tags=["misc"])

# Content-type mapping by extension
_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".npy": "application/octet-stream",
    ".csv": "text/csv",
}


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/outputs/{key:path}")
async def serve_output(key: str):
    """Stream file from R2 as a proxy (frontend never talks to R2 directly)."""
    if not object_exists(key):
        raise HTTPException(status_code=404, detail="File not found")

    body, length, ctype = stream_object(key)
    # Prefer extension-based content type over what S3 reports
    ext = Path(key).suffix.lower()
    if ext in _CONTENT_TYPES:
        ctype = _CONTENT_TYPES[ext]

    return StreamingResponse(
        content=body.iter_chunks(chunk_size=8192),
        media_type=ctype,
        headers={"Content-Length": str(length)},
    )
