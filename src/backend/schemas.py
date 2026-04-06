"""Pydantic schemas for the web API."""

from __future__ import annotations

from pydantic import BaseModel


class PersonInfo(BaseModel):
    """Detected person in a video."""

    track_id: int
    hits: int
    bbox: list[float]  # [x1, y1, x2, y2] normalized
    mid_hip: list[float]  # [x, y] normalized


class PersonClick(BaseModel):
    """Pixel coordinate used to select a target person."""

    x: int
    y: int


class DetectResponse(BaseModel):
    """Response from POST /api/detect."""

    persons: list[PersonInfo]
    preview_image: str  # base64-encoded PNG
    video_path: str  # absolute path to uploaded video (for process endpoint)
    auto_click: PersonClick | None = None
    status: str


class ProcessRequest(BaseModel):
    """Request body for POST /api/process."""

    video_path: str
    person_click: PersonClick
    frame_skip: int = 1
    layer: int = 3
    tracking: str = "auto"
    export: bool = True


class ProcessStats(BaseModel):
    """Statistics from a completed analysis."""

    total_frames: int
    valid_frames: int
    fps: float
    resolution: str


class ProcessResponse(BaseModel):
    """Final response from POST /api/process (sent as last SSE event)."""

    video_path: str
    poses_path: str | None
    csv_path: str | None
    stats: ProcessStats
    status: str
