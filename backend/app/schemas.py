"""Pydantic schemas for the web API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field, field_validator

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = Field(default=None, max_length=100)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None
    avatar_url: str | None
    bio: str | None
    height_cm: int | None
    weight_kg: float | None
    language: str
    timezone: str
    theme: str
    is_active: bool
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def validate_datetime(cls, v: Any) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)


class UpdateProfileRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=100)
    bio: str | None = None
    height_cm: int | None = Field(default=None, ge=50, le=250)
    weight_kg: float | None = Field(default=None, ge=20, le=300)


class UpdateSettingsRequest(BaseModel):
    language: str | None = Field(default=None, max_length=10)
    timezone: str | None = Field(default=None, max_length=50)
    theme: str | None = Field(default=None, pattern=r"^(light|dark|system)$")


# ---------------------------------------------------------------------------
# Detect & Process
# ---------------------------------------------------------------------------


class PersonInfo(BaseModel):
    track_id: int
    hits: int
    bbox: list[float]
    mid_hip: list[float]


class PersonClick(BaseModel):
    x: int
    y: int


class DetectResponse(BaseModel):
    persons: list[PersonInfo]
    preview_image: str
    video_key: str
    auto_click: PersonClick | None = None
    status: str


@dataclass
class MLModelFlags:
    """ML model feature flags for video processing."""

    depth: bool = False
    optical_flow: bool = False
    segment: bool = False
    foot_track: bool = False
    matting: bool = False
    inpainting: bool = False


class DetectQueueResponse(BaseModel):
    task_id: str
    video_key: str
    status: str = "pending"


class DetectResultResponse(BaseModel):
    persons: list[PersonInfo]
    preview_image: str
    video_key: str
    auto_click: PersonClick | None = None
    status: str


class ProcessRequest(BaseModel):
    video_key: str
    person_click: PersonClick
    frame_skip: int = 1
    layer: int = 3
    tracking: str = "auto"
    export: bool = True
    depth: bool = False
    optical_flow: bool = False
    segment: bool = False
    foot_track: bool = False
    matting: bool = False
    inpainting: bool = False


class ProcessStats(BaseModel):
    total_frames: int
    valid_frames: int
    fps: float
    resolution: str


class ProcessResponse(BaseModel):
    video_path: str
    poses_path: str | None
    csv_path: str | None
    stats: ProcessStats
    status: str


class QueueProcessResponse(BaseModel):
    task_id: str
    status: str = "pending"


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    result: ProcessResponse | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    element_type: str = Field(..., min_length=1, max_length=50)


class PatchSessionRequest(BaseModel):
    element_type: str | None = Field(default=None, max_length=50)


class SessionMetricResponse(BaseModel):
    id: str
    metric_name: str
    metric_value: float
    is_pr: bool
    prev_best: float | None
    reference_value: float | None
    is_in_range: bool | None

    model_config = {"from_attributes": True}


class SessionResponse(BaseModel):
    id: str
    user_id: str
    element_type: str
    video_url: str | None
    processed_video_url: str | None
    poses_url: str | None
    csv_url: str | None
    status: str
    error_message: str | None
    phases: dict | None
    recommendations: list[str] | None
    overall_score: float | None
    created_at: str
    processed_at: str | None
    metrics: list[SessionMetricResponse] = []

    model_config = {"from_attributes": True}

    @field_validator("created_at", "processed_at", mode="before")
    @classmethod
    def validate_datetime(cls, v: Any) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]
    total: int


# ---------------------------------------------------------------------------
# Metrics & Progress
# ---------------------------------------------------------------------------


class TrendDataPoint(BaseModel):
    date: str
    value: float
    session_id: str
    is_pr: bool


class TrendResponse(BaseModel):
    metric_name: str
    element_type: str
    data_points: list[TrendDataPoint]
    trend: str  # improving | stable | declining
    current_pr: float | None
    reference_range: dict[str, float] | None


class DiagnosticsFinding(BaseModel):
    severity: str
    element: str
    metric: str
    message: str
    detail: str


class DiagnosticsResponse(BaseModel):
    user_id: str
    findings: list[DiagnosticsFinding]


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


class InviteRequest(BaseModel):
    skater_email: str


class RelationshipResponse(BaseModel):
    id: str
    coach_id: str
    skater_id: str
    status: str
    initiated_by: str | None
    created_at: str
    ended_at: str | None
    coach_name: str | None = None
    skater_name: str | None = None

    model_config = {"from_attributes": True}

    @field_validator("created_at", "ended_at", mode="before")
    @classmethod
    def validate_datetime(cls, v: Any) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)


class RelationshipListResponse(BaseModel):
    relationships: list[RelationshipResponse]
