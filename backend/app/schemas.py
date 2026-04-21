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
    session_id: str | None = None
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
    video_key: str | None = Field(default=None, max_length=500)


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


# Pose and metrics data types (Task 10, 2026-04-16)


class PoseData(BaseModel):
    """Sampled pose data for frontend visualization.

    Frames are sampled (e.g., every 10th frame) to reduce data transfer.
    poses shape: (N_sampled, 17, 3) where 17 = H3.6M keypoints, 3 = (x, y, conf)
    """

    frames: list[int]  # Sampled frame indices
    poses: list[list[list[float]]]  # [frame][keypoint][x,y,conf]
    fps: float  # Video frame rate


class FrameMetrics(BaseModel):
    """Frame-by-frame biomechanics metrics.

    All arrays are aligned with the frames list in PoseData.
    null values indicate metric could not be computed for that frame.
    """

    knee_angles_r: list[float | None]
    knee_angles_l: list[float | None]
    hip_angles_r: list[float | None]
    hip_angles_l: list[float | None]
    trunk_lean: list[float | None]
    com_height: list[float | None]


class PhasesData(BaseModel):
    """Phase markers for element segmentation.

    Frame indices are relative to the original video, not sampled frames.
    """

    takeoff: int | None = None
    peak: int | None = None
    landing: int | None = None


class SessionResponse(BaseModel):
    id: str
    user_id: str
    element_type: str
    video_key: str | None = None
    video_url: str | None
    processed_video_key: str | None = None
    processed_video_url: str | None
    poses_url: str | None  # Deprecated: Replaced by pose_data
    csv_url: str | None  # Deprecated: Replaced by frame_metrics
    pose_data: PoseData | None  # New: Typed pose data storage (JSON)
    frame_metrics: FrameMetrics | None  # New: Typed frame metrics (JSON)
    status: str
    error_message: str | None
    phases: PhasesData | None  # Typed phase markers
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
# Connections
# ---------------------------------------------------------------------------


class InviteRequest(BaseModel):
    to_user_email: str
    connection_type: str = Field(pattern=r"^(coaching|choreography)$")


class ConnectionResponse(BaseModel):
    id: str
    from_user_id: str
    to_user_id: str
    connection_type: str
    status: str
    initiated_by: str | None
    created_at: str
    ended_at: str | None
    from_user_name: str | None = None
    to_user_name: str | None = None

    model_config = {"from_attributes": True}

    @field_validator("created_at", "ended_at", mode="before")
    @classmethod
    def validate_datetime(cls, v):
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)


class ConnectionListResponse(BaseModel):
    connections: list[ConnectionResponse]


# ---------------------------------------------------------------------------
# Choreography
# ---------------------------------------------------------------------------


class MusicAnalysisResponse(BaseModel):
    id: str
    user_id: str
    filename: str
    audio_url: str
    duration_sec: float
    bpm: float | None
    meter: str | None
    structure: list[dict] | None
    energy_curve: dict | None
    downbeats: list[float] | None
    peaks: list[float] | None
    status: str
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def validate_datetime(cls, v: Any) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)


class UploadMusicResponse(BaseModel):
    music_id: str
    filename: str


class GenerateRequest(BaseModel):
    music_id: str
    discipline: str = Field(pattern=r"^(mens_singles|womens_singles)$")
    segment: str = Field(pattern=r"^(short_program|free_skate)$")
    inventory: dict


class LayoutElement(BaseModel):
    code: str
    goe: int = 0
    timestamp: float = 0.0
    position: dict | None = None
    is_back_half: bool = False
    is_jump_pass: bool = False
    jump_pass_index: int | None = None


class Layout(BaseModel):
    elements: list[LayoutElement]
    total_tes: float
    back_half_indices: list[int]


class GenerateResponse(BaseModel):
    layouts: list[Layout]


class ValidateRequest(BaseModel):
    discipline: str = Field(pattern=r"^(mens_singles|womens_singles)$")
    segment: str = Field(pattern=r"^(short_program|free_skate)$")
    elements: list[dict]


class ValidateResponse(BaseModel):
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    total_tes: float | None = None


class RenderRinkRequest(BaseModel):
    elements: list[dict]
    width: int = Field(default=1200, ge=400, le=4000)
    height: int = Field(default=600, ge=200, le=2000)
    rink_width: float = Field(default=60.0, ge=20.0, le=80.0)
    rink_height: float = Field(default=30.0, ge=10.0, le=40.0)


class ChoreographyProgramResponse(BaseModel):
    id: str
    user_id: str
    music_analysis_id: str | None
    title: str | None
    discipline: str
    segment: str
    season: str
    layout: dict | None
    total_tes: float | None
    estimated_goe: float | None
    estimated_pcs: float | None
    estimated_total: float | None
    is_valid: bool | None
    validation_errors: list[str] | None
    validation_warnings: list[str] | None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def validate_datetime(cls, v: Any) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)


class ProgramListResponse(BaseModel):
    programs: list[ChoreographyProgramResponse]
    total: int


class SaveProgramRequest(BaseModel):
    music_analysis_id: str | None = None
    discipline: str | None = None
    segment: str | None = None
    title: str | None = None
    layout: dict | None = None
    total_tes: float | None = None
    estimated_goe: float | None = None
    estimated_pcs: float | None = None
    estimated_total: float | None = None
    is_valid: bool | None = None
    validation_errors: list[str] | None = None
    validation_warnings: list[str] | None = None


class ExportRequest(BaseModel):
    format: str = Field(pattern=r"^(svg|pdf|json)$")
