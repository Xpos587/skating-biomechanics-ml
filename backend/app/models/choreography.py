"""Music analysis and choreography program ORM models."""

from __future__ import annotations

import uuid

from sqlalchemy import JSON, Float, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class MusicAnalysis(TimestampMixin, Base):
    """Cached music analysis result (BPM, structure, energy curve)."""

    __tablename__ = "music_analyses"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(500))
    audio_url: Mapped[str] = mapped_column(String(500))
    duration_sec: Mapped[float] = mapped_column(Float)
    bpm: Mapped[float | None] = mapped_column(Float)
    meter: Mapped[str | None] = mapped_column(String(10))
    structure: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    energy_curve: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    downbeats: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    peaks: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    __table_args__ = (Index("ix_music_analyses_user_created", "user_id", "created_at"),)


class ChoreographyProgram(TimestampMixin, Base):
    """Saved choreography program with layout and scores."""

    __tablename__ = "choreography_programs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    music_analysis_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("music_analyses.id", ondelete="SET NULL"),
        nullable=True,
    )
    title: Mapped[str | None] = mapped_column(String(200))
    discipline: Mapped[str] = mapped_column(String(30))
    segment: Mapped[str] = mapped_column(String(20))
    season: Mapped[str] = mapped_column(String(10), default="2025_26")
    layout: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    total_tes: Mapped[float | None] = mapped_column(Float)
    estimated_goe: Mapped[float | None] = mapped_column(Float)
    estimated_pcs: Mapped[float | None] = mapped_column(Float)
    estimated_total: Mapped[float | None] = mapped_column(Float)
    is_valid: Mapped[bool | None] = mapped_column()
    validation_errors: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    validation_warnings: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (Index("ix_choreo_programs_user_created", "user_id", "created_at"),)
