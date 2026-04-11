"""Session and SessionMetric ORM models."""

from __future__ import annotations

import uuid
from datetime import datetime  # noqa: TC003

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin


class Session(TimestampMixin, Base):
    """Analysis session for a single video upload."""

    __tablename__ = "sessions"

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
    element_type: Mapped[str] = mapped_column(String(50), index=True)
    video_url: Mapped[str | None] = mapped_column(String(500))
    processed_video_url: Mapped[str | None] = mapped_column(String(500))
    poses_url: Mapped[str | None] = mapped_column(String(500))
    csv_url: Mapped[str | None] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(20), default="uploading")
    error_message: Mapped[str | None] = mapped_column(Text)
    phases: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    recommendations: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    overall_score: Mapped[float | None] = mapped_column(Float)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationship to metrics
    metrics: Mapped[list[SessionMetric]] = relationship(
        "SessionMetric",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_sessions_user_element_created", "user_id", "element_type", "created_at"),
    )


class SessionMetric(TimestampMixin, Base):
    """Individual biomechanics metric for a session."""

    __tablename__ = "session_metrics"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        index=True,
    )
    metric_name: Mapped[str] = mapped_column(String(100), index=True)
    metric_value: Mapped[float] = mapped_column(Float)
    is_pr: Mapped[bool] = mapped_column(default=False)
    prev_best: Mapped[float | None] = mapped_column(Float)
    reference_value: Mapped[float | None] = mapped_column(Float)
    is_in_range: Mapped[bool | None] = mapped_column()

    # Relationship back to session
    session: Mapped[Session] = relationship(
        "Session",
        back_populates="metrics",
    )

    __table_args__ = (
        # Unique constraint on (session_id, metric_name)
        Index("uq_session_metric_name", "session_id", "metric_name", unique=True),
    )
