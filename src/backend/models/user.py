"""User ORM model."""

from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Float, SmallInteger, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.backend.models.base import Base, TimestampMixin


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(100))
    avatar_url: Mapped[str | None] = mapped_column(String(500))
    bio: Mapped[str | None] = mapped_column(Text)

    # Body params (for PhysicsEngine: CoM, Dempster tables)
    height_cm: Mapped[int | None] = mapped_column(SmallInteger)
    weight_kg: Mapped[float | None] = mapped_column(Float)

    # Preferences
    language: Mapped[str] = mapped_column(String(10), default="ru")
    timezone: Mapped[str] = mapped_column(String(50), default="Europe/Moscow")
    theme: Mapped[str] = mapped_column(String(10), default="system")

    # Metadata
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
