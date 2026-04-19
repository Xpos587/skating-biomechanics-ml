"""Connection ORM model for user-to-user relationships."""

from __future__ import annotations

import uuid
from datetime import datetime  # noqa: TC003
from enum import StrEnum

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin
from app.models.user import User


class ConnectionType(StrEnum):
    """Type of connection between users."""

    COACHING = "coaching"
    CHOREOGRAPHY = "choreography"


class ConnectionStatus(StrEnum):
    """Lifecycle status of a connection."""

    INVITED = "invited"
    ACTIVE = "active"
    ENDED = "ended"


class Connection(TimestampMixin, Base):
    """User-to-user connection with lifecycle management."""

    __tablename__ = "connections"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    from_user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
    )
    to_user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
    )
    connection_type: Mapped[ConnectionType] = mapped_column(
        SAEnum(ConnectionType),
        default=ConnectionType.COACHING,
    )
    status: Mapped[ConnectionStatus] = mapped_column(
        SAEnum(ConnectionStatus),
        default=ConnectionStatus.INVITED,
    )
    initiated_by: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    from_user: Mapped[User] = relationship(  # type: ignore[valid-type]
        "User", foreign_keys=[from_user_id], lazy="selectin"
    )
    to_user: Mapped[User] = relationship(  # type: ignore[valid-type]
        "User", foreign_keys=[to_user_id], lazy="selectin"
    )
