"""Connection CRUD operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from app.models.connection import Connection, ConnectionStatus, ConnectionType

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def create(
    db: AsyncSession,
    *,
    from_user_id: str,
    to_user_id: str,
    connection_type: ConnectionType,
    initiated_by: str,
) -> Connection:
    """Create a new connection (default status: INVITED)."""
    conn = Connection(
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        connection_type=connection_type,
        initiated_by=initiated_by,
        status=ConnectionStatus.INVITED,
    )
    db.add(conn)
    await db.flush()
    await db.refresh(conn)
    return conn


async def get_by_id(db: AsyncSession, conn_id: str) -> Connection | None:
    """Get a connection by its ID."""
    result = await db.execute(select(Connection).where(Connection.id == conn_id))
    return result.scalar_one_or_none()


async def get_active(
    db: AsyncSession,
    *,
    from_user_id: str,
    to_user_id: str,
    connection_type: ConnectionType,
) -> Connection | None:
    """Get a non-ended connection between two users of a given type."""
    result = await db.execute(
        select(Connection).where(
            Connection.from_user_id == from_user_id,
            Connection.to_user_id == to_user_id,
            Connection.connection_type == connection_type,
            Connection.status != ConnectionStatus.ENDED,
        )
    )
    return result.scalar_one_or_none()


async def list_for_user(db: AsyncSession, user_id: str) -> list[Connection]:
    """List all connections where user is either party, newest first."""
    result = await db.execute(
        select(Connection)
        .where(
            (Connection.from_user_id == user_id) | (Connection.to_user_id == user_id),
        )
        .order_by(Connection.created_at.desc())
    )
    return list(result.scalars().all())


async def list_pending_for_user(db: AsyncSession, user_id: str) -> list[Connection]:
    """List connections where user is the recipient and status is INVITED."""
    result = await db.execute(
        select(Connection)
        .where(
            Connection.to_user_id == user_id,
            Connection.status == ConnectionStatus.INVITED,
        )
        .order_by(Connection.created_at.desc())
    )
    return list(result.scalars().all())


async def is_connected_as(
    db: AsyncSession,
    *,
    from_user_id: str,
    to_user_id: str,
    connection_type: ConnectionType,
) -> bool:
    """Check if an ACTIVE connection of the given type exists.

    Only ACTIVE connections grant access (INVITED does not).
    """
    result = await db.execute(
        select(Connection).where(
            Connection.from_user_id == from_user_id,
            Connection.to_user_id == to_user_id,
            Connection.connection_type == connection_type,
            Connection.status == ConnectionStatus.ACTIVE,
        )
    )
    return result.scalar_one_or_none() is not None
