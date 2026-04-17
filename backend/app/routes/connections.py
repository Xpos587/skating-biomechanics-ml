"""Flexible connection API routes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, status

from app.crud.connection import (
    create as create_conn,
    get_active as get_active_conn,
    get_by_id as get_conn_by_id,
    list_for_user,
    list_pending_for_user,
)
from app.crud.user import get_by_email
from app.models.connection import ConnectionStatus, ConnectionType
from app.schemas import ConnectionListResponse, ConnectionResponse, InviteRequest

if TYPE_CHECKING:
    from app.auth.deps import CurrentUser, DbDep
    from app.models.connection import Connection


router = APIRouter(tags=["connections"])


def _conn_to_response(conn: Connection) -> ConnectionResponse:
    return ConnectionResponse.model_validate(conn)


@router.post("/connections/invite", response_model=ConnectionResponse, status_code=status.HTTP_201_CREATED)
async def invite(body: InviteRequest, user: CurrentUser, db: DbDep):
    """User invites another user to a connection."""
    to_user = await get_by_email(db, body.to_user_email)
    if not to_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    conn_type = ConnectionType(body.connection_type)
    existing = await get_active_conn(db, from_user_id=user.id, to_user_id=to_user.id, connection_type=conn_type)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Connection already exists")

    conn = await create_conn(db, from_user_id=user.id, to_user_id=to_user.id, connection_type=conn_type, initiated_by=user.id)
    return _conn_to_response(conn)


@router.post("/connections/{conn_id}/accept", response_model=ConnectionResponse)
async def accept_invite(conn_id: str, user: CurrentUser, db: DbDep):
    """Invitee accepts a connection."""
    conn = await get_conn_by_id(db, conn_id)
    if not conn:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found")
    if conn.to_user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    if conn.status != ConnectionStatus.INVITED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not an active invite")
    conn.status = ConnectionStatus.ACTIVE
    db.add(conn)
    await db.flush()
    await db.refresh(conn)
    return _conn_to_response(conn)


@router.post("/connections/{conn_id}/end", response_model=ConnectionResponse)
async def end_connection(conn_id: str, user: CurrentUser, db: DbDep):
    """Either party ends the connection."""
    conn = await get_conn_by_id(db, conn_id)
    if not conn:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found")
    if user.id not in (conn.from_user_id, conn.to_user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    if conn.status == ConnectionStatus.ENDED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Already ended")
    conn.status = ConnectionStatus.ENDED
    conn.ended_at = datetime.now(UTC)
    db.add(conn)
    await db.flush()
    await db.refresh(conn)
    return _conn_to_response(conn)


@router.get("/connections", response_model=ConnectionListResponse)
async def list_connections(user: CurrentUser, db: DbDep):
    """List all connections for the current user."""
    conns = await list_for_user(db, user.id)
    return ConnectionListResponse(connections=[_conn_to_response(c) for c in conns])


@router.get("/connections/pending", response_model=ConnectionListResponse)
async def list_pending(user: CurrentUser, db: DbDep):
    """List pending invites received by the current user."""
    conns = await list_pending_for_user(db, user.id)
    return ConnectionListResponse(connections=[_conn_to_response(c) for c in conns])
