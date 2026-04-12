# src/backend/routes/relationships.py
"""Coach-skater relationship API routes."""

from __future__ import annotations

from datetime import UTC

from fastapi import APIRouter, HTTPException, status

from backend.app.auth.deps import CurrentUser, DbDep
from backend.app.crud.relationship import (
    create as create_rel,
)
from backend.app.crud.relationship import (
    get_active as get_active_rel,
)
from backend.app.crud.relationship import (
    get_by_id as get_rel_by_id,
)
from backend.app.crud.relationship import (
    list_for_user,
    list_pending_for_skater,
)
from backend.app.crud.user import get_by_email
from backend.app.models.relationship import Relationship
from backend.app.schemas import InviteRequest, RelationshipListResponse, RelationshipResponse

router = APIRouter(tags=["relationships"])


def _rel_to_response(rel: Relationship) -> RelationshipResponse:
    return RelationshipResponse.model_validate(rel)


@router.post("/relationships/invite", response_model=RelationshipResponse, status_code=status.HTTP_201_CREATED)
async def invite(body: InviteRequest, user: CurrentUser, db: DbDep):
    """Coach invites a skater by email."""
    skater = await get_by_email(db, body.skater_email)
    if not skater:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    existing = await get_active_rel(db, coach_id=user.id, skater_id=skater.id)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Relationship already exists")

    rel = await create_rel(db, coach_id=user.id, skater_id=skater.id, initiated_by=user.id)
    return _rel_to_response(rel)


@router.post("/relationships/{rel_id}/accept", response_model=RelationshipResponse)
async def accept_invite(rel_id: str, user: CurrentUser, db: DbDep):
    """Skater accepts an invite."""
    rel = await get_rel_by_id(db, rel_id)
    if not rel:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Relationship not found")
    if rel.skater_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    if rel.status != "invited":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not an active invite")

    rel.status = "active"
    db.add(rel)
    await db.flush()
    await db.refresh(rel)
    return _rel_to_response(rel)


@router.post("/relationships/{rel_id}/end", response_model=RelationshipResponse)
async def end_relationship(rel_id: str, user: CurrentUser, db: DbDep):
    """Either party ends the relationship."""
    from datetime import datetime
    rel = await get_rel_by_id(db, rel_id)
    if not rel:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Relationship not found")
    if rel.coach_id != user.id and rel.skater_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    if rel.status == "ended":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Already ended")

    rel.status = "ended"
    rel.ended_at = datetime.now(UTC)
    db.add(rel)
    await db.flush()
    await db.refresh(rel)
    return _rel_to_response(rel)


@router.get("/relationships", response_model=RelationshipListResponse)
async def list_relationships(user: CurrentUser, db: DbDep):
    """List all relationships for the current user."""
    rels = await list_for_user(db, user.id)
    return RelationshipListResponse(relationships=[_rel_to_response(r) for r in rels])


@router.get("/relationships/pending", response_model=RelationshipListResponse)
async def list_pending(user: CurrentUser, db: DbDep):
    """List pending invites received by the current user (as skater)."""
    rels = await list_pending_for_skater(db, user.id)
    return RelationshipListResponse(relationships=[_rel_to_response(r) for r in rels])
