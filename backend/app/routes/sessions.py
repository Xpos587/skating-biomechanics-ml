# src/backend/routes/sessions.py
"""Session CRUD API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from backend.app.auth.deps import CurrentUser, DbDep
from backend.app.crud.relationship import is_coach_for_student
from backend.app.crud.session import create, get_by_id, list_by_user, soft_delete, update
from backend.app.schemas import (
    CreateSessionRequest,
    PatchSessionRequest,
    SessionListResponse,
    SessionResponse,
)

router = APIRouter(tags=["sessions"])


def _session_to_response(session) -> SessionResponse:
    """Convert ORM Session to response schema."""
    return SessionResponse.model_validate(session)


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(body: CreateSessionRequest, user: CurrentUser, db: DbDep):
    session = await create(db, user_id=user.id, element_type=body.element_type)
    return _session_to_response(session)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    user: CurrentUser,
    db: DbDep,
    user_id: str | None = None,
    element_type: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sort: str = Query("created_at", pattern="^(created_at|overall_score)$"),
):
    # Coaches can view their students' sessions
    target_user_id = user_id if user_id else user.id
    if user_id and user_id != user.id:
        if not await is_coach_for_student(db, coach_id=user.id, skater_id=user_id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a coach for this user")

    sessions = await list_by_user(
        db, user_id=target_user_id, element_type=element_type, limit=limit, offset=offset, sort=sort,
    )
    return SessionListResponse(sessions=[_session_to_response(s) for s in sessions], total=len(sessions))


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, user: CurrentUser, db: DbDep):
    session = await get_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.user_id != user.id:
        if not await is_coach_for_student(db, coach_id=user.id, skater_id=session.user_id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return _session_to_response(session)


@router.patch("/sessions/{session_id}", response_model=SessionResponse)
async def patch_session(
    session_id: str, body: PatchSessionRequest, user: CurrentUser, db: DbDep,
):
    session = await get_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    session = await update(db, session, **body.model_dump(exclude_unset=True))
    return _session_to_response(session)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, user: CurrentUser, db: DbDep):
    session = await get_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    await soft_delete(db, session)
