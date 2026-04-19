# src/backend/routes/sessions.py
"""Session CRUD API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.auth.deps import CurrentUser, DbDep
from app.crud.connection import is_connected_as
from app.crud.session import count_by_user, create, get_by_id, list_by_user, soft_delete, update
from app.models.connection import ConnectionType
from app.schemas import (
    CreateSessionRequest,
    PatchSessionRequest,
    SessionListResponse,
    SessionResponse,
)
from app.storage import get_object_url_async

router = APIRouter(tags=["sessions"])


async def _session_to_response(session) -> SessionResponse:
    """Convert ORM Session to response schema with presigned URLs."""
    video_url = (
        await get_object_url_async(session.video_key) if session.video_key else session.video_url
    )
    processed_video_url = (
        await get_object_url_async(session.processed_video_key)
        if session.processed_video_key
        else session.processed_video_url
    )
    return SessionResponse.model_validate(
        {
            "id": session.id,
            "user_id": session.user_id,
            "element_type": session.element_type,
            "video_key": session.video_key,
            "video_url": video_url,
            "processed_video_key": session.processed_video_key,
            "processed_video_url": processed_video_url,
            "poses_url": session.poses_url,
            "csv_url": session.csv_url,
            "pose_data": session.pose_data,
            "frame_metrics": session.frame_metrics,
            "status": session.status,
            "error_message": session.error_message,
            "phases": session.phases,
            "recommendations": session.recommendations,
            "overall_score": session.overall_score,
            "created_at": session.created_at,
            "processed_at": session.processed_at,
            "metrics": session.metrics,
        }
    )


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(body: CreateSessionRequest, user: CurrentUser, db: DbDep):
    session = await create(
        db,
        user_id=user.id,
        element_type=body.element_type,
        video_key=body.video_key,
        status="queued" if body.video_key else "uploading",
    )
    return await _session_to_response(session)


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
    if (
        user_id
        and user_id != user.id
        and not await is_connected_as(
            db, from_user_id=user.id, to_user_id=user_id, connection_type=ConnectionType.COACHING
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not a coach for this user"
        )

    sessions = await list_by_user(
        db,
        user_id=target_user_id,
        element_type=element_type,
        limit=limit,
        offset=offset,
        sort=sort,
    )
    total = await count_by_user(db, user_id=target_user_id, element_type=element_type)
    return SessionListResponse(
        sessions=[await _session_to_response(s) for s in sessions], total=total
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, user: CurrentUser, db: DbDep):
    session = await get_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.user_id != user.id and not await is_connected_as(
        db,
        from_user_id=user.id,
        to_user_id=session.user_id,
        connection_type=ConnectionType.COACHING,
    ):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return await _session_to_response(session)


@router.patch("/sessions/{session_id}", response_model=SessionResponse)
async def patch_session(
    session_id: str,
    body: PatchSessionRequest,
    user: CurrentUser,
    db: DbDep,
):
    session = await get_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    session = await update(db, session, **body.model_dump(exclude_unset=True))
    return await _session_to_response(session)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, user: CurrentUser, db: DbDep):
    session = await get_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    await soft_delete(db, session)
