"""POST /api/detect — enqueue person detection job."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.schemas import (
    DetectQueueResponse,
    DetectResultResponse,
    TaskStatusResponse,
)
from app.storage import upload_bytes
from app.task_manager import (
    TaskStatus,
    create_task_state,
    get_task_state,
    get_valkey_client,
)

router = APIRouter()


@router.post("/detect", response_model=DetectQueueResponse)
async def enqueue_detect(
    request: Request,
    video: UploadFile,
    tracking: str = "auto",
) -> DetectQueueResponse:
    """Upload video, enqueue detection job, return task_id immediately."""
    suffix = Path(video.filename or "video.mp4").suffix
    video_key = f"input/{uuid.uuid4().hex}{suffix}"

    content = await video.read()
    upload_bytes(content, video_key)

    task_id = f"det_{uuid.uuid4().hex[:12]}"

    valkey = await get_valkey_client()
    try:
        await create_task_state(task_id, video_key=video_key, valkey=valkey)
    finally:
        await valkey.close()

    await request.app.state.arq_pool.enqueue_job(
        "detect_video_task",
        task_id=task_id,
        video_key=video_key,
        tracking=tracking,
        _priority=0,  # High priority for fast preview
    )

    return DetectQueueResponse(task_id=task_id, video_key=video_key)


@router.get("/detect/{task_id}/status", response_model=TaskStatusResponse)
async def get_detect_status(task_id: str):
    """Poll detection task status."""
    valkey = await get_valkey_client()
    try:
        state = await get_task_state(task_id, valkey=valkey)
    finally:
        await valkey.close()

    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")

    result = None
    if state.get("result"):
        result = DetectResultResponse(**state["result"])

    return TaskStatusResponse(
        task_id=task_id,
        status=state["status"],
        progress=state["progress"],
        message=state.get("message", ""),
        result=result,  # type: ignore[reportArgumentType]
        error=state.get("error"),
    )


@router.get("/detect/{task_id}/result", response_model=DetectResultResponse)
async def get_detect_result(task_id: str):
    """Get detection result (persons, preview)."""
    valkey = await get_valkey_client()
    try:
        state = await get_task_state(task_id, valkey=valkey)
    finally:
        await valkey.close()

    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if state.get("status") != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task not completed yet")

    if not state.get("result"):
        raise HTTPException(status_code=500, detail="No result stored")

    return DetectResultResponse(**state["result"])
