"""POST /api/detect — enqueue person detection job."""

from __future__ import annotations

import uuid
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, HTTPException, UploadFile

from src.backend.schemas import (
    DetectQueueResponse,
    DetectResultResponse,
    PersonInfo,
    TaskStatusResponse,
)
from src.config import get_settings
from src.storage import upload_bytes
from src.task_manager import (
    TaskStatus,
    create_task_state,
    get_task_state,
    get_valkey_client,
)

router = APIRouter()


@router.post("/detect", response_model=DetectQueueResponse)
async def enqueue_detect(
    video: UploadFile,
    tracking: str = "auto",
) -> DetectQueueResponse:
    """Upload video, enqueue detection job, return task_id immediately."""
    suffix = Path(video.filename or "video.mp4").suffix
    video_key = f"input/{uuid.uuid4().hex}{suffix}"

    content = await video.read()
    upload_bytes(content, video_key)

    settings = get_settings()
    task_id = f"det_{uuid.uuid4().hex[:12]}"

    valkey = await get_valkey_client()
    try:
        await create_task_state(task_id, video_key=video_key, valkey=valkey)
    finally:
        await valkey.close()

    arq_pool = await create_pool(
        RedisSettings(
            host=settings.valkey.host,
            port=settings.valkey.port,
            database=settings.valkey.db,
            password=settings.valkey.password.get_secret_value(),
        )
    )
    try:
        await arq_pool.enqueue_job(
            "detect_video_task",
            task_id=task_id,
            video_key=video_key,
            tracking=tracking,
        )
    finally:
        await arq_pool.close()

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
        result=result,
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
