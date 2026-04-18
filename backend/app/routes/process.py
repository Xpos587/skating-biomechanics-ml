"""POST /api/process/queue — enqueue video processing job."""

from __future__ import annotations

import uuid

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.schemas import (
    MLModelFlags,
    ProcessRequest,
    ProcessResponse,
    QueueProcessResponse,
    TaskStatusResponse,
)
from app.task_manager import (
    create_task_state,
    get_task_state,
    get_valkey_client,
    set_cancel_signal,
)

router = APIRouter()


@router.post("/process/queue", response_model=QueueProcessResponse)
async def enqueue_process(req: ProcessRequest):
    """Enqueue video processing job and return task_id immediately."""
    settings = get_settings()
    task_id = f"proc_{uuid.uuid4().hex[:12]}"

    valkey = await get_valkey_client()
    try:
        await create_task_state(task_id, video_key=req.video_key, valkey=valkey)
    finally:
        await valkey.close()

    ml_flags = MLModelFlags(
        depth=req.depth,
        optical_flow=req.optical_flow,
        segment=req.segment,
        foot_track=req.foot_track,
        matting=req.matting,
        inpainting=req.inpainting,
    )

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
            "process_video_task",
            task_id=task_id,
            video_key=req.video_key,
            person_click={"x": req.person_click.x, "y": req.person_click.y},
            frame_skip=req.frame_skip,
            layer=req.layer,
            tracking=req.tracking,
            export=req.export,
            ml_flags=ml_flags,
            session_id=req.session_id,
        )
    finally:
        await arq_pool.close()

    return QueueProcessResponse(task_id=task_id)


@router.get("/process/{task_id}/status", response_model=TaskStatusResponse)
async def get_process_status(task_id: str):
    """Poll task status."""
    valkey = await get_valkey_client()
    try:
        state = await get_task_state(task_id, valkey=valkey)
    finally:
        await valkey.close()

    if state is None:
        raise HTTPException(status_code=404, detail="Task not found")

    result = None
    if state.get("result"):
        result = ProcessResponse(**state["result"])

    return TaskStatusResponse(
        task_id=task_id,
        status=state["status"],
        progress=state["progress"],
        message=state.get("message", ""),
        result=result,
        error=state.get("error"),
    )


@router.post("/process/{task_id}/cancel")
async def cancel_queued_process(task_id: str):
    """Cancel a queued or running task via Valkey signal."""
    await set_cancel_signal(task_id)
    return {"status": "cancel_requested", "task_id": task_id}
