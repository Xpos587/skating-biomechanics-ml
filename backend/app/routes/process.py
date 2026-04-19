"""POST /api/process/queue — enqueue video processing job."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app.schemas import (
    MLModelFlags,
    ProcessRequest,
    ProcessResponse,
    QueueProcessResponse,
    TaskStatusResponse,
)
from app.task_manager import (
    TASK_EVENTS_PREFIX,
    create_task_state,
    get_task_state,
    get_valkey_client,
    set_cancel_signal,
)

logger = logging.getLogger(__name__)
router = APIRouter()

SSE_STREAM_TIMEOUT = 60  # seconds


@router.post("/process/queue", response_model=QueueProcessResponse)
async def enqueue_process(request: Request, req: ProcessRequest):
    """Enqueue video processing job and return task_id immediately."""
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

    await request.app.state.arq_pool.enqueue_job(
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
        _priority=10,  # Low priority for full analysis
    )

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


@router.get("/process/{task_id}/stream")
async def stream_process_status(task_id: str):
    """SSE endpoint for real-time task progress streaming."""

    async def event_generator():
        valkey = await get_valkey_client()
        pubsub = valkey.pubsub()
        channel = f"{TASK_EVENTS_PREFIX}{task_id}"
        await pubsub.subscribe(channel)
        try:
            # Send initial state
            state = await get_task_state(task_id, valkey=valkey)
            if state:
                yield {"data": json.dumps(state)}
            else:
                yield {"data": json.dumps({"status": "unknown"})}

            async with asyncio.timeout(SSE_STREAM_TIMEOUT):
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        yield {"data": message["data"].decode()}
                        try:
                            data = json.loads(message["data"])
                            if data.get("status") in ("completed", "failed", "cancelled"):
                                break
                        except (json.JSONDecodeError, TypeError):
                            pass
        except TimeoutError:
            # No messages for 60s — poll final state and yield timeout event
            logger.warning("SSE stream timeout for task %s", task_id)
            state = await get_task_state(task_id, valkey=valkey)
            payload = state or {"status": "unknown"}
            payload["_timeout"] = True
            yield {"data": json.dumps(payload)}
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
            await valkey.close()

    return EventSourceResponse(event_generator())
