"""Tests for Valkey-backed task state management."""

import pytest

from backend.app.task_manager import (
    TaskStatus,
    create_task_state,
    get_task_state,
    is_cancelled,
    mark_cancelled,
    publish_task_event,
    set_cancel_signal,
    store_error,
    store_result,
    update_progress,
)


@pytest.fixture
async def valkey():
    import redis.asyncio as aioredis

    client = aioredis.Redis(host="localhost", port=6379, db=1, decode_responses=True)
    yield client
    keys = await client.keys("task:*")
    if keys:
        await client.delete(*keys)
    keys = await client.keys("task_cancel:*")
    if keys:
        await client.delete(*keys)
    await client.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_and_get_state(valkey):
    await create_task_state("test_task_1", "/tmp/video.mp4", valkey=valkey)
    state = await get_task_state("test_task_1", valkey=valkey)

    assert state is not None
    assert state["task_id"] == "test_task_1"
    assert state["status"] == TaskStatus.PENDING
    assert state["progress"] == 0.0
    assert state["video_path"] == "/tmp/video.mp4"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_progress(valkey):
    await create_task_state("test_task_2", "/tmp/video.mp4", valkey=valkey)
    await update_progress("test_task_2", 0.5, "Rendering...", valkey=valkey)
    state = await get_task_state("test_task_2", valkey=valkey)

    assert state["progress"] == 0.5
    assert state["message"] == "Rendering..."


@pytest.mark.integration
@pytest.mark.asyncio
async def test_store_result(valkey):
    await create_task_state("test_task_3", "/tmp/video.mp4", valkey=valkey)
    await store_result("test_task_3", {"video_path": "out.mp4", "stats": {}}, valkey=valkey)
    state = await get_task_state("test_task_3", valkey=valkey)

    assert state["status"] == TaskStatus.COMPLETED
    assert state["result"] == {"video_path": "out.mp4", "stats": {}}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_store_error(valkey):
    await create_task_state("test_task_4", "/tmp/video.mp4", valkey=valkey)
    await store_error("test_task_4", "OOM", valkey=valkey)
    state = await get_task_state("test_task_4", valkey=valkey)

    assert state["status"] == TaskStatus.FAILED
    assert state["error"] == "OOM"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancel_flow(valkey):
    await create_task_state("test_task_5", "/tmp/video.mp4", valkey=valkey)
    assert not await is_cancelled("test_task_5", valkey=valkey)

    await set_cancel_signal("test_task_5", valkey=valkey)
    assert await is_cancelled("test_task_5", valkey=valkey)

    await mark_cancelled("test_task_5", valkey=valkey)
    state = await get_task_state("test_task_5", valkey=valkey)
    assert state["status"] == TaskStatus.CANCELLED


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_nonexistent_returns_none(valkey):
    state = await get_task_state("nonexistent", valkey=valkey)
    assert state is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_publish_task_event(valkey):
    """publish_task_event should publish to pub/sub channel."""
    import asyncio
    import json

    pubsub = valkey.pubsub()
    channel = "task_events:test_pub_1"
    await pubsub.subscribe(channel)

    await asyncio.sleep(0.05)

    await publish_task_event("test_pub_1", {"status": "running", "progress": 0.5}, valkey=valkey)

    msg = await asyncio.wait_for(pubsub.get_message(timeout=1.0), timeout=1.0)
    while msg and msg["type"] != "message":
        msg = await asyncio.wait_for(pubsub.get_message(timeout=1.0), timeout=1.0)

    if msg:
        data = json.loads(msg["data"])
        assert data["status"] == "running"
        assert data["progress"] == 0.5
    else:
        pytest.fail("No message received from pub/sub")

    await pubsub.unsubscribe(channel)
    await pubsub.aclose()
