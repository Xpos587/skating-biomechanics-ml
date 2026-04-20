"""Unit tests for Valkey-backed task state management with mocked redis client."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import app.task_manager as tm_module
import pytest
from app.task_manager import (
    TASK_CANCEL_PREFIX,
    TASK_EVENTS_PREFIX,
    TASK_KEY_PREFIX,
    TaskStatus,
)


@pytest.fixture(autouse=True)
def reset_pool():
    """Reset module-level _pool dict before each test."""
    original = tm_module._pool.copy()
    tm_module._pool = {}
    yield
    tm_module._pool = original


def _mock_settings():
    """Create a mock Settings object with sensible defaults."""
    settings = MagicMock()
    settings.valkey.host = "localhost"
    settings.valkey.port = 6379
    settings.valkey.db = 0
    settings.valkey.password.get_secret_value.return_value = ""
    settings.app.task_ttl_seconds = 3600
    return settings


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------


async def test_init_valkey_pool():
    """init_valkey_pool creates connection and stores in _pool."""
    mock_settings = _mock_settings()
    with (
        patch("app.task_manager.get_settings", return_value=mock_settings),
        patch("app.task_manager.aioredis.Redis") as MockRedis,
    ):
        await tm_module.init_valkey_pool()
        MockRedis.assert_called_once_with(
            host="localhost",
            port=6379,
            db=0,
            password="",
            decode_responses=True,
        )
        assert "valkey" in tm_module._pool


async def test_init_valkey_pool_idempotent():
    """Second call to init_valkey_pool does not create new connection."""
    mock_settings = _mock_settings()
    with (
        patch("app.task_manager.get_settings", return_value=mock_settings),
        patch("app.task_manager.aioredis.Redis") as MockRedis,
    ):
        await tm_module.init_valkey_pool()
        await tm_module.init_valkey_pool()
        MockRedis.assert_called_once()


def test_get_valkey_returns_connection():
    """get_valkey returns pooled connection."""
    fake_conn = MagicMock()
    tm_module._pool["valkey"] = fake_conn
    result = tm_module.get_valkey()
    assert result is fake_conn


def test_get_valkey_raises_if_not_initialized():
    """get_valkey raises RuntimeError when pool is empty."""
    with pytest.raises(RuntimeError, match="init_valkey_pool"):
        tm_module.get_valkey()


async def test_close_valkey_pool():
    """close_valkey_pool closes and removes connection from pool."""
    mock_conn = AsyncMock()
    tm_module._pool["valkey"] = mock_conn
    await tm_module.close_valkey_pool()
    mock_conn.aclose.assert_awaited_once()
    assert "valkey" not in tm_module._pool


async def test_close_valkey_pool_noop():
    """close_valkey_pool does not error when pool is empty."""
    await tm_module.close_valkey_pool()  # should not raise


async def test_get_valkey_client():
    """get_valkey_client returns a new Redis instance with correct params."""
    mock_settings = _mock_settings()
    with (
        patch("app.task_manager.get_settings", return_value=mock_settings),
        patch("app.task_manager.aioredis.Redis") as MockRedis,
    ):
        client = await tm_module.get_valkey_client()
        MockRedis.assert_called_once_with(
            host="localhost",
            port=6379,
            db=0,
            password="",
            decode_responses=True,
        )
        assert client is MockRedis.return_value


# ---------------------------------------------------------------------------
# Task state operations with auto-client (valkey=None)
# ---------------------------------------------------------------------------


@patch("app.task_manager.get_valkey_client")
@patch("app.task_manager.get_settings")
async def test_create_task_state(mock_settings, mock_get_client):
    """create_task_state sets hash fields, calls expire, and closes client."""
    mock_settings.return_value = _mock_settings()
    mock_redis = AsyncMock()
    mock_get_client.return_value = mock_redis

    await tm_module.create_task_state("t1", "input/video.mp4")

    mock_redis.hset.assert_awaited_once()
    call_args = mock_redis.hset.call_args
    assert call_args[0][0] == f"{TASK_KEY_PREFIX}t1"
    mapping = call_args[1]["mapping"]
    assert mapping["task_id"] == "t1"
    assert mapping["status"] == TaskStatus.PENDING
    assert mapping["video_key"] == "input/video.mp4"
    assert mapping["progress"] == "0.0"
    assert mapping["message"] == "Queued"
    assert mapping["error"] == ""

    mock_redis.expire.assert_awaited_once_with(f"{TASK_KEY_PREFIX}t1", 3600)
    mock_redis.close.assert_awaited_once()


@patch("app.task_manager.get_valkey_client")
async def test_update_progress(mock_get_client):
    """update_progress updates progress and message fields."""
    mock_redis = AsyncMock()
    mock_get_client.return_value = mock_redis

    await tm_module.update_progress("t1", 0.5, "Processing...")

    mock_redis.hset.assert_awaited_once()
    call_args = mock_redis.hset.call_args
    assert call_args[0][0] == f"{TASK_KEY_PREFIX}t1"
    mapping = call_args[1]["mapping"]
    assert mapping["progress"] == "0.5"
    assert mapping["message"] == "Processing..."

    mock_redis.close.assert_awaited_once()


@patch("app.task_manager.get_valkey_client")
async def test_store_result(mock_get_client):
    """store_result sets status=completed, progress=1.0, and stores JSON result."""
    mock_redis = AsyncMock()
    mock_get_client.return_value = mock_redis

    result_data = {"video_path": "output.mp4", "frames": 100}
    await tm_module.store_result("t1", result_data)

    mock_redis.hset.assert_awaited_once()
    mapping = mock_redis.hset.call_args[1]["mapping"]
    assert mapping["status"] == TaskStatus.COMPLETED
    assert mapping["progress"] == "1.0"
    assert mapping["message"] == "Done"
    assert json.loads(mapping["result"]) == result_data

    mock_redis.close.assert_awaited_once()


@patch("app.task_manager.get_valkey_client")
async def test_store_error(mock_get_client):
    """store_error sets status=failed and stores error message."""
    mock_redis = AsyncMock()
    mock_get_client.return_value = mock_redis

    await tm_module.store_error("t1", "OOM error")

    mock_redis.hset.assert_awaited_once()
    mapping = mock_redis.hset.call_args[1]["mapping"]
    assert mapping["status"] == TaskStatus.FAILED
    assert mapping["error"] == "OOM error"
    assert mapping["completed_at"]  # timestamp set

    mock_redis.close.assert_awaited_once()


@patch("app.task_manager.get_valkey_client")
async def test_mark_cancelled(mock_get_client):
    """mark_cancelled sets status=cancelled."""
    mock_redis = AsyncMock()
    mock_get_client.return_value = mock_redis

    await tm_module.mark_cancelled("t1")

    mock_redis.hset.assert_awaited_once()
    mapping = mock_redis.hset.call_args[1]["mapping"]
    assert mapping["status"] == TaskStatus.CANCELLED
    assert mapping["message"] == "Cancelled"

    mock_redis.close.assert_awaited_once()


@patch("app.task_manager.get_valkey_client")
async def test_get_task_state(mock_get_client):
    """get_task_state retrieves hash and parses JSON result field."""
    mock_redis = AsyncMock()
    mock_get_client.return_value = mock_redis

    result_json = json.dumps({"video_path": "out.mp4"})
    mock_redis.hgetall.return_value = {
        "task_id": "t1",
        "status": "completed",
        "progress": "1.0",
        "result": result_json,
        "error": "",
    }

    state = await tm_module.get_task_state("t1")

    mock_redis.hgetall.assert_awaited_once_with(f"{TASK_KEY_PREFIX}t1")
    assert state is not None
    assert state["task_id"] == "t1"
    assert state["status"] == "completed"
    assert state["progress"] == 1.0
    assert state["result"] == {"video_path": "out.mp4"}
    assert state["error"] == ""

    mock_redis.close.assert_awaited_once()


@patch("app.task_manager.get_valkey_client")
async def test_get_task_state_nonexistent(mock_get_client):
    """get_task_state returns None for missing keys."""
    mock_redis = AsyncMock()
    mock_get_client.return_value = mock_redis

    mock_redis.hgetall.return_value = {}

    state = await tm_module.get_task_state("nonexistent")

    assert state is None
    mock_redis.close.assert_awaited_once()


@patch("app.task_manager.get_valkey_client")
async def test_get_task_state_no_result_field(mock_get_client):
    """get_task_state handles hash without result field (sets result=None)."""
    mock_redis = AsyncMock()
    mock_get_client.return_value = mock_redis

    mock_redis.hgetall.return_value = {
        "task_id": "t1",
        "status": "pending",
        "progress": "0",
    }

    state = await tm_module.get_task_state("t1")

    assert state is not None
    assert state["result"] is None
    assert state["progress"] == 0.0


@patch("app.task_manager.get_valkey_client")
async def test_is_cancelled(mock_get_client):
    """is_cancelled returns True when cancel signal is set."""
    mock_redis = AsyncMock()
    mock_get_client.return_value = mock_redis

    mock_redis.get.return_value = "1"

    result = await tm_module.is_cancelled("t1")

    assert result is True
    mock_redis.get.assert_awaited_once_with(f"{TASK_CANCEL_PREFIX}t1")
    mock_redis.close.assert_awaited_once()


@patch("app.task_manager.get_valkey_client")
async def test_is_cancelled_false(mock_get_client):
    """is_cancelled returns False when no cancel signal."""
    mock_redis = AsyncMock()
    mock_get_client.return_value = mock_redis

    mock_redis.get.return_value = None

    result = await tm_module.is_cancelled("t1")

    assert result is False


@patch("app.task_manager.get_valkey_client")
@patch("app.task_manager.get_settings")
async def test_set_cancel_signal(mock_settings, mock_get_client):
    """set_cancel_signal sets key with TTL."""
    mock_settings.return_value = _mock_settings()
    mock_redis = AsyncMock()
    mock_get_client.return_value = mock_redis

    await tm_module.set_cancel_signal("t1")

    mock_redis.setex.assert_awaited_once_with(f"{TASK_CANCEL_PREFIX}t1", 3600, "1")
    mock_redis.close.assert_awaited_once()


@patch("app.task_manager.get_valkey_client")
async def test_publish_task_event(mock_get_client):
    """publish_task_event publishes JSON to correct channel."""
    mock_redis = AsyncMock()
    mock_get_client.return_value = mock_redis

    event_data = {"status": "running", "progress": 0.5}
    await tm_module.publish_task_event("t1", event_data)

    mock_redis.publish.assert_awaited_once()
    channel, message = mock_redis.publish.call_args[0]
    assert channel == f"{TASK_EVENTS_PREFIX}t1"
    assert json.loads(message) == event_data

    mock_redis.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# With explicit valkey (no auto-close)
# ---------------------------------------------------------------------------


async def test_create_task_state_with_valkey_no_close():
    """When valkey is provided, create_task_state should NOT close it."""
    mock_valkey = AsyncMock()

    with patch("app.task_manager.get_settings") as mock_settings:
        mock_settings.return_value = _mock_settings()
        await tm_module.create_task_state("t1", "input/video.mp4", valkey=mock_valkey)

    mock_valkey.hset.assert_awaited_once()
    mock_valkey.expire.assert_awaited_once()
    mock_valkey.close.assert_not_awaited()


async def test_get_task_state_with_valkey_no_close():
    """When valkey is provided, get_task_state should NOT close it."""
    mock_valkey = AsyncMock()
    mock_valkey.hgetall.return_value = {"task_id": "t1", "status": "pending", "progress": "0"}

    state = await tm_module.get_task_state("t1", valkey=mock_valkey)

    assert state is not None
    mock_valkey.hgetall.assert_awaited_once()
    mock_valkey.close.assert_not_awaited()


async def test_publish_task_event_with_valkey_no_close():
    """When valkey is provided, publish_task_event should NOT close it."""
    mock_valkey = AsyncMock()

    await tm_module.publish_task_event("t1", {"status": "done"}, valkey=mock_valkey)

    mock_valkey.publish.assert_awaited_once()
    mock_valkey.close.assert_not_awaited()


async def test_update_progress_with_valkey_no_close():
    """When valkey is provided, update_progress should NOT close it."""
    mock_valkey = AsyncMock()

    await tm_module.update_progress("t1", 0.75, "Almost done", valkey=mock_valkey)

    mock_valkey.hset.assert_awaited_once()
    mock_valkey.close.assert_not_awaited()


async def test_store_result_with_valkey_no_close():
    """When valkey is provided, store_result should NOT close it."""
    mock_valkey = AsyncMock()

    await tm_module.store_result("t1", {"key": "val"}, valkey=mock_valkey)

    mock_valkey.hset.assert_awaited_once()
    mock_valkey.close.assert_not_awaited()


async def test_store_error_with_valkey_no_close():
    """When valkey is provided, store_error should NOT close it."""
    mock_valkey = AsyncMock()

    await tm_module.store_error("t1", "fail", valkey=mock_valkey)

    mock_valkey.hset.assert_awaited_once()
    mock_valkey.close.assert_not_awaited()


async def test_mark_cancelled_with_valkey_no_close():
    """When valkey is provided, mark_cancelled should NOT close it."""
    mock_valkey = AsyncMock()

    await tm_module.mark_cancelled("t1", valkey=mock_valkey)

    mock_valkey.hset.assert_awaited_once()
    mock_valkey.close.assert_not_awaited()


async def test_is_cancelled_with_valkey_no_close():
    """When valkey is provided, is_cancelled should NOT close it."""
    mock_valkey = AsyncMock()
    mock_valkey.get.return_value = "1"

    result = await tm_module.is_cancelled("t1", valkey=mock_valkey)

    assert result is True
    mock_valkey.close.assert_not_awaited()


async def test_set_cancel_signal_with_valkey_no_close():
    """When valkey is provided, set_cancel_signal should NOT close it."""
    mock_valkey = AsyncMock()

    with patch("app.task_manager.get_settings") as mock_settings:
        mock_settings.return_value = _mock_settings()
        await tm_module.set_cancel_signal("t1", valkey=mock_valkey)

    mock_valkey.setex.assert_awaited_once()
    mock_valkey.close.assert_not_awaited()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@patch("app.task_manager.get_valkey_client")
async def test_update_progress_rounding(mock_get_client):
    """update_progress rounds fraction to 3 decimal places."""
    mock_redis = AsyncMock()
    mock_get_client.return_value = mock_redis

    await tm_module.update_progress("t1", 0.123456, "msg")

    mapping = mock_redis.hset.call_args[1]["mapping"]
    assert mapping["progress"] == "0.123"


@patch("app.task_manager.get_valkey_client")
async def test_get_task_state_progress_default(mock_get_client):
    """get_task_state defaults progress to 0.0 when field is missing."""
    mock_redis = AsyncMock()
    mock_get_client.return_value = mock_redis

    mock_redis.hgetall.return_value = {"task_id": "t1"}

    state = await tm_module.get_task_state("t1")

    assert state["progress"] == 0.0
