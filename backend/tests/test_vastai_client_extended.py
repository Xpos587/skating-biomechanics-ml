"""Extended tests for Vast.ai client async functions (lines 146-221)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.vastai.client import (
    VastResult,
    _asyncio_get_worker_url,
    _get_async_client,
    process_video_remote_async,
)


def _reset_caches():
    """Reset module-level caches between tests."""
    import app.vastai.client as _vc

    _vc._worker_url_cache = None
    _vc._worker_url_cache_time = 0.0
    _vc._async_client = None


def _make_settings():
    s = MagicMock()
    s.vastai.api_key.get_secret_value.return_value = "test-api-key"
    s.vastai.endpoint_name = "skating-ml-gpu"
    s.r2.endpoint_url = "https://r2.example.com"
    s.r2.access_key_id.get_secret_value.return_value = "r2-key-id"
    s.r2.secret_access_key.get_secret_value.return_value = "r2-secret"
    s.r2.bucket = "test-bucket"
    return s


def _make_process_result(**overrides):
    data = {
        "video_r2_key": "output/test_analyzed.mp4",
        "poses_r2_key": "output/test_poses.npy",
        "csv_r2_key": "output/test.csv",
        "stats": {"total_frames": 100, "valid_frames": 90, "fps": 30.0},
        "metrics": [{"name": "airtime", "value": 0.5}],
        "phases": {"takeoff": 10, "peak": 20, "landing": 30},
        "recommendations": ["Keep your back straight"],
    }
    data.update(overrides)
    return data


@pytest.fixture(autouse=True)
def reset_caches():
    _reset_caches()
    yield
    _reset_caches()


# ---------------------------------------------------------------------------
# _asyncio_get_worker_url
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_asyncio_get_worker_url_success():
    """Makes HTTP call and caches the result."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"url": "https://async-worker.vast.ai:8000"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp

    with (
        patch("app.vastai.client._get_async_client", return_value=mock_client),
    ):
        url = await _asyncio_get_worker_url("skating-ml-gpu", "test-key")

    assert url == "https://async-worker.vast.ai:8000"
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["json"]["endpoint"] == "skating-ml-gpu"
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_asyncio_get_worker_url_uses_cache():
    """Second call within TTL returns cached URL without HTTP call."""
    import app.vastai.client as _vc

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"url": "https://cached-async.vast.ai:8000"}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp

    with patch("app.vastai.client._get_async_client", return_value=mock_client):
        url1 = await _asyncio_get_worker_url("ep", "key")
        # Set cache manually so second call uses it
        url2 = await _asyncio_get_worker_url("ep", "key")

    assert url1 == "https://cached-async.vast.ai:8000"
    assert url2 == "https://cached-async.vast.ai:8000"
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_asyncio_get_worker_url_raises_on_error():
    """Propagates HTTP errors from the route endpoint."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.raise_for_status.side_effect = Exception("Unauthorized")

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp

    with (
        patch("app.vastai.client._get_async_client", return_value=mock_client),
        pytest.raises(Exception, match="Unauthorized"),
    ):
        await _asyncio_get_worker_url("ep", "bad-key")


# ---------------------------------------------------------------------------
# process_video_remote_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_video_remote_async_happy_path():
    """Full async flow: route -> process -> return VastResult."""
    mock_route_resp = MagicMock()
    mock_route_resp.status_code = 200
    mock_route_resp.json.return_value = {"url": "https://worker.vast.ai:8000"}
    mock_route_resp.raise_for_status = MagicMock()

    mock_process_resp = MagicMock()
    mock_process_resp.status_code = 200
    mock_process_resp.json.return_value = _make_process_result()
    mock_process_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.side_effect = [mock_route_resp, mock_process_resp]
    mock_client.is_closed = False

    with (
        patch("app.vastai.client._get_async_client", return_value=mock_client),
        patch("app.vastai.client.get_settings", return_value=_make_settings()),
    ):
        result = await process_video_remote_async(
            video_key="input/test.mp4",
            person_click={"x": 100, "y": 200},
            frame_skip=8,
            layer=2,
            tracking="centroid",
            export=False,
            ml_flags={"depth": True},
            element_type="waltz_jump",
        )

    assert isinstance(result, VastResult)
    assert result.video_key == "output/test_analyzed.mp4"
    assert result.poses_key == "output/test_poses.npy"
    assert result.csv_key == "output/test.csv"
    assert result.stats["total_frames"] == 100
    assert result.metrics == [{"name": "airtime", "value": 0.5}]
    assert result.phases == {"takeoff": 10, "peak": 20, "landing": 30}
    assert result.recommendations == ["Keep your back straight"]

    # Verify payload structure
    process_call = mock_client.post.call_args_list[1]
    payload = process_call.kwargs["json"]
    assert payload["video_r2_key"] == "input/test.mp4"
    assert payload["person_click"] == {"x": 100, "y": 200}
    assert payload["frame_skip"] == 8
    assert payload["layer"] == 2
    assert payload["tracking"] == "centroid"
    assert payload["export"] is False
    assert payload["ml_flags"] == {"depth": True}
    assert payload["element_type"] == "waltz_jump"
    assert payload["r2_endpoint_url"] == "https://r2.example.com"
    assert payload["r2_access_key_id"] == "r2-key-id"
    assert payload["r2_secret_access_key"] == "r2-secret"
    assert payload["r2_bucket"] == "test-bucket"


@pytest.mark.asyncio
async def test_process_video_remote_async_defaults():
    """Default parameter values produce correct payload."""
    mock_route_resp = MagicMock()
    mock_route_resp.status_code = 200
    mock_route_resp.json.return_value = {"url": "https://worker.vast.ai:8000"}
    mock_route_resp.raise_for_status = MagicMock()

    result_data = _make_process_result()
    # Remove optional fields to test None handling
    del result_data["poses_r2_key"]
    del result_data["csv_r2_key"]
    del result_data["metrics"]
    del result_data["phases"]
    del result_data["recommendations"]

    mock_process_resp = MagicMock()
    mock_process_resp.status_code = 200
    mock_process_resp.json.return_value = result_data
    mock_process_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post.side_effect = [mock_route_resp, mock_process_resp]
    mock_client.is_closed = False

    with (
        patch("app.vastai.client._get_async_client", return_value=mock_client),
        patch("app.vastai.client.get_settings", return_value=_make_settings()),
    ):
        result = await process_video_remote_async(video_key="input/test.mp4")

    assert result.video_key == "output/test_analyzed.mp4"
    assert result.poses_key is None
    assert result.csv_key is None
    assert result.metrics is None
    assert result.phases is None
    assert result.recommendations is None

    # Verify defaults in payload
    process_call = mock_client.post.call_args_list[1]
    payload = process_call.kwargs["json"]
    assert payload["person_click"] is None
    assert payload["frame_skip"] == 1
    assert payload["layer"] == 3
    assert payload["tracking"] == "auto"
    assert payload["export"] is True
    assert payload["ml_flags"] == {}
    assert payload["element_type"] is None


@pytest.mark.asyncio
async def test_process_video_remote_async_route_failure():
    """Raises when the route endpoint returns an error."""
    mock_route_resp = MagicMock()
    mock_route_resp.status_code = 404
    mock_route_resp.raise_for_status.side_effect = Exception("Not Found")

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_route_resp

    with (
        patch("app.vastai.client._get_async_client", return_value=mock_client),
        patch("app.vastai.client.get_settings", return_value=_make_settings()),
        pytest.raises(Exception, match="Not Found"),
    ):
        await process_video_remote_async(video_key="input/test.mp4")


@pytest.mark.asyncio
async def test_process_video_remote_async_process_failure():
    """Raises when the worker process endpoint returns an error."""
    mock_route_resp = MagicMock()
    mock_route_resp.status_code = 200
    mock_route_resp.json.return_value = {"url": "https://worker.vast.ai:8000"}
    mock_route_resp.raise_for_status = MagicMock()

    mock_process_resp = MagicMock()
    mock_process_resp.status_code = 500
    mock_process_resp.raise_for_status.side_effect = Exception("Worker Error")

    mock_client = AsyncMock()
    mock_client.post.side_effect = [mock_route_resp, mock_process_resp]
    mock_client.is_closed = False

    with (
        patch("app.vastai.client._get_async_client", return_value=mock_client),
        patch("app.vastai.client.get_settings", return_value=_make_settings()),
        pytest.raises(Exception, match="Worker Error"),
    ):
        await process_video_remote_async(video_key="input/test.mp4")


# ---------------------------------------------------------------------------
# _get_async_client lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_async_client_recreates_when_closed():
    """Creates a new client when the existing one is closed."""
    import app.vastai.client as _vc

    closed_client = MagicMock(spec=_vc.httpx.AsyncClient)
    closed_client.is_closed = True

    _vc._async_client = closed_client

    new_client = _get_async_client()
    assert new_client is not closed_client
    assert not new_client.is_closed

    _vc._async_client = None
