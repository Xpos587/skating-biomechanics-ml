"""Tests for Vast.ai client module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.vastai.client import VastResult, _get_worker_url


def test_get_worker_url_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"url": "https://worker-1.vast.ai:8000"}
    mock_resp.raise_for_status = MagicMock()

    with patch("app.vastai.client.httpx.post", return_value=mock_resp) as mock_post:
        url = _get_worker_url("skating-ml-gpu", "test-key")

    assert url == "https://worker-1.vast.ai:8000"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["endpoint"] == "skating-ml-gpu"


def test_vast_result_fields():
    r = VastResult(
        video_key="output/test_analyzed.mp4",
        poses_key="output/test_poses.npy",
        csv_key=None,
        stats={"frames": 100},
        metrics=None,
        phases=None,
        recommendations=None,
    )
    assert r.video_key == "output/test_analyzed.mp4"
    assert r.csv_key is None
    assert r.stats == {"frames": 100}


@patch("app.vastai.client.httpx.post")
def test_process_video_remote_passes_r2_key(mock_post):
    # Reset global worker URL cache (may be set by earlier tests)
    import app.vastai.client as _vc

    _vc._worker_url_cache = None
    _vc._worker_url_cache_time = 0.0

    mock_route_resp = MagicMock()
    mock_route_resp.status_code = 200
    mock_route_resp.json.return_value = {"url": "https://worker.vast.ai:8000"}
    mock_route_resp.raise_for_status = MagicMock()

    mock_process_resp = MagicMock()
    mock_process_resp.status_code = 200
    mock_process_resp.json.return_value = {
        "video_r2_key": "output/test_analyzed.mp4",
        "poses_r2_key": "output/test_poses.npy",
        "csv_r2_key": None,
        "stats": {"total_frames": 100, "valid_frames": 90, "fps": 30.0, "resolution": "1920x1080"},
    }
    mock_process_resp.raise_for_status = MagicMock()

    mock_post.side_effect = [mock_route_resp, mock_process_resp]

    with patch("app.vastai.client.get_settings") as mock_settings:
        s = MagicMock()
        s.vastai.api_key.get_secret_value.return_value = "test-key"
        s.vastai.endpoint_name = "skating-ml-gpu"
        s.r2.endpoint_url = "https://r2.example.com"
        s.r2.access_key_id.get_secret_value.return_value = "key-id"
        s.r2.secret_access_key.get_secret_value.return_value = "secret"
        s.r2.bucket = "test-bucket"
        mock_settings.return_value = s

        from app.vastai.client import process_video_remote

        result = process_video_remote(
            video_key="input/test.mp4",
            person_click={"x": 100, "y": 200},
        )

    assert result.video_key == "output/test_analyzed.mp4"
    assert result.poses_key == "output/test_poses.npy"
    assert result.csv_key is None
    assert mock_post.call_count == 2


def test_worker_url_cache_is_thread_safe():
    """Verify cache access is protected by a lock."""
    import threading

    import app.vastai.client as _vc

    # Reset cache
    _vc._worker_url_cache = None
    _vc._worker_url_cache_time = 0.0

    # Access the lock — it should exist
    assert hasattr(_vc, "_worker_url_lock")
    assert isinstance(_vc._worker_url_lock, type(threading.Lock()))


def test_get_worker_url_uses_cache():
    """Second call within TTL should not make HTTP request."""
    import app.vastai.client as _vc

    _vc._worker_url_cache = None
    _vc._worker_url_cache_time = 0.0

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"url": "https://cached.vast.ai:8000"}
    mock_resp.raise_for_status = MagicMock()

    with patch("app.vastai.client.httpx.post", return_value=mock_resp) as mock_post:
        url1 = _vc._get_worker_url("ep", "key")
        url2 = _vc._get_worker_url("ep", "key")

    assert url1 == "https://cached.vast.ai:8000"
    assert url2 == "https://cached.vast.ai:8000"
    mock_post.assert_called_once()  # Only one HTTP call, second uses cache


async def test_async_client_is_reused_across_calls():
    """Verify _get_async_client returns the same client instance."""
    import app.vastai.client as _vc

    # Reset module-level client
    _vc._async_client = None

    c1 = _vc._get_async_client()
    c2 = _vc._get_async_client()
    assert c1 is c2

    # Cleanup
    await c1.aclose()
    _vc._async_client = None
