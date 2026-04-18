"""Client for calling Vast.ai Serverless GPU endpoint.

Flow:
  1. POST /route to get worker URL from Vast.ai
  2. POST /process to the worker with R2 key + credentials
  3. Worker processes and uploads results to R2
  4. Return R2 keys (no local download)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

from backend.app.config import get_settings

logger = logging.getLogger(__name__)

ROUTE_URL = "https://run.vast.ai/route/"
REQUEST_TIMEOUT = 600  # 10 min for video processing
ROUTE_TIMEOUT = 30

# Worker URL cache to avoid repeated HTTP calls
_worker_url_cache: str | None = None
_worker_url_cache_time: float = 0.0
_WORKER_URL_TTL = 60  # Cache for 60 seconds


@dataclass
class VastResult:
    video_key: str
    poses_key: str | None
    csv_key: str | None
    stats: dict
    metrics: list | None
    phases: object | None
    recommendations: list | None


def _get_worker_url(endpoint_name: str, api_key: str) -> str:
    """Route request to get a ready worker URL."""
    global _worker_url_cache, _worker_url_cache_time  # noqa: PLW0603
    now = time.monotonic()
    if _worker_url_cache and (now - _worker_url_cache_time) < _WORKER_URL_TTL:
        return _worker_url_cache
    resp = httpx.post(
        ROUTE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"endpoint": endpoint_name},
        timeout=ROUTE_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    url = data["url"]
    _worker_url_cache = url  # noqa: PLW0603
    _worker_url_cache_time = now  # noqa: PLW0603
    return url


def process_video_remote(
    video_key: str,
    person_click: dict[str, int] | None = None,
    frame_skip: int = 1,
    layer: int = 3,
    tracking: str = "auto",
    export: bool = True,
    ml_flags: dict[str, bool] | None = None,
    element_type: str | None = None,
) -> VastResult:
    """Send video processing to Vast.ai Serverless GPU.

    Video must already be in R2 at `video_key`.
    Returns R2 keys for results (no local download).

    Raises httpx.HTTPStatusError on routing/processing failures.
    """
    settings = get_settings()
    if ml_flags is None:
        ml_flags = {}

    api_key = settings.vastai.api_key.get_secret_value()
    endpoint_name = settings.vastai.endpoint_name

    # 1. Route to worker
    logger.info("Routing to Vast.ai endpoint: %s", endpoint_name)
    worker_url = _get_worker_url(endpoint_name, api_key)
    logger.info("Worker URL: %s", worker_url)

    # 2. Send processing request (video is already in R2)
    payload = {
        "video_r2_key": video_key,
        "person_click": person_click,
        "frame_skip": frame_skip,
        "layer": layer,
        "tracking": tracking,
        "export": export,
        "ml_flags": ml_flags,
        "element_type": element_type,
        "r2_endpoint_url": settings.r2.endpoint_url,
        "r2_access_key_id": settings.r2.access_key_id.get_secret_value(),
        "r2_secret_access_key": settings.r2.secret_access_key.get_secret_value(),
        "r2_bucket": settings.r2.bucket,
    }
    resp = httpx.post(
        f"{worker_url}/process",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    result = resp.json()

    # 3. Return R2 keys directly (no download)
    return VastResult(
        video_key=result["video_r2_key"],
        poses_key=result.get("poses_r2_key"),
        csv_key=result.get("csv_r2_key"),
        stats=result["stats"],
        metrics=result.get("metrics"),
        phases=result.get("phases"),
        recommendations=result.get("recommendations"),
    )


async def _asyncio_get_worker_url(endpoint_name: str, api_key: str) -> str:
    """Async route request to get a ready worker URL (with TTL cache)."""
    global _worker_url_cache, _worker_url_cache_time  # noqa: PLW0603
    now = time.monotonic()
    if _worker_url_cache and (now - _worker_url_cache_time) < _WORKER_URL_TTL:
        return _worker_url_cache
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            ROUTE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"endpoint": endpoint_name},
            timeout=ROUTE_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        url = data["url"]
        _worker_url_cache = url  # noqa: PLW0603
        _worker_url_cache_time = now  # noqa: PLW0603
        return url


async def process_video_remote_async(
    video_key: str,
    person_click: dict[str, int] | None = None,
    frame_skip: int = 1,
    layer: int = 3,
    tracking: str = "auto",
    export: bool = True,
    ml_flags: dict[str, bool] | None = None,
    element_type: str | None = None,
) -> VastResult:
    """Async version of process_video_remote using httpx.AsyncClient."""
    settings = get_settings()
    if ml_flags is None:
        ml_flags = {}

    api_key = settings.vastai.api_key.get_secret_value()
    endpoint_name = settings.vastai.endpoint_name

    # 1. Route to worker (async)
    logger.info("Routing to Vast.ai endpoint: %s", endpoint_name)
    worker_url = await _asyncio_get_worker_url(endpoint_name, api_key)
    logger.info("Worker URL: %s", worker_url)

    # 2. Send processing request (video is already in R2)
    payload = {
        "video_r2_key": video_key,
        "person_click": person_click,
        "frame_skip": frame_skip,
        "layer": layer,
        "tracking": tracking,
        "export": export,
        "ml_flags": ml_flags,
        "element_type": element_type,
        "r2_endpoint_url": settings.r2.endpoint_url,
        "r2_access_key_id": settings.r2.access_key_id.get_secret_value(),
        "r2_secret_access_key": settings.r2.secret_access_key.get_secret_value(),
        "r2_bucket": settings.r2.bucket,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{worker_url}/process",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()

    # 3. Return R2 keys directly (no download)
    return VastResult(
        video_key=result["video_r2_key"],
        poses_key=result.get("poses_r2_key"),
        csv_key=result.get("csv_r2_key"),
        stats=result["stats"],
        metrics=result.get("metrics"),
        phases=result.get("phases"),
        recommendations=result.get("recommendations"),
    )
