"""Tests for analyze_music_task arq job."""

from __future__ import annotations

import sys
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

# Mock aiobotocore before importing app.worker (which imports app.storage)
_mock_aiobotocore = MagicMock()
_mock_aiobotocore_session = MagicMock()
sys.modules["aiobotocore"] = _mock_aiobotocore
sys.modules["aiobotocore.session"] = _mock_aiobotocore_session


@pytest.mark.asyncio
async def test_analyze_music_task_duplicate_found():
    """Test analyze_music_task when duplicate fingerprint is found."""
    from app.worker import analyze_music_task

    mock_ctx = {}

    music_id = "music_123"
    r2_key = "music/user_123/music_123.mp3"

    # Mock all external dependencies
    with (
        patch("app.worker.get_valkey_client") as mock_get_valkey,
        patch("asyncio.to_thread") as mock_to_thread,
        patch("app.database.async_session_factory") as mock_async_session_factory,
        patch("app.worker.download_file"),
        patch(
            "app.services.choreography.fingerprint.compute_fingerprint"
        ) as mock_compute_fingerprint,
        patch("app.crud.choreography.get_music_analysis_by_id") as mock_get_music,
        patch("app.crud.choreography.find_music_by_fingerprint") as mock_find_duplicate,
        patch("app.crud.choreography.update_music_analysis") as mock_update_music,
    ):
        # Setup mocks
        mock_valkey = AsyncMock()
        mock_valkey.close = AsyncMock()
        mock_get_valkey.return_value = mock_valkey

        # Mock asyncio.to_thread to return different values based on function
        def mock_to_thread_side_effect(func, *args, **kwargs):
            # Check by string representation since mocks don't have __name__
            func_str = str(func)
            if "download_file" in func_str:
                return None
            elif "compute_fingerprint" in func_str:
                return "a" * 32
            return None

        mock_to_thread.side_effect = mock_to_thread_side_effect

        # Mock fingerprint (also needed for direct call check)
        mock_compute_fingerprint.return_value = "a" * 32

        # Mock music record
        mock_music = MagicMock()
        mock_music.id = music_id
        mock_music.audio_url = ""
        mock_get_music.return_value = mock_music

        # Mock duplicate record
        mock_duplicate = MagicMock()
        mock_duplicate.id = "music_456"
        mock_duplicate.fingerprint = "a" * 32
        mock_duplicate.bpm = 120.0
        mock_duplicate.duration_sec = 180.0
        mock_duplicate.peaks = [10.0, 20.0]
        mock_duplicate.structure = [{"type": "verse", "start": 0.0, "end": 30.0}]
        mock_duplicate.energy_curve = {"timestamps": [0.0, 0.5], "values": [0.5, 0.6]}
        mock_find_duplicate.return_value = mock_duplicate

        # Mock DB session
        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_async_session_factory.return_value.__aenter__.return_value = mock_db

        # Run task
        result = await analyze_music_task(mock_ctx, music_id=music_id, r2_key=r2_key)

        # Verify result
        assert result["status"] == "completed"
        assert result["music_id"] == music_id
        assert result["duplicate_of"] == "music_456"
        assert result["bpm"] == 120.0
        assert result["duration_sec"] == 180.0

        # Verify fingerprint was stored
        mock_update_music.assert_any_call(mock_db, mock_music, fingerprint="a" * 32)

        # Verify duplicate analysis was copied
        mock_update_music.assert_any_call(
            mock_db,
            mock_music,
            audio_url="",
            duration_sec=180.0,
            bpm=120.0,
            peaks=[10.0, 20.0],
            structure=[{"type": "verse", "start": 0.0, "end": 30.0}],
            energy_curve={"timestamps": [0.0, 0.5], "values": [0.5, 0.6]},
            status="completed",
        )


@pytest.mark.asyncio
async def test_analyze_music_task_full_analysis():
    """Test analyze_music_task when no duplicate is found."""
    from app.worker import analyze_music_task

    mock_ctx = {}

    music_id = "music_123"
    r2_key = "music/user_123/music_123.mp3"

    with (
        patch("app.worker.get_valkey_client") as mock_get_valkey,
        patch("asyncio.to_thread") as mock_to_thread,
        patch("app.database.async_session_factory") as mock_async_session_factory,
        patch("app.worker.download_file"),
        patch(
            "app.services.choreography.fingerprint.compute_fingerprint"
        ) as mock_compute_fingerprint,
        patch("app.crud.choreography.get_music_analysis_by_id") as mock_get_music,
        patch("app.crud.choreography.find_music_by_fingerprint") as mock_find_duplicate,
        patch("app.services.choreography.music_analyzer.analyze_music_sync") as mock_analyze,
        patch("app.crud.choreography.update_music_analysis") as mock_update_music,
    ):
        # Setup mocks
        mock_valkey = AsyncMock()
        mock_valkey.close = AsyncMock()
        mock_get_valkey.return_value = mock_valkey

        # Mock asyncio.to_thread to return different values based on function
        def mock_to_thread_side_effect(func, *args, **kwargs):
            # Check by string representation since mocks don't have __name__
            func_str = str(func)
            if "download_file" in func_str:
                return None
            elif "compute_fingerprint" in func_str:
                return "b" * 32
            elif "analyze_music_sync" in func_str:
                return {
                    "bpm": 140.0,
                    "duration_sec": 200.0,
                    "peaks": [15.0, 25.0],
                    "structure": [{"type": "chorus", "start": 30.0, "end": 60.0}],
                    "energy_curve": {"timestamps": [0.0, 0.5], "values": [0.7, 0.8]},
                }
            return None

        mock_to_thread.side_effect = mock_to_thread_side_effect

        # Mock fingerprint (also needed for direct call check)
        mock_compute_fingerprint.return_value = "b" * 32

        # Mock music record
        mock_music = MagicMock()
        mock_music.id = music_id
        mock_music.audio_url = ""
        mock_get_music.return_value = mock_music

        # No duplicate found
        mock_find_duplicate.return_value = None

        # Mock DB session
        mock_db = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_async_session_factory.return_value.__aenter__.return_value = mock_db

        # Run task
        result = await analyze_music_task(mock_ctx, music_id=music_id, r2_key=r2_key)

        # Verify result
        assert result["status"] == "completed"
        assert result["music_id"] == music_id
        assert result["bpm"] == 140.0
        assert result["duration_sec"] == 200.0

        # Verify full analysis was run
        mock_to_thread.assert_any_call(mock_analyze, ANY)

        # Verify results were stored
        mock_update_music.assert_any_call(
            mock_db,
            mock_music,
            audio_url="/files/music/user_123/music_123.mp3",
            duration_sec=200.0,
            bpm=140.0,
            peaks=[15.0, 25.0],
            structure=[{"type": "chorus", "start": 30.0, "end": 60.0}],
            energy_curve={"timestamps": [0.0, 0.5], "values": [0.7, 0.8]},
            status="completed",
        )


@pytest.mark.asyncio
async def test_analyze_music_task_failure():
    """Test analyze_music_task error handling."""
    from app.worker import analyze_music_task

    mock_ctx = {}

    music_id = "music_123"
    r2_key = "music/user_123/music_123.mp3"

    with (
        patch("app.worker.get_valkey_client") as mock_get_valkey,
        patch("app.database.async_session_factory") as mock_async_session_factory,
        patch("app.crud.choreography.get_music_analysis_by_id") as mock_get_music,
        patch("app.crud.choreography.update_music_analysis") as mock_update_music,
    ):
        # Setup mocks
        mock_valkey = AsyncMock()
        mock_valkey.close = AsyncMock()
        mock_get_valkey.return_value = mock_valkey

        # Mock download to raise error
        async def mock_download_error(*args, **kwargs):
            raise RuntimeError("Download failed")

        with patch("asyncio.to_thread", side_effect=mock_download_error):
            # Mock music record for failure update
            mock_music = MagicMock()
            mock_get_music.return_value = mock_music

            # Mock DB session
            mock_db = AsyncMock()
            mock_db.commit = AsyncMock()
            mock_async_session_factory.return_value.__aenter__.return_value = mock_db

            # Run task - should raise
            with pytest.raises(RuntimeError, match="Download failed"):
                await analyze_music_task(mock_ctx, music_id=music_id, r2_key=r2_key)

            # Verify status was updated to failed
            mock_update_music.assert_called_with(mock_db, mock_music, status="failed")
