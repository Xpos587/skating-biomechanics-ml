"""Tests for POST /choreography/music/upload route."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status

# Mock aiobotocore before importing
_mock_aiobotocore = MagicMock()
_mock_aiobotocore_session = MagicMock()
sys.modules["aiobotocore"] = _mock_aiobotocore
sys.modules["aiobotocore.session"] = _mock_aiobotocore_session


@pytest.mark.asyncio
async def test_upload_music_enqueues_job():
    """Test that upload_music enqueues analyze_music_task and returns immediately."""
    from app.routes.choreography import upload_music
    from app.schemas import UploadMusicResponse

    # Mock user
    mock_user = MagicMock()
    mock_user.id = "user_123"

    # Mock DB session
    mock_db = AsyncMock()

    # Mock file
    mock_file = MagicMock()
    mock_file.filename = "test.mp3"
    mock_file.read = AsyncMock(return_value=b"fake audio content")

    # Mock music record
    mock_music = MagicMock()
    mock_music.id = "music_456"
    mock_music.filename = "test.mp3"

    # Mock request with arq_pool
    mock_request = MagicMock()
    mock_arq_pool = AsyncMock()
    mock_request.app.state.arq_pool = mock_arq_pool

    with (
        patch("app.routes.choreography.create_music_analysis") as mock_create,
        patch("app.routes.choreography.upload_file") as mock_upload,
        patch("app.routes.choreography.Path") as mock_path_class,
    ):
        # Setup mocks
        mock_create.return_value = mock_music

        # Mock tempfile
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/test.mp3"
        mock_tmp.write = MagicMock()
        mock_tmp.__enter__ = MagicMock(return_value=mock_tmp)
        mock_tmp.__exit__ = MagicMock(return_value=False)

        with patch("app.routes.choreography.tempfile.NamedTemporaryFile", return_value=mock_tmp):
            # Call the route
            response = await upload_music(mock_request, mock_user, mock_db, mock_file)

            # Verify response
            assert isinstance(response, UploadMusicResponse)
            assert response.music_id == "music_456"
            assert response.filename == "test.mp3"

            # Verify music record was created with status="pending"
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["status"] == "pending"
            assert call_kwargs["user_id"] == "user_123"
            assert call_kwargs["filename"] == "test.mp3"

            # Verify file was uploaded to R2
            mock_upload.assert_called_once()

            # Verify job was enqueued (not analyze_music_sync)
            mock_arq_pool.enqueue_job.assert_called_once_with(
                "analyze_music_task",
                music_id="music_456",
                r2_key="music/user_123/music_456.mp3",
                _queue_name="skating:queue:fast",
            )


@pytest.mark.asyncio
async def test_upload_music_handles_upload_failure():
    """Test that upload_music sets status to failed on upload error."""
    from app.routes.choreography import upload_music
    from fastapi import HTTPException

    # Mock user
    mock_user = MagicMock()
    mock_user.id = "user_123"

    # Mock DB session
    mock_db = AsyncMock()

    # Mock file
    mock_file = MagicMock()
    mock_file.filename = "test.mp3"
    mock_file.read = AsyncMock(return_value=b"fake audio content")

    # Mock music record
    mock_music = MagicMock()
    mock_music.id = "music_456"

    # Mock request with arq_pool
    mock_request = MagicMock()
    mock_arq_pool = AsyncMock()
    mock_request.app.state.arq_pool = mock_arq_pool

    with (
        patch("app.routes.choreography.create_music_analysis") as mock_create,
        patch("app.routes.choreography.upload_file", side_effect=OSError("Upload failed")),
        patch("app.routes.choreography.update_music_analysis") as mock_update,
    ):
        # Setup mocks
        mock_create.return_value = mock_music

        # Mock tempfile
        mock_tmp = MagicMock()
        mock_tmp.name = "/tmp/test.mp3"
        mock_tmp.write = MagicMock()
        mock_tmp.__enter__ = MagicMock(return_value=mock_tmp)
        mock_tmp.__exit__ = MagicMock(return_value=False)

        with patch("app.routes.choreography.tempfile.NamedTemporaryFile", return_value=mock_tmp):
            # Call the route - should raise HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await upload_music(mock_request, mock_user, mock_db, mock_file)

            # Verify error details
            assert exc_info.value.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            assert "Upload failed" in exc_info.value.detail

            # Verify status was updated to failed
            mock_update.assert_called_once_with(mock_db, mock_music, status="failed")
