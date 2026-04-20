"""Tests for worker async task functions (process_video_task, detect_video_task, analyze_music_task)."""

from __future__ import annotations

import sys
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Mock aiobotocore before importing app.worker (which imports app.storage)
_mock_aiobotocore = MagicMock()
_mock_aiobotocore_session = MagicMock()
sys.modules["aiobotocore"] = _mock_aiobotocore
sys.modules["aiobotocore.session"] = _mock_aiobotocore_session


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_vast_result(**overrides):
    """Build a mock VastResult with sensible defaults."""
    result = MagicMock()
    result.video_key = "output/video.mp4"
    result.poses_key = None
    result.csv_key = None
    result.stats = {"fps": 30}
    result.metrics = None
    result.phases = None
    result.recommendations = None
    for k, v in overrides.items():
        setattr(result, k, v)
    return result


def _make_async_session_cm(mock_db):
    """Return an async context manager mock for async_session()."""
    cm = AsyncMock()
    cm.__aenter__.return_value = mock_db
    cm.__aexit__.return_value = False
    return cm


@pytest.fixture
def mock_valkey():
    """Fresh AsyncMock for Valkey connection, reused across all test classes."""
    return AsyncMock()


# ---------------------------------------------------------------------------
# process_video_task
# ---------------------------------------------------------------------------


class TestProcessVideoTask:
    """Tests for process_video_task."""

    @pytest.mark.asyncio
    async def test_success_basic(self, mock_valkey):
        """Successful processing returns 'Analysis complete!' with video_key."""
        from app.worker import process_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch(
                "app.vastai.client.process_video_remote_async", new_callable=AsyncMock
            ) as mock_remote,
            # async_session is imported inside process_video_task body
            patch("app.database.async_session", create=True) as mock_async_session,
        ):
            mock_db = AsyncMock()
            mock_async_session.return_value = _make_async_session_cm(mock_db)

            mock_remote.return_value = _make_vast_result()

            result = await process_video_task(
                ctx={},
                task_id="proc_1",
                video_key="input/video.mp4",
                person_click={"x": 100, "y": 200},
            )

        assert result["status"] == "Analysis complete!"
        assert result["video_path"] == "output/video.mp4"
        assert result["stats"] == {"fps": 30}

    @pytest.mark.asyncio
    async def test_cancellation_before_gpu_dispatch(self, mock_valkey):
        """Task returns 'cancelled' when cancelled before GPU dispatch."""
        from app.worker import process_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("app.worker.is_cancelled", return_value=True) as mock_cancelled,
            patch("app.worker.mark_cancelled", new_callable=AsyncMock) as mock_mark,
            patch(
                "app.vastai.client.process_video_remote_async", new_callable=AsyncMock
            ) as mock_remote,
            patch("app.database.async_session", create=True),
        ):
            result = await process_video_task(
                ctx={},
                task_id="proc_cancel_1",
                video_key="input/video.mp4",
                person_click={"x": 100, "y": 200},
            )

        assert result["status"] == "cancelled"
        mock_mark.assert_called_once()
        # GPU dispatch must NOT have been called
        mock_remote.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancellation_after_gpu_returns(self, mock_valkey):
        """Task returns 'cancelled' when cancelled after GPU processing completes."""
        from app.worker import process_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("app.worker.is_cancelled", new_callable=AsyncMock) as mock_cancelled,
            patch("app.worker.mark_cancelled", new_callable=AsyncMock) as mock_mark,
            patch(
                "app.vastai.client.process_video_remote_async", new_callable=AsyncMock
            ) as mock_remote,
            patch("app.database.async_session", create=True),
        ):
            mock_remote.return_value = _make_vast_result()
            # First call: not cancelled (before GPU). Second call: cancelled (after GPU).
            mock_cancelled.side_effect = [False, True]

            result = await process_video_task(
                ctx={},
                task_id="proc_cancel_2",
                video_key="input/video.mp4",
                person_click={"x": 100, "y": 200},
            )

        assert result["status"] == "cancelled"
        mock_remote.assert_called_once()
        mock_mark.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_vastai_exception(self, mock_valkey):
        """Non-network exception from Vast.ai is stored and re-raised."""
        from app.worker import process_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch(
                "app.vastai.client.process_video_remote_async", new_callable=AsyncMock
            ) as mock_remote,
            patch("app.worker.store_error", new_callable=AsyncMock) as mock_store_err,
            patch("app.database.async_session", create=True),
        ):
            # Use an error message that does NOT contain "timeout"/"connection"/"network"
            # to avoid triggering the Retry logic in the worker
            mock_remote.side_effect = RuntimeError("GPU out of memory")

            with pytest.raises(RuntimeError, match="GPU out of memory"):
                await process_video_task(
                    ctx={},
                    task_id="proc_err",
                    video_key="input/video.mp4",
                    person_click={"x": 100, "y": 200},
                )

        mock_store_err.assert_called_once()
        call_args = mock_store_err.call_args
        assert call_args[0][0] == "proc_err"
        assert "GPU out of memory" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_network_error_triggers_retry(self, mock_valkey):
        """Network errors raise arq.Retry with deferred backoff."""
        from app.worker import process_video_task
        from arq import Retry

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch(
                "app.vastai.client.process_video_remote_async", new_callable=AsyncMock
            ) as mock_remote,
            patch("app.worker.store_error", new_callable=AsyncMock),
            patch("app.database.async_session", create=True),
        ):
            mock_remote.side_effect = ConnectionError("Network unreachable")

            with pytest.raises(Retry):
                await process_video_task(
                    ctx={"job_try": 2},
                    task_id="proc_retry",
                    video_key="input/video.mp4",
                    person_click={"x": 100, "y": 200},
                )

    @pytest.mark.asyncio
    async def test_with_session_id_saves_results(self, mock_valkey):
        """When session_id is provided, results are saved to DB."""
        from app.worker import process_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch(
                "app.vastai.client.process_video_remote_async", new_callable=AsyncMock
            ) as mock_remote,
            patch("app.database.async_session", create=True) as mock_async_session,
            patch("app.crud.session.get_by_id", new_callable=AsyncMock) as mock_get_session,
            patch("app.crud.session.update_session_analysis", new_callable=AsyncMock),
            patch(
                "app.services.session_saver.save_analysis_results", new_callable=AsyncMock
            ) as mock_save,
        ):
            mock_remote.return_value = _make_vast_result(
                metrics=[{"name": "airtime", "value": 0.5}],
                phases=[{"phase": "takeoff", "frame": 10}],
                recommendations=["Good jump."],
            )

            mock_db = AsyncMock()
            mock_async_session.return_value = _make_async_session_cm(mock_db)

            # Mock session record
            mock_session = MagicMock()
            mock_session.element_type = "waltz_jump"
            mock_get_session.return_value = mock_session

            result = await process_video_task(
                ctx={},
                task_id="proc_session",
                video_key="input/video.mp4",
                person_click={"x": 100, "y": 200},
                session_id="session_42",
            )

        assert result["status"] == "Analysis complete!"
        mock_save.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_poses_key_downloads_and_samples(self, mock_valkey):
        """When vast_result has poses_key, poses are downloaded, sampled, and metrics computed."""
        from app.worker import process_video_task

        poses_data = np.random.rand(50, 17, 3).astype(np.float32)

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch(
                "app.vastai.client.process_video_remote_async", new_callable=AsyncMock
            ) as mock_remote,
            patch("app.worker.download_file") as mock_download,
            patch("numpy.load", return_value=poses_data),
            patch("app.database.async_session", create=True),
        ):
            mock_remote.return_value = _make_vast_result(
                poses_key="output/poses.npy",
                stats={"fps": 30.0},
            )

            result = await process_video_task(
                ctx={},
                task_id="proc_poses",
                video_key="input/video.mp4",
                person_click={"x": 100, "y": 200},
            )

        assert result["status"] == "Analysis complete!"
        mock_download.assert_called_once()

    @pytest.mark.parametrize(
        "remote_side_effect, expect_raises",
        [
            (None, False),  # success path
            (RuntimeError("boom"), True),  # error path
        ],
        ids=["success", "error"],
    )
    @pytest.mark.asyncio
    async def test_valkey_closed(self, mock_valkey, remote_side_effect, expect_raises):
        """Valkey connection is always closed, regardless of success or error."""
        from app.worker import process_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch(
                "app.vastai.client.process_video_remote_async", new_callable=AsyncMock
            ) as mock_remote,
            patch("app.worker.store_error", new_callable=AsyncMock),
            patch("app.database.async_session", create=True),
        ):
            mock_remote.return_value = _make_vast_result()
            if remote_side_effect is not None:
                mock_remote.side_effect = remote_side_effect

            if expect_raises:
                with pytest.raises(RuntimeError, match="boom"):
                    await process_video_task(
                        ctx={},
                        task_id="proc_close",
                        video_key="input/video.mp4",
                        person_click={"x": 100, "y": 200},
                    )
            else:
                await process_video_task(
                    ctx={},
                    task_id="proc_close",
                    video_key="input/video.mp4",
                    person_click={"x": 100, "y": 200},
                )

        mock_valkey.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_progress_updates_sent(self, mock_valkey):
        """Progress and events are published at expected stages."""
        from app.worker import process_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch(
                "app.vastai.client.process_video_remote_async", new_callable=AsyncMock
            ) as mock_remote,
            patch("app.worker.update_progress", new_callable=AsyncMock) as mock_progress,
            patch("app.worker.publish_task_event", new_callable=AsyncMock),
            patch("app.database.async_session", create=True),
        ):
            mock_remote.return_value = _make_vast_result()

            await process_video_task(
                ctx={},
                task_id="proc_progress",
                video_key="input/video.mp4",
                person_click={"x": 100, "y": 200},
            )

        progress_values = [c[0][1] for c in mock_progress.call_args_list]
        assert 0.0 in progress_values
        assert 0.1 in progress_values
        assert 0.7 in progress_values
        assert 1.0 in progress_values

    @pytest.mark.asyncio
    async def test_element_type_fetched_from_session(self, mock_valkey):
        """element_type is fetched from DB session when session_id is given."""
        from app.worker import process_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch(
                "app.vastai.client.process_video_remote_async", new_callable=AsyncMock
            ) as mock_remote,
            patch("app.database.async_session", create=True) as mock_async_session,
            patch("app.crud.session.get_by_id", new_callable=AsyncMock) as mock_get_session,
        ):
            mock_remote.return_value = _make_vast_result()

            mock_db = AsyncMock()
            mock_async_session.return_value = _make_async_session_cm(mock_db)

            mock_session = MagicMock()
            mock_session.element_type = "axel"
            mock_get_session.return_value = mock_session

            await process_video_task(
                ctx={},
                task_id="proc_element",
                video_key="input/video.mp4",
                person_click={"x": 100, "y": 200},
                session_id="session_99",
            )

            mock_remote.assert_called_once()
            call_kwargs = mock_remote.call_args[1]
            assert call_kwargs["element_type"] == "axel"

    @pytest.mark.asyncio
    async def test_pose_preparation_failure_continues(self, mock_valkey):
        """When _compute_frame_metrics raises, the exception is caught and task completes."""
        from app.worker import process_video_task

        poses_data = np.random.rand(50, 17, 3).astype(np.float32)

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch(
                "app.vastai.client.process_video_remote_async", new_callable=AsyncMock
            ) as mock_remote,
            patch("app.worker.download_file"),
            patch("numpy.load", return_value=poses_data),
            patch("app.worker._compute_frame_metrics", side_effect=RuntimeError("metric boom")),
            patch("app.database.async_session", create=True),
        ):
            mock_remote.return_value = _make_vast_result(
                poses_key="output/poses.npy",
                stats={"fps": 30.0},
            )

            # Should NOT raise -- the exception is caught inside the try block
            result = await process_video_task(
                ctx={},
                task_id="proc_pose_fail",
                video_key="input/video.mp4",
                person_click={"x": 100, "y": 200},
            )

        assert result["status"] == "Analysis complete!"

    @pytest.mark.asyncio
    async def test_save_session_results_failure_continues(self, mock_valkey):
        """When save_analysis_results raises, the exception is caught and task still completes."""
        from app.worker import process_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch(
                "app.vastai.client.process_video_remote_async", new_callable=AsyncMock
            ) as mock_remote,
            patch("app.database.async_session", create=True) as mock_async_session,
            patch("app.crud.session.get_by_id", new_callable=AsyncMock),
            patch(
                "app.services.session_saver.save_analysis_results",
                side_effect=RuntimeError("save boom"),
            ),
        ):
            mock_remote.return_value = _make_vast_result(
                metrics=[{"name": "airtime", "value": 0.5}],
                phases=[{"phase": "takeoff", "frame": 10}],
                recommendations=["Good jump."],
            )

            mock_db = AsyncMock()
            mock_async_session.return_value = _make_async_session_cm(mock_db)

            mock_session = MagicMock()
            mock_session.element_type = "waltz_jump"

            # get_by_id is imported inside the function, patch at the import site
            with patch("app.crud.session.get_by_id", return_value=mock_session):
                result = await process_video_task(
                    ctx={},
                    task_id="proc_save_fail",
                    video_key="input/video.mp4",
                    person_click={"x": 100, "y": 200},
                    session_id="session_42",
                )

        assert result["status"] == "Analysis complete!"

    @pytest.mark.asyncio
    async def test_publish_error_event_failure(self, mock_valkey):
        """When publish_task_event raises on the error path, the original error is still raised."""
        from app.worker import process_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch(
                "app.vastai.client.process_video_remote_async", new_callable=AsyncMock
            ) as mock_remote,
            patch("app.worker.store_error", new_callable=AsyncMock),
            # publish_task_event succeeds during normal flow, fails only in error handler
            patch("app.worker.publish_task_event", new_callable=AsyncMock) as mock_publish,
            patch("app.database.async_session", create=True),
        ):
            mock_remote.side_effect = RuntimeError("GPU out of memory")

            # First 2 calls succeed (0.0 Starting, 0.1 Dispatching), 3rd call (error event) fails
            mock_publish.side_effect = [None, None, ConnectionError("event bus down")]

            # The original RuntimeError should still be raised even though
            # publish_task_event also fails
            with pytest.raises(RuntimeError, match="GPU out of memory"):
                await process_video_task(
                    ctx={},
                    task_id="proc_pub_err",
                    video_key="input/video.mp4",
                    person_click={"x": 100, "y": 200},
                )

    @pytest.mark.asyncio
    async def test_with_pose_data_and_frame_metrics(self, mock_valkey):
        """When poses_key is set AND session_id has metrics, update_session_analysis is called."""
        from app.worker import process_video_task

        poses_data = np.random.rand(50, 17, 3).astype(np.float32)

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch(
                "app.vastai.client.process_video_remote_async", new_callable=AsyncMock
            ) as mock_remote,
            patch("app.worker.download_file"),
            patch("numpy.load", return_value=poses_data),
            patch("app.database.async_session", create=True) as mock_async_session,
            patch("app.crud.session.get_by_id", new_callable=AsyncMock) as mock_get_session,
            patch(
                "app.crud.session.update_session_analysis", new_callable=AsyncMock
            ) as mock_update_analysis,
            patch(
                "app.services.session_saver.save_analysis_results", new_callable=AsyncMock
            ) as mock_save,
        ):
            mock_remote.return_value = _make_vast_result(
                poses_key="output/poses.npy",
                stats={"fps": 30.0},
                metrics=[{"name": "airtime", "value": 0.5}],
                phases=[{"phase": "takeoff", "frame": 10}],
                recommendations=["Good jump."],
            )

            mock_db = AsyncMock()
            mock_async_session.return_value = _make_async_session_cm(mock_db)

            mock_session = MagicMock()
            mock_session.element_type = "axel"
            mock_get_session.return_value = mock_session

            result = await process_video_task(
                ctx={},
                task_id="proc_pose_metrics",
                video_key="input/video.mp4",
                person_click={"x": 100, "y": 200},
                session_id="session_42",
            )

        assert result["status"] == "Analysis complete!"
        # update_session_analysis should be called because pose_data is truthy
        mock_update_analysis.assert_called_once()


# ---------------------------------------------------------------------------
# detect_video_task
#
# NOTE: detect_video_task imports cv2, PoseExtractor, DeviceConfig,
# get_video_meta, render_person_preview, download_file inside its body.
# We mock asyncio.to_thread to intercept all synchronous calls, and mock
# the module-level objects that the function references.
# ---------------------------------------------------------------------------


class TestDetectVideoTask:
    """Tests for detect_video_task."""

    @pytest.fixture
    def empty_detect_thread(self):
        """asyncio.to_thread mock that returns no persons."""

        async def _mock_to_thread(func, *args, **kwargs):
            func_name = getattr(func, "__name__", str(func))
            if "download" in func_name or "download" in str(func):
                return None
            if "preview_persons" in func_name or "preview" in func_name:
                return ([], {})
            return None

        return _mock_to_thread

    @pytest.fixture
    def single_person_thread(self):
        """asyncio.to_thread mock that returns one person."""

        async def _mock_to_thread(func, *args, **kwargs):
            func_name = getattr(func, "__name__", str(func))
            if "download" in func_name or "download" in str(func):
                return None
            if "preview_persons" in func_name or "preview" in func_name:
                return (
                    [
                        {
                            "track_id": 0,
                            "hits": 50,
                            "bbox": [0.1, 0.1, 0.5, 0.8],
                            "mid_hip": [0.3, 0.6],
                        }
                    ],
                    {"fps": 30.0},
                )
            return None

        return _mock_to_thread

    @pytest.fixture
    def mock_cv2_ok(self):
        """Mock cv2 operations for detect_video_task tests (successful read + encode)."""
        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, b"fake_frame")
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.imencode.return_value = (True, b"fake_png_bytes")
        return mock_cv2

    @pytest.mark.asyncio
    async def test_no_persons_found(self, mock_valkey, empty_detect_thread):
        """No persons found -> returns empty list and appropriate Russian message."""
        from app.worker import detect_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("app.worker.get_settings"),
            patch("asyncio.to_thread", side_effect=empty_detect_thread),
        ):
            result = await detect_video_task(
                ctx={},
                task_id="det_empty",
                video_key="input/video.mp4",
                tracking="auto",
            )

        assert result["persons"] == []
        assert result["auto_click"] is None
        assert result["preview_image"] == ""
        assert "не найдены" in result["status"].lower()

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_valkey):
        """Exceptions are stored and re-raised."""
        from app.worker import detect_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("app.worker.get_settings"),
            patch("app.worker.store_error", new_callable=AsyncMock) as mock_store_err,
        ):
            with patch("asyncio.to_thread", side_effect=RuntimeError("R2 connection refused")):
                with pytest.raises(RuntimeError, match="R2 connection refused"):
                    await detect_video_task(
                        ctx={},
                        task_id="det_err",
                        video_key="input/video.mp4",
                        tracking="auto",
                    )

        mock_store_err.assert_called_once()

    @pytest.mark.parametrize(
        "to_thread_side_effect, expect_raises",
        [
            (None, False),  # success path (uses empty_detect_thread)
            (RuntimeError("fail"), True),  # error path
        ],
        ids=["success", "error"],
    )
    @pytest.mark.asyncio
    async def test_valkey_closed(
        self, mock_valkey, empty_detect_thread, to_thread_side_effect, expect_raises
    ):
        """Valkey is closed regardless of success or error."""
        from app.worker import detect_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("app.worker.get_settings"),
            patch("app.worker.store_error", new_callable=AsyncMock),
        ):
            side_effect = (
                to_thread_side_effect if to_thread_side_effect is not None else empty_detect_thread
            )

            with patch("asyncio.to_thread", side_effect=side_effect):
                if expect_raises:
                    with pytest.raises(RuntimeError, match="fail"):
                        await detect_video_task(
                            ctx={},
                            task_id="det_close",
                            video_key="input/video.mp4",
                            tracking="auto",
                        )
                else:
                    await detect_video_task(
                        ctx={},
                        task_id="det_close",
                        video_key="input/video.mp4",
                        tracking="auto",
                    )

        mock_valkey.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_result_contains_expected_keys_when_empty(self, mock_valkey, empty_detect_thread):
        """Result dict for empty detection has expected top-level keys."""
        from app.worker import detect_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("app.worker.get_settings"),
            patch("asyncio.to_thread", side_effect=empty_detect_thread),
        ):
            result = await detect_video_task(
                ctx={},
                task_id="det_keys",
                video_key="input/video.mp4",
                tracking="auto",
            )

        assert "persons" in result
        assert "preview_image" in result
        assert "video_key" in result
        assert "auto_click" in result
        assert "status" in result

    @pytest.mark.asyncio
    async def test_single_person_detected(self, mock_valkey, single_person_thread, mock_cv2_ok):
        """Single person detected -> auto_click is set, preview_image is non-empty."""
        from app.worker import detect_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("app.worker.get_settings"),
            patch("asyncio.to_thread", side_effect=single_person_thread),
            patch.dict("sys.modules", {"cv2": mock_cv2_ok}),
            patch("src.web_helpers.render_person_preview", return_value=b"fake_frame"),
            patch(
                "src.utils.video.get_video_meta", return_value=MagicMock(width=1920, height=1080)
            ),
            patch("builtins.open", MagicMock()),
        ):
            result = await detect_video_task(
                ctx={},
                task_id="det_single",
                video_key="input/video.mp4",
                tracking="auto",
            )

        assert result["auto_click"] is not None
        assert "x" in result["auto_click"]
        assert "y" in result["auto_click"]
        assert len(result["preview_image"]) > 0
        assert "1 человек" in result["status"]
        assert len(result["persons"]) == 1

    @pytest.mark.asyncio
    async def test_multiple_persons_detected(self, mock_valkey, mock_cv2_ok):
        """Multiple persons detected -> auto_click is None, status mentions count."""

        async def mock_to_thread(func, *args, **kwargs):
            func_name = getattr(func, "__name__", str(func))
            if "download" in func_name or "download" in str(func):
                return None
            if "preview_persons" in func_name or "preview" in func_name:
                return (
                    [
                        {
                            "track_id": 0,
                            "hits": 50,
                            "bbox": [0.1, 0.1, 0.5, 0.8],
                            "mid_hip": [0.3, 0.6],
                        },
                        {
                            "track_id": 1,
                            "hits": 30,
                            "bbox": [0.6, 0.1, 0.9, 0.8],
                            "mid_hip": [0.7, 0.6],
                        },
                    ],
                    {"fps": 30.0},
                )
            return None

        from app.worker import detect_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("app.worker.get_settings"),
            patch("asyncio.to_thread", side_effect=mock_to_thread),
            patch.dict("sys.modules", {"cv2": mock_cv2_ok}),
            patch("src.web_helpers.render_person_preview", return_value=b"fake_frame"),
            patch(
                "src.utils.video.get_video_meta", return_value=MagicMock(width=1920, height=1080)
            ),
            patch("builtins.open", MagicMock()),
        ):
            result = await detect_video_task(
                ctx={},
                task_id="det_multi",
                video_key="input/video.mp4",
                tracking="auto",
            )

        assert result["auto_click"] is None
        assert "2" in result["status"]
        assert len(result["persons"]) == 2

    @pytest.mark.asyncio
    async def test_error_publish_failure_still_raises(self, mock_valkey):
        """When publish_task_event raises in the error handler, original error is still raised."""
        from app.worker import detect_video_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("app.worker.get_settings"),
            patch("app.worker.store_error", new_callable=AsyncMock),
            # publish_task_event: first 3 calls succeed (progress), 4th fails (error handler)
            patch("app.worker.publish_task_event", new_callable=AsyncMock) as mock_publish,
        ):
            # First 2 calls succeed (0.0, 0.1), 3rd call (error handler) raises
            mock_publish.side_effect = [None, None, ConnectionError("event bus down")]

            with patch("asyncio.to_thread", side_effect=RuntimeError("R2 connection refused")):
                with pytest.raises(RuntimeError, match="R2 connection refused"):
                    await detect_video_task(
                        ctx={},
                        task_id="det_pub_err",
                        video_key="input/video.mp4",
                        tracking="auto",
                    )

    @pytest.mark.asyncio
    async def test_video_read_failure(self, mock_valkey, single_person_thread):
        """When VideoCapture.read() fails, RuntimeError is raised."""
        from app.worker import detect_video_task

        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)  # Video read failure
        mock_cv2.VideoCapture.return_value = mock_cap

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("app.worker.get_settings"),
            patch("asyncio.to_thread", side_effect=single_person_thread),
            patch.dict("sys.modules", {"cv2": mock_cv2}),
            patch("src.web_helpers.render_person_preview", return_value=b"fake_frame"),
            patch(
                "src.utils.video.get_video_meta", return_value=MagicMock(width=1920, height=1080)
            ),
            patch("builtins.open", MagicMock()),
        ):
            with pytest.raises(RuntimeError, match="Failed to read video frame"):
                await detect_video_task(
                    ctx={},
                    task_id="det_read_fail",
                    video_key="input/video.mp4",
                    tracking="auto",
                )

    @pytest.mark.asyncio
    async def test_imencode_failure(self, mock_valkey, single_person_thread):
        """When cv2.imencode fails, RuntimeError is raised."""
        from app.worker import detect_video_task

        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, b"fake_frame")
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.imencode.return_value = (False, None)  # imencode failure

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("app.worker.get_settings"),
            patch("asyncio.to_thread", side_effect=single_person_thread),
            patch.dict("sys.modules", {"cv2": mock_cv2}),
            patch("src.web_helpers.render_person_preview", return_value=b"fake_frame"),
            patch(
                "src.utils.video.get_video_meta", return_value=MagicMock(width=1920, height=1080)
            ),
            patch("builtins.open", MagicMock()),
        ):
            with pytest.raises(RuntimeError, match="Failed to encode preview image"):
                await detect_video_task(
                    ctx={},
                    task_id="det_encode_fail",
                    video_key="input/video.mp4",
                    tracking="auto",
                )


# ---------------------------------------------------------------------------
# analyze_music_task — additional coverage beyond test_music_worker.py
# ---------------------------------------------------------------------------


class TestAnalyzeMusicTaskExtended:
    """Extended tests for analyze_music_task not covered in test_music_worker.py."""

    @pytest.mark.asyncio
    async def test_music_not_found_raises(self, mock_valkey):
        """When music record is not in DB, RuntimeError is raised and status set to failed."""
        from app.worker import analyze_music_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("asyncio.to_thread", return_value="a" * 32),
            patch("app.database.async_session_factory") as mock_session_factory,
            patch("app.crud.choreography.get_music_analysis_by_id", return_value=None) as mock_get,
            patch(
                "app.crud.choreography.update_music_analysis", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_db = AsyncMock()
            mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(RuntimeError, match="not found"):
                await analyze_music_task(
                    ctx={},
                    music_id="nonexistent",
                    r2_key="music/test.mp3",
                )

    @pytest.mark.asyncio
    async def test_fingerprint_failure_raises(self, mock_valkey):
        """When fingerprint computation returns None, RuntimeError is raised."""
        from app.worker import analyze_music_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("asyncio.to_thread", return_value=None),
            patch("app.database.async_session_factory") as mock_session_factory,
            patch("app.crud.choreography.get_music_analysis_by_id") as mock_get,
            patch(
                "app.crud.choreography.update_music_analysis", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_music = MagicMock()
            mock_get.return_value = mock_music

            mock_db = AsyncMock()
            mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(RuntimeError, match="Failed to compute fingerprint"):
                await analyze_music_task(
                    ctx={},
                    music_id="music_fp_fail",
                    r2_key="music/test.mp3",
                )

            mock_update.assert_called_with(mock_db, mock_music, status="failed")

    @pytest.mark.asyncio
    async def test_analysis_failure_sets_status_failed(self, mock_valkey):
        """When analyze_music_sync raises, status is set to 'failed' in DB."""
        from app.worker import analyze_music_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("asyncio.to_thread") as mock_to_thread,
            patch("app.database.async_session_factory") as mock_session_factory,
            patch("app.crud.choreography.get_music_analysis_by_id") as mock_get,
            patch("app.crud.choreography.find_music_by_fingerprint", return_value=None),
            patch(
                "app.crud.choreography.update_music_analysis", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_music = MagicMock()
            mock_get.return_value = mock_music

            call_count = 0

            def side_effect(func, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return "c" * 32
                raise RuntimeError("librosa crashed")

            mock_to_thread.side_effect = side_effect

            mock_db = AsyncMock()
            mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(RuntimeError, match="librosa crashed"):
                await analyze_music_task(
                    ctx={},
                    music_id="music_analysis_fail",
                    r2_key="music/test.mp3",
                )

            mock_update.assert_called_with(mock_db, mock_music, status="failed")

    @pytest.mark.asyncio
    async def test_valkey_closed_on_error(self, mock_valkey):
        """Valkey connection is closed even when analyze_music_task raises."""
        from app.worker import analyze_music_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("asyncio.to_thread", return_value=None),
            patch("app.database.async_session_factory") as mock_session_factory,
            patch("app.crud.choreography.get_music_analysis_by_id") as mock_get,
            patch("app.crud.choreography.update_music_analysis", new_callable=AsyncMock),
        ):
            mock_music = MagicMock()
            mock_get.return_value = mock_music

            mock_db = AsyncMock()
            mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(RuntimeError, match="Failed to compute fingerprint"):
                await analyze_music_task(
                    ctx={},
                    music_id="music_close_err",
                    r2_key="music/test.mp3",
                )

        mock_valkey.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_music_status_failed_exception_swallowed(self, mock_valkey):
        """When update_music_analysis raises in the error handler, the original error is still raised."""
        from app.worker import analyze_music_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("asyncio.to_thread") as mock_to_thread,
            patch("app.database.async_session_factory") as mock_session_factory,
            patch("app.crud.choreography.get_music_analysis_by_id") as mock_get,
            patch("app.crud.choreography.find_music_by_fingerprint", return_value=None),
            patch(
                "app.crud.choreography.update_music_analysis", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_music = MagicMock()
            mock_get.return_value = mock_music

            # First to_thread call = fingerprint, second = analyze_music_sync which raises
            call_count = 0

            def to_thread_side_effect(func, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return "d" * 32
                raise RuntimeError("original analysis error")

            mock_to_thread.side_effect = to_thread_side_effect

            mock_db = AsyncMock()
            mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

            # Make update_music_analysis fail when called from the error handler
            # The error handler re-imports update_music_analysis from app.crud.choreography
            with patch(
                "app.crud.choreography.update_music_analysis", side_effect=RuntimeError("db locked")
            ):
                with pytest.raises(RuntimeError, match="original analysis error"):
                    await analyze_music_task(
                        ctx={},
                        music_id="music_status_fail",
                        r2_key="music/test.mp3",
                    )

    @pytest.mark.asyncio
    async def test_duplicate_fingerprint_returns_cached_results(self, mock_valkey):
        """When a duplicate fingerprint is found, analysis results are copied and returned."""
        from app.worker import analyze_music_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("asyncio.to_thread", return_value="e" * 32),
            patch("app.database.async_session_factory") as mock_session_factory,
            patch("app.crud.choreography.get_music_analysis_by_id") as mock_get,
            patch("app.crud.choreography.find_music_by_fingerprint") as mock_find_dup,
            patch(
                "app.crud.choreography.update_music_analysis", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_music = MagicMock()
            mock_music.audio_url = "/files/music/test.mp3"
            mock_get.return_value = mock_music

            mock_duplicate = MagicMock()
            mock_duplicate.id = "other_music_id"
            mock_duplicate.fingerprint = "e" * 32
            mock_duplicate.duration_sec = 180.5
            mock_duplicate.bpm = 120.0
            mock_find_dup.return_value = mock_duplicate

            mock_db = AsyncMock()
            mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await analyze_music_task(
                ctx={},
                music_id="music_dup",
                r2_key="music/test.mp3",
            )

        assert result["status"] == "completed"
        assert result["duplicate_of"] == "other_music_id"
        assert result["bpm"] == 120.0
        assert result["duration_sec"] == 180.5

    @pytest.mark.asyncio
    async def test_full_analysis_success(self, mock_valkey):
        """When no duplicate found, full music analysis runs and returns results."""
        from app.worker import analyze_music_task

        with (
            patch("app.worker.get_valkey_client", return_value=mock_valkey),
            patch("asyncio.to_thread") as mock_to_thread,
            patch("app.database.async_session_factory") as mock_session_factory,
            patch("app.crud.choreography.get_music_analysis_by_id") as mock_get,
            patch("app.crud.choreography.find_music_by_fingerprint", return_value=None),
            patch(
                "app.crud.choreography.update_music_analysis", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_music = MagicMock()
            mock_get.return_value = mock_music

            call_count = 0

            def to_thread_side_effect(func, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return None  # download_file
                if call_count == 2:
                    return "f" * 32  # fingerprint
                return {  # analyze_music_sync result
                    "duration_sec": 200.0,
                    "bpm": 140.5,
                    "peaks": [10, 50, 100],
                    "structure": [{"start": 0, "end": 30, "label": "intro"}],
                    "energy_curve": [0.1, 0.2, 0.3],
                }

            mock_to_thread.side_effect = to_thread_side_effect

            mock_db = AsyncMock()
            mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await analyze_music_task(
                ctx={},
                music_id="music_full",
                r2_key="music/test.mp3",
            )

        assert result["status"] == "completed"
        assert result["bpm"] == 140.5
        assert result["duration_sec"] == 200.0
        assert "duplicate_of" not in result


# ---------------------------------------------------------------------------
# startup / shutdown
# ---------------------------------------------------------------------------


class TestWorkerLifecycle:
    """Tests for worker startup and shutdown hooks."""

    @pytest.mark.asyncio
    async def test_startup_completes_without_error(self):
        """startup() is a no-op that just logs; it should not raise."""
        from app.worker import startup

        await startup(ctx={})

    @pytest.mark.asyncio
    async def test_shutdown_closes_pool(self):
        """shutdown() closes the redis/valkey pool from ctx if present."""
        from app.worker import shutdown

        mock_pool = AsyncMock()
        ctx = {"redis": mock_pool}

        await shutdown(ctx)

        mock_pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_without_pool(self):
        """shutdown() does nothing when ctx has no 'redis' key."""
        from app.worker import shutdown

        mock_pool = AsyncMock()
        await shutdown(ctx={})
        mock_pool.close.assert_not_awaited()
