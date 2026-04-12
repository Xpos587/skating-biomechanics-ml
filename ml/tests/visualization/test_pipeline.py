"""Tests for unified visualization pipeline."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from skating_ml.visualization.pipeline import VizPipeline


def _fake_meta(w=640, h=480, fps=30, num_frames=10):
    return SimpleNamespace(width=w, height=h, fps=fps, num_frames=num_frames)


class TestVizPipelineInit:
    def test_minimal_init(self):
        meta = _fake_meta()
        poses = np.random.rand(10, 17, 2).astype(np.float32)
        pipe = VizPipeline(meta=meta, poses_norm=poses)
        assert pipe.layer == 0
        assert len(pipe.layers) == 0

    def test_layer_1_builds_no_layers(self):
        meta = _fake_meta()
        poses = np.random.rand(10, 17, 2).astype(np.float32)
        pipe = VizPipeline(meta=meta, poses_norm=poses, layer=1)
        assert len(pipe.layers) == 0

    def test_layer_2_adds_axis(self):
        meta = _fake_meta()
        poses = np.random.rand(10, 17, 2).astype(np.float32)
        pipe1 = VizPipeline(meta=meta, poses_norm=poses, layer=1)
        pipe2 = VizPipeline(meta=meta, poses_norm=poses, layer=2)
        assert len(pipe1.layers) == 0
        assert len(pipe2.layers) == 1

    def test_with_poses_3d(self):
        meta = _fake_meta()
        poses = np.random.rand(10, 17, 2).astype(np.float32)
        poses_3d = np.random.rand(10, 17, 3).astype(np.float32)
        pipe = VizPipeline(meta=meta, poses_norm=poses, poses_3d=poses_3d)
        assert pipe.poses_3d is not None


class TestVizPipelineBuildLayers:
    def test_rebuild_layers_changes_count(self):
        meta = _fake_meta()
        poses = np.random.rand(10, 17, 2).astype(np.float32)
        pipe = VizPipeline(meta=meta, poses_norm=poses, layer=0)
        assert len(pipe.layers) == 0
        pipe.layer = 2
        pipe.build_layers()
        assert len(pipe.layers) == 1


class TestVizPipelineIntegration:
    """Integration tests verifying full render + export cycle."""

    def test_render_all_frames(self):
        """Render 10 frames with poses, verify no crash."""
        meta = _fake_meta(num_frames=10)
        poses = np.zeros((10, 17, 2), dtype=np.float32)
        cx = 0.5
        # Set up hip joints
        poses[:, 0] = cx - 0.02  # LHIP x
        poses[:, 1] = cx + 0.02  # RHIP x
        poses[:, 6] = 0.6  # hip y (LHIP is index 6 in H36Key)
        poses[:, 7] = 0.6  # RHIP y
        # Set up shoulder joints
        poses[:, 5] = cx - 0.015  # LSHOULDER x (index 5)
        poses[:, 8] = cx + 0.015  # RSHOULDER x (index 8)
        poses[:, 11] = 0.35  # shoulder y (LSHOULDER is index 11)
        poses[:, 12] = 0.35  # RSHOULDER y

        pipe = VizPipeline(meta=meta, poses_norm=poses, layer=2)

        pose_idx = 0
        for frame_idx in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            current, pose_idx = pipe.find_pose_idx(frame_idx, pose_idx)
            frame, _ = pipe.render_frame(frame, frame_idx, current)
            pipe.draw_frame_counter(frame, frame_idx)
            pipe.collect_export_data(frame_idx, current)

        assert len(pipe.export_frames) == 10

    def test_save_exports_creates_files(self, tmp_path):
        """save_exports creates .npy and .csv files."""
        meta = _fake_meta(num_frames=5)
        poses = np.random.rand(5, 17, 2).astype(np.float32)
        pipe = VizPipeline(meta=meta, poses_norm=poses, layer=0)

        for i in range(5):
            pipe.collect_export_data(i, i)

        out = tmp_path / "test.mp4"
        result = pipe.save_exports(out)

        assert result["poses_path"] is not None
        assert result["csv_path"] is not None
        assert Path(result["poses_path"]).exists()
        assert Path(result["csv_path"]).exists()

    def test_no_pose_frame_does_not_crash(self):
        """Frames with no matching pose should not crash."""
        meta = _fake_meta(num_frames=5)
        # Only 2 poses for 5 frames
        poses = np.random.rand(2, 17, 2).astype(np.float32)
        pipe = VizPipeline(meta=meta, poses_norm=poses, layer=1, frame_indices=np.array([0, 3]))

        pose_idx = 0
        for frame_idx in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            current, pose_idx = pipe.find_pose_idx(frame_idx, pose_idx)
            frame, _ = pipe.render_frame(frame, frame_idx, current)

        # Only frames 0 and 3 had poses
        assert len(pipe.export_frames) <= 2


# ------------------------------------------------------------------
# Tests for unified pose preparation pipeline
# ------------------------------------------------------------------


def _fake_extraction(n=10):
    """Create a fake TrackedExtraction result."""
    extraction = mock.MagicMock()
    extraction.poses = np.random.rand(n, 17, 3).astype(np.float32)
    extraction.frame_indices = np.arange(n)
    extraction.valid_mask.return_value = np.ones(n, dtype=bool)
    return extraction


class TestPreparePoses:
    def test_returns_prepared_poses(self):
        """prepare_poses returns PreparedPoses with correct shapes."""
        from skating_ml.visualization.pipeline import PreparedPoses, prepare_poses

        with (
            mock.patch(
                "skating_ml.visualization.pipeline.get_video_meta", return_value=_fake_meta()
            ),
            mock.patch("skating_ml.visualization.pipeline.RTMPoseExtractor") as MockExt,
            mock.patch("skating_ml.visualization.pipeline.CorrectiveLens") as MockLens,
            mock.patch("skating_ml.visualization.pipeline.ONNXPoseExtractor") as MockOnnx,
            mock.patch(
                "skating_ml.visualization.pipeline._resolve_model_3d",
                return_value=Path("model.onnx"),
            ),
        ):
            MockExt.return_value.extract_video_tracked.return_value = _fake_extraction()
            MockLens.return_value.correct_sequence.return_value = (
                np.random.rand(10, 17, 2).astype(np.float32) * 0.5 + 0.25,
                np.random.rand(10, 17, 3).astype(np.float32),
            )
            MockOnnx.return_value.estimate_3d.return_value = np.random.rand(10, 17, 3).astype(
                np.float32
            )

            result = prepare_poses(Path("test.mp4"))

        assert isinstance(result, PreparedPoses)
        assert result.poses_norm.shape == (10, 17, 2)
        assert result.poses_px.shape == (10, 17, 3)
        assert result.poses_3d is not None
        assert result.poses_3d.shape == (10, 17, 3)
        assert result.n_valid == 10
        assert result.n_total == 10

    def test_no_corrective_lens_when_disabled(self):
        """When use_corrective_lens=False, CorrectiveLens is not called."""
        from skating_ml.visualization.pipeline import prepare_poses

        with (
            mock.patch(
                "skating_ml.visualization.pipeline.get_video_meta", return_value=_fake_meta()
            ),
            mock.patch("skating_ml.visualization.pipeline.RTMPoseExtractor") as MockExt,
            mock.patch("skating_ml.visualization.pipeline.CorrectiveLens") as MockLens,
            mock.patch("skating_ml.visualization.pipeline.ONNXPoseExtractor") as MockONNX,
            mock.patch(
                "skating_ml.visualization.pipeline._resolve_model_3d",
                return_value=Path("model.onnx"),
            ),
        ):
            MockExt.return_value.extract_video_tracked.return_value = _fake_extraction()
            MockONNX.return_value.estimate_3d.return_value = np.random.rand(10, 17, 3).astype(
                np.float32
            )

            result = prepare_poses(Path("test.mp4"), use_corrective_lens=False)

        MockLens.assert_not_called()
        assert result.poses_3d is not None

    def test_no_3d_when_model_missing(self):
        """When model not found, poses_3d is None but poses_norm is still valid."""
        from skating_ml.visualization.pipeline import prepare_poses

        with (
            mock.patch(
                "skating_ml.visualization.pipeline.get_video_meta", return_value=_fake_meta()
            ),
            mock.patch("skating_ml.visualization.pipeline.RTMPoseExtractor") as MockExt,
            mock.patch("skating_ml.visualization.pipeline._resolve_model_3d", return_value=None),
        ):
            MockExt.return_value.extract_video_tracked.return_value = _fake_extraction()

            result = prepare_poses(Path("test.mp4"))

        assert result.poses_3d is None
        assert result.poses_norm.shape == (10, 17, 2)

    def test_gap_filling_when_frame_skip(self):
        """NaN frames from frame_skip are filled."""
        from skating_ml.visualization.pipeline import prepare_poses

        extraction = _fake_extraction(20)
        raw = np.full((20, 17, 3), np.nan, dtype=np.float32)
        for i in [0, 4, 8, 12, 16]:
            raw[i] = np.random.rand(17, 3).astype(np.float32)
        extraction.poses = raw
        extraction.valid_mask.return_value = np.array([i in [0, 4, 8, 12, 16] for i in range(20)])

        with (
            mock.patch(
                "skating_ml.visualization.pipeline.get_video_meta",
                return_value=_fake_meta(num_frames=20),
            ),
            mock.patch("skating_ml.visualization.pipeline.RTMPoseExtractor") as MockExt,
            mock.patch("skating_ml.visualization.pipeline.CorrectiveLens") as MockLens,
            mock.patch("skating_ml.visualization.pipeline.ONNXPoseExtractor") as MockOnnx,
            mock.patch(
                "skating_ml.visualization.pipeline._resolve_model_3d",
                return_value=Path("model.onnx"),
            ),
        ):
            MockExt.return_value.extract_video_tracked.return_value = extraction
            MockLens.return_value.correct_sequence.return_value = (
                np.random.rand(20, 17, 2).astype(np.float32) * 0.5 + 0.25,
                np.random.rand(20, 17, 3).astype(np.float32),
            )
            MockOnnx.return_value.estimate_3d.return_value = np.random.rand(20, 17, 3).astype(
                np.float32
            )

            result = prepare_poses(Path("test.mp4"), frame_skip=4)

        assert not np.isnan(result.poses_norm).any()
        assert result.n_valid == 5


class TestResolveModel3d:
    def test_explicit_path_returned(self, tmp_path):
        from skating_ml.visualization.pipeline import _resolve_model_3d

        model = tmp_path / "model.onnx"
        model.touch()
        result = _resolve_model_3d(model)
        assert result == model

    def test_none_when_not_found(self, tmp_path):
        from skating_ml.visualization.pipeline import _resolve_model_3d

        result = _resolve_model_3d(tmp_path / "nonexistent.onnx")
        assert result is None
