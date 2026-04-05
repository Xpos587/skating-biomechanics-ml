"""Tests for unified visualization pipeline."""

from pathlib import Path

import numpy as np

from src.visualization.pipeline import VizPipeline


def _fake_meta(w=640, h=480, fps=30, num_frames=10):
    from types import SimpleNamespace

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
