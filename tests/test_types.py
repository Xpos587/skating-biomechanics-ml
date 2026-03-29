"""Tests for core data types."""

import numpy as np
import pytest
from pathlib import Path

from src.types import (
    H36Key,
    BoundingBox,
    VideoMeta,
    ElementPhase,
    MetricResult,
    AnalysisReport,
    ReferenceData,
    H36M_INDICES,
    H36M_SKELETON_EDGES,
)


class TestH36Key:
    """Test H3.6M 17-keypoint enum."""

    def test_keypoint_indices(self):
        """Expected indices for key landmarks."""
        assert H36Key.HIP_CENTER == 0
        assert H36Key.LSHOULDER == 11
        assert H36Key.RSHOULDER == 14
        assert H36Key.LHIP == 4
        assert H36Key.RHIP == 1
        assert H36Key.HEAD == 10

    def test_all_17_keypoints(self):
        """Should have exactly 17 keypoints (H3.6M format)."""
        assert len(list(H36Key)) == 17

    def test_foot_keypoints(self):
        """Foot keypoints for edge detection."""
        assert H36Key.LFOOT == 6
        assert H36Key.RFOOT == 3
        # Backward compatibility aliases
        assert H36Key.LEFT_ANKLE == H36Key.LFOOT
        assert H36Key.RIGHT_ANKLE == H36Key.RFOOT

    def test_backward_compatibility_aliases(self):
        """BlazePose-style aliases should work."""
        # These should map to H3.6M equivalents
        assert H36Key.LEFT_SHOULDER == H36Key.LSHOULDER
        assert H36Key.RIGHT_SHOULDER == H36Key.RSHOULDER
        assert H36Key.LEFT_HIP == H36Key.LHIP
        assert H36Key.RIGHT_HIP == H36Key.RHIP
        # Deprecated keypoints map to nearest available
        assert H36Key.NOSE == H36Key.HEAD
        assert H36Key.LEFT_EAR == H36Key.HEAD


class TestH36MIndices:
    """Test H3.6M indices."""

    def test_h36m_size(self):
        """H3.6M should have 17 keypoints."""
        assert len(H36M_INDICES) == 17

    def test_h36m_contains_key_joints(self):
        """Should include major joints."""
        assert H36Key.HIP_CENTER in H36M_INDICES
        assert H36Key.LSHOULDER in H36M_INDICES
        assert H36Key.RSHOULDER in H36M_INDICES
        assert H36Key.LHIP in H36M_INDICES
        assert H36Key.RHIP in H36M_INDICES
        assert H36Key.LFOOT in H36M_INDICES


class TestSkeletonEdges:
    """Test skeleton edge definitions."""

    def test_edges_are_tuples(self):
        """All edges should be (joint_a, joint_b) tuples."""
        for edge in H36M_SKELETON_EDGES:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            assert isinstance(edge[0], (int, H36Key))
            assert isinstance(edge[1], (int, H36Key))

    def test_upper_body_edges(self):
        """Should have upper body connections."""
        # Torso connections
        assert (H36Key.SPINE, H36Key.THORAX) in H36M_SKELETON_EDGES
        assert (H36Key.THORAX, H36Key.NECK) in H36M_SKELETON_EDGES
        assert (H36Key.NECK, H36Key.HEAD) in H36M_SKELETON_EDGES

        # Arm connections (from thorax in H3.6M)
        assert (H36Key.THORAX, H36Key.LSHOULDER) in H36M_SKELETON_EDGES
        assert (H36Key.LSHOULDER, H36Key.LELBOW) in H36M_SKELETON_EDGES
        assert (H36Key.LELBOW, H36Key.LWRIST) in H36M_SKELETON_EDGES

    def test_lower_body_edges(self):
        """Should have lower body connections."""
        # Hip connections (from hip_center in H3.6M)
        assert (H36Key.HIP_CENTER, H36Key.LHIP) in H36M_SKELETON_EDGES
        assert (H36Key.HIP_CENTER, H36Key.RHIP) in H36M_SKELETON_EDGES

        # Leg connections
        assert (H36Key.LHIP, H36Key.LKNEE) in H36M_SKELETON_EDGES
        assert (H36Key.LKNEE, H36Key.LFOOT) in H36M_SKELETON_EDGES


class TestBoundingBox:
    """Test BoundingBox dataclass."""

    def test_bounding_box_properties(self):
        """Should calculate width, height, center correctly."""
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=120, confidence=0.9)

        assert bbox.width == 100
        assert bbox.height == 100
        assert bbox.center_x == 60
        assert bbox.center_y == 70

    def test_bounding_box_negative_coords(self):
        """Should handle negative coordinates."""
        bbox = BoundingBox(x1=-50, y1=-50, x2=50, y2=50, confidence=0.8)

        assert bbox.width == 100
        assert bbox.height == 100
        assert bbox.center_x == 0
        assert bbox.center_y == 0


class TestVideoMeta:
    """Test VideoMeta dataclass."""

    def test_video_meta_properties(self, tmp_path: Path):
        """Should calculate duration correctly."""
        meta = VideoMeta(
            path=tmp_path / "test.mp4",
            fps=30.0,
            width=1920,
            height=1080,
            num_frames=900,
        )

        assert meta.duration_sec == 30.0

    def test_video_meta_zero_fps(self, tmp_path: Path):
        """Should handle zero fps gracefully."""
        meta = VideoMeta(
            path=tmp_path / "test.mp4",
            fps=0.0,
            width=1920,
            height=1080,
            num_frames=900,
        )

        assert meta.duration_sec == 0.0


class TestElementPhase:
    """Test ElementPhase dataclass."""

    def test_jump_phase(self):
        """Jump phases should have takeoff != start."""
        phase = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=30,
            peak=45,
            landing=60,
            end=90,
        )

        assert phase.has_takeoff is True
        assert phase.airtime_frames == 30

    def test_airtime_calculation(self):
        """Should calculate airtime correctly."""
        phase = ElementPhase(
            name="toe_loop",
            start=0,
            takeoff=25,
            peak=40,
            landing=55,
            end=80,
        )

        assert phase.airtime_frames == 30
        assert phase.airtime_sec(fps=30.0) == pytest.approx(1.0)

    def test_step_element_no_takeoff(self):
        """Step elements should have takeoff == start."""
        phase = ElementPhase(
            name="three_turn",
            start=0,
            takeoff=0,
            peak=0,
            landing=0,
            end=60,
        )

        assert phase.has_takeoff is False
        assert phase.airtime_frames == 0


class TestMetricResult:
    """Test MetricResult dataclass."""

    def test_metric_result_creation(self):
        """Should create metric result correctly."""
        result = MetricResult(
            name="airtime",
            value=0.5,
            unit="s",
            is_good=True,
            reference_range=(0.4, 0.7),
        )

        assert result.name == "airtime"
        assert result.value == 0.5
        assert result.unit == "s"
        assert result.is_good is True
        assert result.reference_range == (0.4, 0.7)


class TestAnalysisReport:
    """Test AnalysisReport dataclass."""

    def test_analysis_report_creation(self):
        """Should create complete report."""
        phase = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=30,
            peak=45,
            landing=60,
            end=90,
        )

        metric = MetricResult(
            name="airtime",
            value=0.5,
            unit="s",
            is_good=True,
            reference_range=(0.4, 0.7),
        )

        report = AnalysisReport(
            element_type="waltz_jump",
            phases=phase,
            metrics=[metric],
            recommendations=["Good landing!"],
            overall_score=8.5,
            dtw_distance=0.15,
        )

        assert report.element_type == "waltz_jump"
        assert len(report.metrics) == 1
        assert report.overall_score == 8.5
        assert report.dtw_distance == 0.15


class TestReferenceData:
    """Test ReferenceData dataclass."""

    def test_reference_data_creation(self, tmp_path: Path):
        """Should create reference data."""
        # H3.6M 17-keypoint format
        poses = np.zeros((100, 17, 2), dtype=np.float32)

        meta = VideoMeta(
            path=tmp_path / "ref.mp4",
            fps=30.0,
            width=1920,
            height=1080,
            num_frames=100,
        )

        phase = ElementPhase(
            name="three_turn",
            start=0,
            takeoff=0,
            peak=0,
            landing=0,
            end=100,
        )

        ref = ReferenceData(
            element_type="three_turn",
            name="test_reference",
            poses=poses,
            meta=meta,
            phases=phase,
            fps=30.0,
            source="YouTube: Expert Skater",
        )

        assert ref.element_type == "three_turn"
        assert ref.poses.shape == (100, 17, 2)  # H3.6M format
        assert ref.source == "YouTube: Expert Skater"
