"""Tests for core data types."""

import numpy as np
import pytest
from pathlib import Path

from skating_biomechanics_ml.types import (
    BKey,
    BoundingBox,
    VideoMeta,
    ElementPhase,
    MetricResult,
    AnalysisReport,
    ReferenceData,
    BLAZEPOSE_INDICES,
    BLAZEPOSE_SKELETON_EDGES,
)


class TestBKey:
    """Test BlazePose keypoint enum."""

    def test_keypoint_indices(self):
        """Expected indices for key landmarks."""
        assert BKey.NOSE == 0
        assert BKey.LEFT_SHOULDER == 11
        assert BKey.RIGHT_SHOULDER == 12
        assert BKey.LEFT_HIP == 23
        assert BKey.RIGHT_HIP == 24

    def test_all_33_keypoints(self):
        """Should have exactly 33 keypoints."""
        assert len(list(BKey)) == 33

    def test_foot_keypoints(self):
        """Foot keypoints for edge detection."""
        assert BKey.LEFT_HEEL == 29
        assert BKey.RIGHT_HEEL == 30
        assert BKey.LEFT_FOOT_INDEX == 31
        assert BKey.RIGHT_FOOT_INDEX == 32


class TestBlazePoseIndices:
    """Test BlazePose indices."""

    def test_blazepose_size(self):
        """BlazePose should have 33 keypoints."""
        assert len(BLAZEPOSE_INDICES) == 33

    def test_blazepose_contains_key_joints(self):
        """Should include major joints."""
        assert BKey.NOSE in BLAZEPOSE_INDICES
        assert BKey.LEFT_SHOULDER in BLAZEPOSE_INDICES
        assert BKey.RIGHT_SHOULDER in BLAZEPOSE_INDICES
        assert BKey.LEFT_HIP in BLAZEPOSE_INDICES
        assert BKey.RIGHT_HIP in BLAZEPOSE_INDICES
        assert BKey.LEFT_FOOT_INDEX in BLAZEPOSE_INDICES


class TestSkeletonEdges:
    """Test skeleton edge definitions."""

    def test_edges_are_tuples(self):
        """All edges should be (joint_a, joint_b) tuples."""
        for edge in BLAZEPOSE_SKELETON_EDGES:
            assert isinstance(edge, tuple)
            assert len(edge) == 2
            assert isinstance(edge[0], (int, BKey))
            assert isinstance(edge[1], (int, BKey))

    def test_upper_body_edges(self):
        """Should have upper body connections."""
        # Shoulder to shoulder
        assert (BKey.LEFT_SHOULDER, BKey.RIGHT_SHOULDER) in BLAZEPOSE_SKELETON_EDGES

        # Arm connections
        assert (BKey.LEFT_SHOULDER, BKey.LEFT_ELBOW) in BLAZEPOSE_SKELETON_EDGES
        assert (BKey.LEFT_ELBOW, BKey.LEFT_WRIST) in BLAZEPOSE_SKELETON_EDGES

    def test_lower_body_edges(self):
        """Should have lower body connections."""
        # Hip to hip
        assert (BKey.LEFT_HIP, BKey.RIGHT_HIP) in BLAZEPOSE_SKELETON_EDGES

        # Leg connections
        assert (BKey.LEFT_HIP, BKey.LEFT_KNEE) in BLAZEPOSE_SKELETON_EDGES
        assert (BKey.LEFT_KNEE, BKey.LEFT_ANKLE) in BLAZEPOSE_SKELETON_EDGES

        # Foot edges for edge detection
        assert (BKey.LEFT_ANKLE, BKey.LEFT_HEEL) in BLAZEPOSE_SKELETON_EDGES
        assert (BKey.LEFT_ANKLE, BKey.LEFT_FOOT_INDEX) in BLAZEPOSE_SKELETON_EDGES


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
        poses = np.zeros((100, 33, 2), dtype=np.float32)

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
        assert ref.poses.shape == (100, 33, 2)
        assert ref.source == "YouTube: Expert Skater"
