"""Tests for MotionDTW keyframe-aware alignment."""

import numpy as np
import pytest

from skating_ml.alignment.motion_dtw import MotionDTWAligner
from skating_ml.types import ElementPhase


@pytest.fixture
def sample_poses():
    """Create sample pose sequences for testing."""
    np.random.seed(42)

    # Create 100 frames with 33 keypoints, 2D coordinates
    num_frames = 100
    poses = np.random.randn(num_frames, 33, 2).astype(np.float32) * 0.1

    # Add a simple pattern (sine wave on hip y-position)
    t = np.linspace(0, 4 * np.pi, num_frames)
    poses[:, 23, 1] = np.sin(t) * 0.3  # Left hip
    poses[:, 24, 1] = np.sin(t) * 0.3  # Right hip

    return poses


@pytest.fixture
def jump_phases():
    """Create jump phase boundaries."""
    return ElementPhase(
        name="test_jump",
        start=0,
        takeoff=30,
        peak=50,
        landing=70,
        end=99,
    )


@pytest.fixture
def step_phases():
    """Create step/turn phase boundaries (no takeoff/landing)."""
    return ElementPhase(
        name="three_turn",
        start=0,
        takeoff=0,
        peak=0,
        landing=0,
        end=99,
    )


class TestMotionDTWAligner:
    """Test MotionDTWAligner functionality."""

    def test_init(self):
        """Test aligner initialization."""
        aligner = MotionDTWAligner()
        assert aligner._window_type == "sakoechiba"
        assert aligner._window_size == 0.2

        aligner_custom = MotionDTWAligner(window_type="itakura", window_size=0.3, phase_weight=2.0)
        assert aligner_custom._window_type == "itakura"
        assert aligner_custom._window_size == 0.3
        assert aligner_custom._phase_weight == 2.0

    def test_align_with_keyframes_jump(self, sample_poses, jump_phases):
        """Test keyframe-aware alignment for jump."""
        aligner = MotionDTWAligner()

        # Use shifted version as reference
        user_poses = sample_poses[:80]
        ref_poses = sample_poses[20:]

        user_phases = ElementPhase(
            name="test_jump", start=0, takeoff=20, peak=40, landing=60, end=79
        )
        ref_phases = ElementPhase(
            name="test_jump", start=0, takeoff=20, peak=40, landing=60, end=79
        )

        result = aligner.align_with_keyframes(user_poses, user_phases, ref_poses, ref_phases)

        # Check result structure
        assert result.aligned_user.shape[1] == 33
        assert result.aligned_user.shape[2] == 2
        assert len(result.phase_alignments) == 3  # entry, flight, landing
        assert len(result.full_warp_path) > 0

        # Check phase alignment
        phase_names = [pa.name for pa in result.phase_alignments]
        assert "entry" in phase_names
        assert "flight" in phase_names
        assert "landing" in phase_names

    def test_align_with_keyframes_step(self, sample_poses, step_phases):
        """Test keyframe-aware alignment for step/turn."""
        aligner = MotionDTWAligner()

        result = aligner.align_with_keyframes(sample_poses, step_phases, sample_poses, step_phases)

        # For steps, should have single phase
        assert len(result.phase_alignments) == 1
        assert result.phase_alignments[0].name == "full"

    def test_compute_phase_distances(self, sample_poses, jump_phases):
        """Test phase-wise distance computation."""
        aligner = MotionDTWAligner()

        distances = aligner.compute_phase_distances(
            sample_poses, jump_phases, sample_poses, jump_phases
        )

        # Should have distances for all three phases
        assert "entry" in distances
        assert "flight" in distances
        assert "landing" in distances

        # Identical sequences should have low distance
        for _phase_name, distance in distances.items():
            assert distance >= 0
            assert distance < 1.0  # Should be relatively low for identical

    def test_keyframe_extraction(self, sample_poses, jump_phases):
        """Test keyframe extraction logic."""
        aligner = MotionDTWAligner()

        keyframes = aligner._extract_keyframes(jump_phases, jump_phases)

        # Should extract 3 keyframes for jump
        assert len(keyframes) == 3
        assert [kf.name for kf in keyframes] == ["takeoff", "peak", "landing"]

        # Check indices
        assert keyframes[0].user_idx == 30  # takeoff
        assert keyframes[1].user_idx == 50  # peak
        assert keyframes[2].user_idx == 70  # landing

    def test_phase_splitting_jump(self, jump_phases):
        """Test phase splitting for jumps."""
        aligner = MotionDTWAligner()

        phases = aligner._split_into_phases(jump_phases, jump_phases)

        assert len(phases) == 3

        # Check entry phase
        assert phases[0]["name"] == "entry"
        assert phases[0]["user_start"] == 0
        assert phases[0]["user_end"] == 30

        # Check flight phase
        assert phases[1]["name"] == "flight"
        assert phases[1]["user_start"] == 30
        assert phases[1]["user_end"] == 70

        # Check landing phase
        assert phases[2]["name"] == "landing"
        assert phases[2]["user_start"] == 70
        assert phases[2]["user_end"] == 99

    def test_phase_splitting_step(self, step_phases):
        """Test phase splitting for steps/turns."""
        aligner = MotionDTWAligner()

        phases = aligner._split_into_phases(step_phases, step_phases)

        # Should have single phase
        assert len(phases) == 1
        assert phases[0]["name"] == "full"
        assert phases[0]["user_start"] == 0
        assert phases[0]["user_end"] == 99

    def test_empty_phase_handling(self, sample_poses):
        """Test handling of empty phases."""
        aligner = MotionDTWAligner()

        # Create phases with empty entry phase
        user_phases = ElementPhase(name="test", start=0, takeoff=0, peak=30, landing=60, end=99)
        ref_phases = ElementPhase(name="test", start=0, takeoff=0, peak=30, landing=60, end=99)

        result = aligner.align_with_keyframes(sample_poses, user_phases, sample_poses, ref_phases)

        # Should skip empty entry phase
        phase_names = [pa.name for pa in result.phase_alignments]
        assert "entry" not in phase_names or all(pa.distance >= 0 for pa in result.phase_alignments)

    def test_different_length_sequences(self, sample_poses, jump_phases):
        """Test alignment of sequences with different lengths."""
        aligner = MotionDTWAligner()

        user_poses = sample_poses[:70]
        ref_poses = sample_poses[:80]

        result = aligner.align_with_keyframes(user_poses, jump_phases, ref_poses, jump_phases)

        # Should produce aligned output
        assert result.aligned_user.shape[0] == len(ref_poses)
        assert len(result.full_warp_path) > 0
