"""Tests for DTW motion alignment."""

import numpy as np
import pytest

from skating_biomechanics_ml.alignment import MotionAligner
from skating_biomechanics_ml.types import ElementPhase


class TestMotionAligner:
    """Test MotionAligner."""

    def test_aligner_initialization(self):
        """Should initialize with default parameters."""
        aligner = MotionAligner()

        assert aligner._window_type == "sakoechiba"
        assert aligner._window_size == 0.2

    def test_aligner_custom_window(self):
        """Should initialize with custom window parameters."""
        aligner = MotionAligner(window_type="itakura", window_size=0.3)

        assert aligner._window_type == "itakura"
        assert aligner._window_size == 0.3

    def test_align_identical_sequences(self):
        """Should align identical sequences with zero distance."""
        aligner = MotionAligner()

        # Create identical sequences
        seq1 = np.random.randn(50, 17, 2).astype(np.float32)
        seq2 = seq1.copy()

        distance = aligner.compute_distance(seq1, seq2)

        # Distance should be very small (not exactly zero due to numerical errors)
        assert distance < 0.1

    def test_align_shifted_sequence(self):
        """Should align shifted sequences with small distance."""
        aligner = MotionAligner()

        # Create base sequence
        base = np.zeros((50, 17, 2), dtype=np.float32)
        for i in range(50):
            base[i, :, 0] = i * 0.01  # Gradual x movement

        # Shifted version (same pattern, different starting offset)
        shifted = base.copy()
        shifted[:, :, 0] += 0.1

        distance = aligner.compute_distance(base, shifted)

        # Distance should be small (same pattern, just shifted)
        assert distance < 1.0

    def test_align_different_sequences(self):
        """Should have larger distance for different sequences."""
        aligner = MotionAligner()

        # Create different sequences
        seq1 = np.zeros((50, 17, 2), dtype=np.float32)
        seq2 = np.ones((50, 17, 2), dtype=np.float32)

        distance = aligner.compute_distance(seq1, seq2)

        # Distance should be large
        assert distance > 1.0

    def test_align_returns_correct_shapes(self):
        """Should return aligned sequence with correct shape."""
        # Disable window constraint for random data
        aligner = MotionAligner(window_type=None)

        user = np.random.randn(30, 17, 2).astype(np.float32)
        reference = np.random.randn(50, 17, 2).astype(np.float32)

        aligned, warp_path = aligner.align(user, reference)

        # Aligned should match reference length
        assert aligned.shape[0] == reference.shape[0]
        assert aligned.shape[1] == 17
        assert aligned.shape[2] == 2

        # Warp path should have 2 columns
        assert warp_path.shape[1] == 2

    def test_align_with_subset_joints(self):
        """Should align using only specified joints."""
        # Disable window constraint for random data
        aligner = MotionAligner(window_type=None)

        user = np.random.randn(30, 17, 2).astype(np.float32)
        reference = np.random.randn(50, 17, 2).astype(np.float32)

        # Use only lower body joints
        joints = list(range(11, 17))  # Hips, knees, ankles

        distance = aligner.compute_distance(user, reference, joints=joints)

        assert isinstance(distance, float)

    def test_align_phases(self):
        """Should compute per-phase distances."""
        aligner = MotionAligner()

        user = np.random.randn(100, 17, 2).astype(np.float32)
        reference = np.random.randn(100, 17, 2).astype(np.float32)

        user_phases = ElementPhase(
            name="test_jump",
            start=0,
            takeoff=20,
            peak=50,
            landing=80,
            end=100,
        )

        ref_phases = ElementPhase(
            name="test_jump",
            start=0,
            takeoff=20,
            peak=50,
            landing=80,
            end=100,
        )

        distances = aligner.align_phases(user, user_phases, reference, ref_phases)

        # Should have distances for all phases
        assert "entry" in distances
        assert "flight" in distances
        assert "landing" in distances

        # All distances should be non-negative
        assert all(d >= 0 for d in distances.values())

    def test_extract_phase(self):
        """Should extract poses for a specific phase."""
        aligner = MotionAligner()

        poses = np.random.randn(100, 17, 2).astype(np.float32)

        phase = ElementPhase(
            name="test",
            start=20,
            takeoff=30,
            peak=50,
            landing=70,
            end=80,
        )

        extracted = aligner.extract_phase(poses, phase)

        # Should extract from start to end
        assert extracted.shape[0] == 60  # 80 - 20
        assert extracted.shape[1] == 17
        assert extracted.shape[2] == 2

    def test_align_no_window(self):
        """Should work without window constraint."""
        aligner = MotionAligner(window_type=None)

        user = np.random.randn(30, 17, 2).astype(np.float32)
        reference = np.random.randn(50, 17, 2).astype(np.float32)

        distance = aligner.compute_distance(user, reference)

        assert isinstance(distance, float)


class TestMotionAlignerEdgeCases:
    """Test edge cases and error handling."""

    def test_align_empty_sequences(self):
        """Should handle empty sequences gracefully."""
        aligner = MotionAligner()

        user = np.zeros((0, 17, 2), dtype=np.float32)
        reference = np.zeros((50, 17, 2), dtype=np.float32)

        # May raise error or return nan/inf
        try:
            distance = aligner.compute_distance(user, reference)
            # If it doesn't raise, check it's a valid float
            assert not np.isnan(distance)
            assert not np.isinf(distance)
        except (ValueError, IndexError):
            # Also acceptable to raise error
            pass

    def test_align_single_frame(self):
        """Should handle single-frame sequences."""
        aligner = MotionAligner()

        user = np.random.randn(1, 17, 2).astype(np.float32)
        reference = np.random.randn(1, 17, 2).astype(np.float32)

        distance = aligner.compute_distance(user, reference)

        assert isinstance(distance, float)
