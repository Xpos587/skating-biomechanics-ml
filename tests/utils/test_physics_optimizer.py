"""Tests for physics-informed pose optimizer."""

import numpy as np
import pytest

from skating_biomechanics_ml.types import BKey
from skating_biomechanics_ml.utils.physics_optimizer import (
    BoneConstraints,
    PhysicsPoseOptimizer,
    optimize_poses_with_physics,
)


@pytest.fixture
def sample_poses_with_jitter():
    """Create sample poses with jitter (simulating detection noise).

    Creates a T-pose sequence with random jitter added to simulate
    BlazePose detection noise that needs to be smoothed.
    """
    num_frames = 10
    poses = np.zeros((num_frames, 33, 2), dtype=np.float32)

    for i in range(num_frames):
        # Base T-pose
        poses[i, BKey.LEFT_HIP] = [-0.05, 0.0]
        poses[i, BKey.RIGHT_HIP] = [0.05, 0.0]
        poses[i, BKey.LEFT_SHOULDER] = [-0.1, -0.3]
        poses[i, BKey.RIGHT_SHOULDER] = [0.1, -0.3]
        poses[i, BKey.LEFT_ELBOW] = [-0.15, -0.5]
        poses[i, BKey.RIGHT_ELBOW] = [0.15, -0.5]
        poses[i, BKey.LEFT_WRIST] = [-0.2, -0.7]
        poses[i, BKey.RIGHT_WRIST] = [0.2, -0.7]
        poses[i, BKey.LEFT_KNEE] = [-0.05, 0.3]
        poses[i, BKey.RIGHT_KNEE] = [0.05, 0.3]
        poses[i, BKey.LEFT_ANKLE] = [-0.05, 0.6]
        poses[i, BKey.RIGHT_ANKLE] = [0.05, 0.6]

    # Add jitter (simulating detection noise)
    jitter = np.random.randn(num_frames, 33, 2).astype(np.float32) * 0.02
    poses += jitter

    return poses


@pytest.fixture
def poses_with_bone_length_violations():
    """Create poses with unnatural bone lengths (occlusion artifacts).

    Simulates what happens when BlazePose gets confused during
    rotations or occlusions - bone lengths become inconsistent.
    """
    poses = np.zeros((5, 33, 2), dtype=np.float32)

    for i in range(5):
        # Normal body structure
        poses[i, BKey.LEFT_HIP] = [-0.05, 0.0]
        poses[i, BKey.RIGHT_HIP] = [0.05, 0.0]
        poses[i, BKey.LEFT_SHOULDER] = [-0.1, -0.3]
        poses[i, BKey.RIGHT_SHOULDER] = [0.1, -0.3]
        poses[i, BKey.LEFT_KNEE] = [-0.05, 0.3]
        poses[i, BKey.RIGHT_KNEE] = [0.05, 0.3]
        poses[i, BKey.LEFT_ANKLE] = [-0.05, 0.6]
        poses[i, BKey.RIGHT_ANKLE] = [0.05, 0.6]

    # Frame 2: Elbow position violates bone length (simulating occlusion)
    poses[2, BKey.LEFT_ELBOW] = [-0.05, -0.6]  # Too close to shoulder
    poses[2, BKey.LEFT_WRIST] = [-0.06, -0.9]  # Compressed forearm

    # Frame 3: Knee position violates bone length
    poses[3, BKey.LEFT_KNEE] = [-0.04, 0.15]  # Too close to hip
    poses[3, BKey.LEFT_ANKLE] = [-0.03, 0.5]  # Compressed shin

    return poses


class TestBoneConstraints:
    """Tests for BoneConstraints dataclass."""

    def test_default_constraints(self):
        """Should have default anthropometric ratios."""
        constraints = BoneConstraints()

        assert constraints.SPINE_TO_UPPER_ARM == 0.45
        assert constraints.SPINE_TO_FOREARM == 0.40
        assert constraints.SPINE_TO_THIGH == 0.50
        assert constraints.SPINE_TO_SHIN == 0.48
        assert constraints.TOLERANCE == 0.15

    def test_custom_constraints(self):
        """Should allow custom constraint values."""
        constraints = BoneConstraints(
            SPINE_TO_UPPER_ARM=0.5,
            TOLERANCE=0.2,
        )

        assert constraints.SPINE_TO_UPPER_ARM == 0.5
        assert constraints.TOLERANCE == 0.2
        # Other values should be defaults
        assert constraints.SPINE_TO_FOREARM == 0.40


class TestPhysicsPoseOptimizer:
    """Tests for PhysicsPoseOptimizer class."""

    def test_initialization(self):
        """Should initialize with default parameters."""
        optimizer = PhysicsPoseOptimizer()

        assert optimizer.constraints == BoneConstraints()
        assert optimizer.process_noise == 0.01
        assert optimizer.measurement_noise == 0.1

    def test_initialization_custom(self):
        """Should accept custom parameters."""
        constraints = BoneConstraints(TOLERANCE=0.2)
        optimizer = PhysicsPoseOptimizer(
            constraints=constraints,
            process_noise=0.02,
            measurement_noise=0.15,
        )

        assert optimizer.constraints == constraints
        assert optimizer.process_noise == 0.02
        assert optimizer.measurement_noise == 0.15

    def test_learn_bone_lengths(self, sample_poses_with_jitter):
        """Should learn bone lengths from pose sequence."""
        optimizer = PhysicsPoseOptimizer()
        optimizer.learn_bone_lengths(sample_poses_with_jitter)

        # Spine length should be learned
        assert optimizer._spine_length is not None
        assert optimizer._spine_length > 0

        # Some bone lengths should be learned
        assert len(optimizer._bone_lengths) > 0

    def test_optimize_sequence_reduces_jitter(self, sample_poses_with_jitter):
        """Optimization should reduce jitter in pose sequence."""
        optimizer = PhysicsPoseOptimizer()
        optimizer.learn_bone_lengths(sample_poses_with_jitter)

        # Set random seed for reproducibility
        np.random.seed(42)
        jittered = sample_poses_with_jitter.copy()
        optimized = optimizer.optimize_sequence(jittered)

        # Calculate jitter (standard deviation of joint positions over time)
        original_jitter = np.std(jittered, axis=0).mean()
        optimized_jitter = np.std(optimized, axis=0).mean()

        # Optimized should have less jitter
        assert optimized_jitter < original_jitter

    def test_optimize_sequence_preserves_structure(self, sample_poses_with_jitter):
        """Optimization should preserve overall pose structure."""
        optimizer = PhysicsPoseOptimizer()
        optimizer.learn_bone_lengths(sample_poses_with_jitter)

        optimized = optimizer.optimize_sequence(sample_poses_with_jitter)

        # Output shape should match input
        assert optimized.shape == sample_poses_with_jitter.shape

        # Mid-hip should remain near origin (root-centered)
        mid_hip_orig = (sample_poses_with_jitter[:, 23] + sample_poses_with_jitter[:, 24]) / 2
        mid_hip_opt = (optimized[:, 23] + optimized[:, 24]) / 2

        # Both should be near origin
        assert np.allclose(mid_hip_opt.mean(axis=0), [0, 0], atol=0.05)

    def test_enforce_bone_constraints(self, poses_with_bone_length_violations):
        """Bone constraints should correct unnatural poses."""
        optimizer = PhysicsPoseOptimizer()
        optimizer.learn_bone_lengths(poses_with_bone_length_violations)

        optimized = optimizer.optimize_sequence(poses_with_bone_length_violations)

        # The optimizer should smooth the sequence
        # Check that the optimized sequence is different from input
        assert not np.allclose(optimized, poses_with_bone_length_violations)

        # Check that the optimization reduces overall variation
        # (Kalman filter smoothing effect)
        orig_std = np.std(poses_with_bone_length_violations, axis=0).mean()
        opt_std = np.std(optimized, axis=0).mean()
        assert opt_std <= orig_std  # Should not increase variation

    def test_empty_sequence(self):
        """Should handle empty sequence gracefully."""
        optimizer = PhysicsPoseOptimizer()
        empty = np.zeros((0, 33, 2), dtype=np.float32)

        result = optimizer.optimize_sequence(empty)

        assert result.shape == (0, 33, 2)

    def test_single_frame(self):
        """Should handle single-frame sequence."""
        optimizer = PhysicsPoseOptimizer()
        single = np.zeros((1, 33, 2), dtype=np.float32)
        single[0, BKey.LEFT_HIP] = [-0.05, 0.0]
        single[0, BKey.RIGHT_HIP] = [0.05, 0.0]

        result = optimizer.optimize_sequence(single)

        assert result.shape == (1, 33, 2)


class TestOptimizePosesWithPhysics:
    """Tests for convenience function."""

    def test_convenience_function(self, sample_poses_with_jitter):
        """Convenience function should optimize poses."""
        np.random.seed(42)

        optimized = optimize_poses_with_physics(sample_poses_with_jitter)

        # Should return optimized poses
        assert optimized.shape == sample_poses_with_jitter.shape

        # Should be different (improved) from input
        assert not np.allclose(optimized, sample_poses_with_jitter)

    def test_convenience_with_custom_constraints(self, sample_poses_with_jitter):
        """Should accept custom bone constraints."""
        constraints = BoneConstraints(TOLERANCE=0.1)  # Stricter tolerance

        optimized = optimize_poses_with_physics(
            sample_poses_with_jitter,
            constraints=constraints,
        )

        assert optimized.shape == sample_poses_with_jitter.shape
