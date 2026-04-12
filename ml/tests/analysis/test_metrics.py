"""Tests for biomechanics metrics computation."""

import numpy as np
import pytest

from skating_ml.analysis.element_defs import get_element_def
from skating_ml.analysis.metrics import BiomechanicsAnalyzer
from skating_ml.types import ElementPhase, H36Key, MetricResult


class TestBiomechanicsAnalyzer:
    """Test BiomechanicsAnalyzer."""

    def test_analyzer_initialization(self):
        """Should initialize with element definition."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        assert analyzer._element_def == element_def

    def test_compute_airtime(self):
        """Should compute airtime correctly."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=30,
            peak=45,
            landing=60,
            end=90,
        )

        airtime = analyzer.compute_airtime(phases, fps=30.0)

        assert airtime == pytest.approx(1.0)  # 30 frames / 30 fps

    def test_compute_angle_series(self, sample_normalized_poses):
        """Should compute angle series correctly."""
        element_def = get_element_def("three_turn")
        analyzer = BiomechanicsAnalyzer(element_def)

        angles = analyzer.compute_angle_series(
            sample_normalized_poses,
            H36Key.LEFT_SHOULDER,
            H36Key.LEFT_ELBOW,
            H36Key.LEFT_WRIST,
        )

        assert len(angles) == 3
        assert all(0 <= a <= 180 for a in angles)

    def test_compute_angular_velocity(self):
        """Should compute angular velocity."""
        element_def = get_element_def("three_turn")
        analyzer = BiomechanicsAnalyzer(element_def)

        # Create simple angle series: 0, 10, 20, 30 degrees
        angles = np.array([0, 10, 20, 30], dtype=np.float32)

        velocity = analyzer.compute_angular_velocity(angles, fps=10.0)

        assert len(velocity) == 4
        # Velocity should be ~100 deg/s (10 deg per frame at 10 fps)
        assert velocity[1] == pytest.approx(100, abs=1)

    def test_compute_jump_height(self):
        """Should compute jump height from hip trajectory."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        # Create hip Y trajectory: baseline, peak, return
        hip_y = np.array([0.3, 0.2, 0.1, 0.2, 0.3], dtype=np.float32)

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=0,
            peak=2,
            landing=4,
            end=4,
        )

        height = analyzer.compute_jump_height(hip_y, phases)

        # Height = baseline(0.3) - peak(0.1) = 0.2
        assert height == pytest.approx(0.2)

    def test_compute_jump_height_com(self, sample_normalized_poses):
        """Should compute jump height using Center of Mass trajectory.

        This tests the physics-accurate method that fixes the 60% error
        in the hip-only method for low jumps.
        """
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        # Create poses simulating a jump
        # Frame 0-1: takeoff, Frame 2: peak, Frame 3-4: landing
        jump_poses = np.zeros((5, 33, 2), dtype=np.float32)

        for i in range(5):
            # Hips move up then down (Y inverted, so negative = up)
            hip_y = 0.0 if i < 2 else (0.3 if i > 2 else -0.2)
            jump_poses[i, H36Key.LEFT_HIP] = [-0.05, hip_y]
            jump_poses[i, H36Key.RIGHT_HIP] = [0.05, hip_y]

            # Shoulders follow hips
            jump_poses[i, H36Key.LEFT_SHOULDER] = [-0.1, hip_y - 0.3]
            jump_poses[i, H36Key.RIGHT_SHOULDER] = [0.1, hip_y - 0.3]

            # Arms and legs also move
            jump_poses[i, H36Key.LEFT_WRIST] = [-0.2, hip_y - 0.7]
            jump_poses[i, H36Key.RIGHT_WRIST] = [0.2, hip_y - 0.7]
            jump_poses[i, H36Key.LEFT_ANKLE] = [-0.05, hip_y + 0.6]
            jump_poses[i, H36Key.RIGHT_ANKLE] = [0.05, hip_y + 0.6]
            jump_poses[i, H36Key.LEFT_KNEE] = [-0.05, hip_y + 0.3]
            jump_poses[i, H36Key.RIGHT_KNEE] = [0.05, hip_y + 0.3]
            jump_poses[i, H36Key.NOSE] = [0, hip_y - 0.5]
            jump_poses[i, H36Key.LEFT_ELBOW] = [-0.15, hip_y - 0.5]
            jump_poses[i, H36Key.RIGHT_ELBOW] = [0.15, hip_y - 0.5]

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=0,
            peak=2,
            landing=4,
            end=4,
        )

        height = analyzer.compute_jump_height_com(jump_poses, phases)

        # Height should be positive and reasonable
        assert height > 0
        # Peak CoM at frame 2 (-0.2 for hips), takeoff at frame 0 (0.0)
        # With full body, height should be less than hip-only difference
        # because arms/legs move with hips
        assert 0.1 < height < 0.5

    def test_jump_height_com_vs_hip_method(self):
        """CoM method should give different (more accurate) results than hip-only.

        This demonstrates the 60% error fix for bent-knee landings.
        """
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        # Simulate landing with bent knees (realistic skating landing)
        # Frame 0: takeoff (straight legs)
        # Frame 1: peak (in air)
        # Frame 2: landing (bent knees - hips lower but CoM similar)
        poses = np.zeros((3, 33, 2), dtype=np.float32)

        # Takeoff: straight legs
        poses[0, H36Key.LEFT_HIP] = [-0.05, 0.0]
        poses[0, H36Key.RIGHT_HIP] = [0.05, 0.0]
        poses[0, H36Key.LEFT_KNEE] = [-0.05, 0.3]
        poses[0, H36Key.RIGHT_KNEE] = [0.05, 0.3]
        poses[0, H36Key.LEFT_ANKLE] = [-0.05, 0.6]
        poses[0, H36Key.RIGHT_ANKLE] = [0.05, 0.6]
        poses[0, H36Key.LEFT_SHOULDER] = [-0.1, -0.3]
        poses[0, H36Key.RIGHT_SHOULDER] = [0.1, -0.3]
        poses[0, H36Key.NOSE] = [0, -0.5]

        # Peak: in air, body extended
        poses[1] = poses[0] * 0.9  # Everything higher (more negative Y)

        # Landing: bent knees (hips drop relative to ankles)
        poses[2, H36Key.LEFT_HIP] = [-0.05, 0.15]  # Hips dropped
        poses[2, H36Key.RIGHT_HIP] = [0.05, 0.15]
        poses[2, H36Key.LEFT_KNEE] = [-0.05, 0.4]  # Knees bent forward
        poses[2, H36Key.RIGHT_KNEE] = [0.05, 0.4]
        poses[2, H36Key.LEFT_ANKLE] = [-0.05, 0.6]  # Ankles same
        poses[2, H36Key.RIGHT_ANKLE] = [0.05, 0.6]
        poses[2, H36Key.LEFT_SHOULDER] = [-0.1, -0.15]  # Upper body similar
        poses[2, H36Key.RIGHT_SHOULDER] = [0.1, -0.15]
        poses[2, H36Key.NOSE] = [0, -0.35]

        phases = ElementPhase(
            name="test",
            start=0,
            takeoff=0,
            peak=1,
            landing=2,
            end=2,
        )

        # Hip-only method (overestimates due to bent knees)
        hip_y = (poses[:, H36Key.LEFT_HIP, 1] + poses[:, H36Key.RIGHT_HIP, 1]) / 2
        height_hip = analyzer.compute_jump_height(hip_y, phases)

        # CoM method (physics-accurate)
        height_com = analyzer.compute_jump_height_com(poses, phases)

        # CoM method should give lower (more accurate) height
        # because it accounts for full body mass distribution
        assert height_com < height_hip

        # The difference demonstrates the error in hip-only method
        # (would be ~60% for very low jumps with deep knee bend)

    def test_compute_arm_position(self, sample_normalized_poses):
        """Should compute arm position score."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        score = analyzer.compute_arm_position(sample_normalized_poses)

        assert 0 <= score <= 1

    def test_compute_trunk_lean(self, sample_normalized_poses):
        """Should compute trunk lean angle."""
        element_def = get_element_def("three_turn")
        analyzer = BiomechanicsAnalyzer(element_def)

        leans = analyzer.compute_trunk_lean(sample_normalized_poses)

        assert len(leans) == 3
        # All values should be reasonable angles
        assert all(-90 <= lean <= 90 for lean in leans)

    def test_compute_edge_indicator(self, sample_normalized_poses):
        """Should compute edge indicator."""
        element_def = get_element_def("three_turn")
        analyzer = BiomechanicsAnalyzer(element_def)

        indicator = analyzer.compute_edge_indicator(sample_normalized_poses, side="left")

        assert len(indicator) == 3
        # Values should be in range [-1, 1]
        assert all(-1 <= v <= 1 for v in indicator)

    def test_compute_symmetry(self, sample_normalized_poses):
        """Should compute symmetry score."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=0,
            peak=1,
            landing=2,
            end=2,
        )

        symmetry = analyzer.compute_symmetry(sample_normalized_poses, phases)

        assert 0 <= symmetry <= 1

    def test_analyze_jump(self, sample_normalized_poses):
        """Should analyze jump and return metrics."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=0,
            peak=1,
            landing=2,
            end=2,
        )

        metrics = analyzer.analyze(sample_normalized_poses, phases, fps=30.0)

        assert len(metrics) > 0
        assert all(isinstance(m, MetricResult) for m in metrics)

        # Check specific metrics exist
        metric_names = [m.name for m in metrics]
        assert "airtime" in metric_names

    def test_analyze_step(self, sample_normalized_poses):
        """Should analyze step element and return metrics."""
        element_def = get_element_def("three_turn")
        analyzer = BiomechanicsAnalyzer(element_def)

        phases = ElementPhase(
            name="three_turn",
            start=0,
            takeoff=0,
            peak=1,
            landing=0,
            end=2,
        )

        metrics = analyzer.analyze(sample_normalized_poses, phases, fps=30.0)

        assert len(metrics) > 0

        # Check step-specific metrics
        metric_names = [m.name for m in metrics]
        assert "trunk_lean" in metric_names

    def test_analyze_jump_includes_landing_metrics(self, sample_normalized_poses):
        """Jump analysis should include new landing quality metrics."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=0,
            peak=1,
            landing=2,
            end=2,
        )

        metrics = analyzer.analyze(sample_normalized_poses, phases, fps=30.0)
        metric_names = [m.name for m in metrics]

        assert "landing_knee_stability" in metric_names
        assert "landing_trunk_recovery" in metric_names
        assert "relative_jump_height" in metric_names

    def test_compute_landing_knee_stability_stable(self):
        """Should return high score for constant knee angles after landing."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        # Create poses with stable knee angles after landing
        poses = np.zeros((10, 17, 2), dtype=np.float32)
        for i in range(10):
            # Set up full body skeleton
            poses[i, H36Key.LHIP] = [-0.05, 0.0]
            poses[i, H36Key.RHIP] = [0.05, 0.0]
            poses[i, H36Key.LKNEE] = [-0.05, 0.3]
            poses[i, H36Key.RKNEE] = [0.05, 0.3]
            poses[i, H36Key.LFOOT] = [-0.05, 0.6]
            poses[i, H36Key.RFOOT] = [0.05, 0.6]
            poses[i, H36Key.LSHOULDER] = [-0.1, -0.3]
            poses[i, H36Key.RSHOULDER] = [0.1, -0.3]
            poses[i, H36Key.HEAD] = [0, -0.5]

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=2,
            peak=4,
            landing=6,
            end=9,
        )

        score = analyzer.compute_landing_knee_stability(poses, phases)

        # Constant knee angles should give high stability score
        assert score > 0.7
        assert score <= 1.0

    def test_compute_landing_knee_stability_wobbly(self):
        """Should return low score for oscillating knee angles after landing."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        # Create poses with wobbling knees after landing
        poses = np.zeros((10, 17, 2), dtype=np.float32)
        for i in range(10):
            # Set up full body skeleton
            poses[i, H36Key.LHIP] = [-0.05, 0.0]
            poses[i, H36Key.RHIP] = [0.05, 0.0]
            poses[i, H36Key.LFOOT] = [-0.05, 0.6]
            poses[i, H36Key.RFOOT] = [0.05, 0.6]
            poses[i, H36Key.LSHOULDER] = [-0.1, -0.3]
            poses[i, H36Key.RSHOULDER] = [0.1, -0.3]
            poses[i, H36Key.HEAD] = [0, -0.5]

            # Add extreme wobble to knee positions after landing
            # Dramatic X movement to create large angle changes
            if i >= 6:  # After landing frame
                # Create large angle variation: knee moves far from hip-foot line
                if i % 2 == 0:
                    # Even frames: knee way forward
                    poses[i, H36Key.LKNEE] = [0.1, 0.35]
                    poses[i, H36Key.RKNEE] = [0.2, 0.35]
                else:
                    # Odd frames: knee way back
                    poses[i, H36Key.LKNEE] = [-0.15, 0.25]
                    poses[i, H36Key.RKNEE] = [-0.05, 0.25]
            else:
                poses[i, H36Key.LKNEE] = [-0.05, 0.3]
                poses[i, H36Key.RKNEE] = [0.05, 0.3]

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=2,
            peak=4,
            landing=6,
            end=9,
        )

        score = analyzer.compute_landing_knee_stability(poses, phases)

        # Oscillating knees should give low stability score
        assert score < 0.5
        assert score >= 0.0

    def test_compute_landing_knee_stability_no_post_landing(self):
        """Should return 1.0 when there's no post-landing data."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        # Create minimal poses
        poses = np.zeros((5, 17, 2), dtype=np.float32)
        for i in range(5):
            poses[i, H36Key.LHIP] = [-0.05, 0.0]
            poses[i, H36Key.RHIP] = [0.05, 0.0]
            poses[i, H36Key.LKNEE] = [-0.05, 0.3]
            poses[i, H36Key.RKNEE] = [0.05, 0.3]
            poses[i, H36Key.LFOOT] = [-0.05, 0.6]
            poses[i, H36Key.RFOOT] = [0.05, 0.6]
            poses[i, H36Key.LSHOULDER] = [-0.1, -0.3]
            poses[i, H36Key.RSHOULDER] = [0.1, -0.3]
            poses[i, H36Key.HEAD] = [0, -0.5]

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=2,
            peak=3,
            landing=4,
            end=4,  # No post-landing frames
        )

        score = analyzer.compute_landing_knee_stability(poses, phases)

        # Should return perfect stability when no post-landing data
        assert score == 1.0

    def test_compute_relative_jump_height(self):
        """Should compute jump height normalized by spine length."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        # Create 10 frames with a jump parabola
        poses = np.zeros((10, 17, 2), dtype=np.float32)

        # Spine length is 0.3 (hips at base_y, shoulders at base_y - 0.3)
        base_y = 0.5
        spine_length = 0.3
        jump_amplitude = 0.15

        for i in range(10):
            # Parabolic jump at frames 2-7
            # Frame 2: takeoff, Frame 4-5: peak, Frame 7: landing
            if 2 <= i <= 7:
                mid = 4.5
                width = 5.0
                parabola = jump_amplitude * (1 - 4 * ((i - mid) / width) ** 2)
                hip_y = base_y - parabola  # Y inverted, so subtract for up
            else:
                hip_y = base_y

            # Set keypoints
            poses[i, H36Key.LHIP] = [-0.05, hip_y]
            poses[i, H36Key.RHIP] = [0.05, hip_y]
            poses[i, H36Key.LSHOULDER] = [-0.1, hip_y - spine_length]
            poses[i, H36Key.RSHOULDER] = [0.1, hip_y - spine_length]
            poses[i, H36Key.LKNEE] = [-0.05, hip_y + 0.3]
            poses[i, H36Key.RKNEE] = [0.05, hip_y + 0.3]
            poses[i, H36Key.LFOOT] = [-0.05, hip_y + 0.6]
            poses[i, H36Key.RFOOT] = [0.05, hip_y + 0.6]
            poses[i, H36Key.HEAD] = [0, hip_y - 0.5]

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=2,
            peak=5,
            landing=7,
            end=9,
        )

        height = analyzer.compute_relative_jump_height(poses, phases)

        # Height should be positive and reasonable
        # With amplitude 0.15 and spine 0.3, ratio should be ~0.5
        assert height > 0
        assert height < 3.0  # Sanity check

    def test_compute_relative_jump_height_no_jump(self):
        """Should return 0.0 when takeoff >= landing."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        # Create 5 frames with no jump
        poses = np.zeros((5, 17, 2), dtype=np.float32)
        base_y = 0.5
        spine_length = 0.3

        for i in range(5):
            poses[i, H36Key.LHIP] = [-0.05, base_y]
            poses[i, H36Key.RHIP] = [0.05, base_y]
            poses[i, H36Key.LSHOULDER] = [-0.1, base_y - spine_length]
            poses[i, H36Key.RSHOULDER] = [0.1, base_y - spine_length]
            poses[i, H36Key.LKNEE] = [-0.05, base_y + 0.3]
            poses[i, H36Key.RKNEE] = [0.05, base_y + 0.3]
            poses[i, H36Key.LFOOT] = [-0.05, base_y + 0.6]
            poses[i, H36Key.RFOOT] = [0.05, base_y + 0.6]
            poses[i, H36Key.HEAD] = [0, base_y - 0.5]

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=1,
            peak=1,
            landing=1,  # Same as takeoff
            end=4,
        )

        height = analyzer.compute_relative_jump_height(poses, phases)

        # Should return 0.0 when no jump
        assert height == 0.0

    def test_compute_landing_trunk_recovery_good(self):
        """Should return high score when trunk stays upright after landing."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        # Create poses with upright trunk throughout
        poses = np.zeros((10, 17, 2), dtype=np.float32)
        for i in range(10):
            # Set up full body skeleton with shoulders directly above hips
            poses[i, H36Key.LHIP] = [-0.05, 0.0]
            poses[i, H36Key.RHIP] = [0.05, 0.0]
            poses[i, H36Key.LKNEE] = [-0.05, 0.3]
            poses[i, H36Key.RKNEE] = [0.05, 0.3]
            poses[i, H36Key.LFOOT] = [-0.05, 0.6]
            poses[i, H36Key.RFOOT] = [0.05, 0.6]
            # Shoulders directly above hips (same X, different Y) = upright
            poses[i, H36Key.LSHOULDER] = [-0.05, -0.3]
            poses[i, H36Key.RSHOULDER] = [0.05, -0.3]
            poses[i, H36Key.HEAD] = [0, -0.5]

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=2,
            peak=4,
            landing=6,
            end=9,
        )

        score = analyzer.compute_landing_trunk_recovery(poses, phases)

        # Upright trunk should give high recovery score
        assert score > 0.7
        assert score <= 1.0

    def test_compute_landing_trunk_recovery_leaned(self):
        """Should return low score when trunk leans forward after landing."""
        element_def = get_element_def("waltz_jump")
        analyzer = BiomechanicsAnalyzer(element_def)

        # Create poses with forward lean after landing
        poses = np.zeros((10, 17, 2), dtype=np.float32)
        for i in range(10):
            # Set up full body skeleton
            poses[i, H36Key.LHIP] = [-0.05, 0.0]
            poses[i, H36Key.RHIP] = [0.05, 0.0]
            poses[i, H36Key.LKNEE] = [-0.05, 0.3]
            poses[i, H36Key.RKNEE] = [0.05, 0.3]
            poses[i, H36Key.LFOOT] = [-0.05, 0.6]
            poses[i, H36Key.RFOOT] = [0.05, 0.6]
            poses[i, H36Key.HEAD] = [0, -0.5]

            # After landing, shoulders shift forward (Y less negative = higher)
            # This creates forward trunk lean
            if i >= 6:  # After landing frame
                # Shoulders higher (less negative Y) and slightly forward (positive X)
                poses[i, H36Key.LSHOULDER] = [0.05, -0.15]  # Forward and up
                poses[i, H36Key.RSHOULDER] = [0.15, -0.15]  # Forward and up
            else:
                # Before landing: upright
                poses[i, H36Key.LSHOULDER] = [-0.05, -0.3]
                poses[i, H36Key.RSHOULDER] = [0.05, -0.3]

        phases = ElementPhase(
            name="waltz_jump",
            start=0,
            takeoff=2,
            peak=4,
            landing=6,
            end=9,
        )

        score = analyzer.compute_landing_trunk_recovery(poses, phases)

        # Forward lean should give low recovery score
        assert score < 0.5
        assert score >= 0.0


class TestMetricResult:
    """Test MetricResult dataclass."""

    def test_metric_result_creation(self):
        """Should create metric result correctly."""
        result = MetricResult(
            name="test_metric",
            value=0.5,
            unit="s",
            is_good=True,
            reference_range=(0.4, 0.6),
        )

        assert result.name == "test_metric"
        assert result.value == 0.5
        assert result.unit == "s"
        assert result.is_good is True
        assert result.reference_range == (0.4, 0.6)
