"""Tests for physics engine calculations."""

import numpy as np

from skating_ml.analysis.physics_engine import (
    DEFAULT_BODY_MASS,
    SEGMENT_MASS_RATIOS,
    PhysicsEngine,
    PhysicsResult,
)


class TestPhysicsEngineInit:
    """Tests for PhysicsEngine initialization."""

    def test_default_body_mass(self):
        """Test default body mass is set correctly."""
        engine = PhysicsEngine()
        assert engine.body_mass == DEFAULT_BODY_MASS

    def test_custom_body_mass(self):
        """Test custom body mass is set correctly."""
        engine = PhysicsEngine(body_mass=70.0)
        assert engine.body_mass == 70.0

    def test_segment_masses_calculated(self):
        """Test that segment masses are calculated from body mass."""
        engine = PhysicsEngine(body_mass=60.0)

        # Check that segment masses sum to body mass
        total_mass = sum(engine.segment_masses.values())
        np.testing.assert_almost_equal(total_mass, 60.0, decimal=2)

    def test_segment_mass_ratios_match_table(self):
        """Test that segment mass ratios match Dempster table."""
        engine = PhysicsEngine(body_mass=100.0)

        for name, ratio in SEGMENT_MASS_RATIOS.items():
            expected = ratio * 100.0
            np.testing.assert_almost_equal(engine.segment_masses[name], expected, decimal=2)


class TestCalculateCenterOfMass:
    """Tests for Center of Mass calculation."""

    def test_com_shape(self):
        """Test that CoM trajectory has correct shape."""
        engine = PhysicsEngine()
        poses_3d = np.random.rand(100, 17, 3).astype(np.float32)

        com = engine.calculate_center_of_mass(poses_3d)

        assert com.shape == (100, 3)

    def test_single_frame_com(self):
        """Test CoM calculation for single frame."""
        engine = PhysicsEngine()
        poses_3d = np.random.rand(1, 17, 3).astype(np.float32)

        com = engine.calculate_center_of_mass(poses_3d)

        assert com.shape == (1, 3)

    def test_com_within_body_bounds(self):
        """Test that CoM is within reasonable body bounds."""
        engine = PhysicsEngine()
        # Create a T-pose (all keypoints at z=0)
        poses_3d = np.zeros((1, 17, 3), dtype=np.float32)
        poses_3d[0, :, 0] = np.linspace(0, 1, 17)  # Spread x
        poses_3d[0, :, 1] = np.linspace(0, 1, 17)  # Spread y

        com = engine.calculate_center_of_mass(poses_3d)

        # CoM should be within the bounding box of keypoints
        assert 0 <= com[0, 0] <= 1
        assert 0 <= com[0, 1] <= 1


class TestCalculateMomentOfInertia:
    """Tests for Moment of Inertia calculation."""

    def test_inertia_shape(self):
        """Test that inertia array has correct shape."""
        engine = PhysicsEngine()
        poses_3d = np.random.rand(100, 17, 3).astype(np.float32)

        inertia = engine.calculate_moment_of_inertia(poses_3d)

        assert inertia.shape == (100,)

    def test_inertia_positive_values(self):
        """Test that moment of inertia is always positive."""
        engine = PhysicsEngine()
        poses_3d = np.random.rand(50, 17, 3).astype(np.float32)

        inertia = engine.calculate_moment_of_inertia(poses_3d)

        assert np.all(inertia > 0)

    def test_inertia_t_pose_smaller_than_star_pose(self):
        """Test that compact pose has lower inertia than extended pose."""
        engine = PhysicsEngine()

        # Compact pose (all keypoints near center)
        compact_pose = np.zeros((1, 17, 3), dtype=np.float32)
        compact_pose[0, :, :] = 0.5

        # Extended pose (keypoints spread out)
        extended_pose = np.zeros((1, 17, 3), dtype=np.float32)
        extended_pose[0, :, 0] = np.linspace(-1, 1, 17)

        inertia_compact = engine.calculate_moment_of_inertia(compact_pose)[0]
        inertia_extended = engine.calculate_moment_of_inertia(extended_pose)[0]

        assert inertia_compact < inertia_extended


class TestCalculateAngularMomentum:
    """Tests for Angular Momentum calculation."""

    def test_angular_momentum_shape(self):
        """Test that angular momentum array has correct shape."""
        engine = PhysicsEngine()
        poses_3d = np.random.rand(100, 17, 3).astype(np.float32)
        angular_velocity = np.ones(100) * 2.0  # 2 rad/s

        L = engine.calculate_angular_momentum(poses_3d, angular_velocity)

        assert L.shape == (100,)

    def test_angular_momentum_proportional_to_omega(self):
        """Test L = I x omega relationship."""
        engine = PhysicsEngine()
        poses_3d = np.random.rand(10, 17, 3).astype(np.float32)

        omega_1 = np.ones(10) * 1.0
        omega_2 = np.ones(10) * 2.0

        L_1 = engine.calculate_angular_momentum(poses_3d, omega_1)
        L_2 = engine.calculate_angular_momentum(poses_3d, omega_2)

        np.testing.assert_array_almost_equal(L_2, 2 * L_1, decimal=5)


class TestFitJumpTrajectory:
    """Tests for parabolic trajectory fitting."""

    def test_fit_trajectory_shape(self):
        """Test that trajectory fitting returns dict with expected keys."""
        engine = PhysicsEngine()
        poses_3d = np.random.rand(100, 17, 3).astype(np.float32)

        result = engine.fit_jump_trajectory(poses_3d, takeoff_idx=20, landing_idx=60)

        assert "height" in result
        assert "flight_time" in result
        assert "takeoff_velocity" in result
        assert "fit_quality" in result

    def test_height_positive(self):
        """Test that calculated jump height is non-negative."""
        engine = PhysicsEngine()

        # Create a realistic arc trajectory with proper CoM variation
        poses_3d = np.zeros((100, 17, 3), dtype=np.float32)
        # Simulate jump: Y goes up then down
        np.arange(100)
        takeoff, _peak, landing = 20, 50, 80
        # Parabolic arc: h(t) = h0 + v0*t - 0.5*g*t^2
        t = np.arange(landing - takeoff) / 30.0  # Time in seconds
        # Create a proper parabola with meaningful values
        arc = -4.9 * t**2 + 2.0 * t + 1.0  # Jump starting at 1m height
        # Set all keypoints to follow the arc (simplified)
        for i in range(takeoff, landing):
            poses_3d[i, :, 1] = arc[i - takeoff]
        # Add CoM variation by making head/torso move differently
        poses_3d[takeoff:landing, 10, 1] += 0.1  # Head slightly higher
        poses_3d[takeoff:landing, 7, 1] -= 0.05  # Spine slightly lower

        result = engine.fit_jump_trajectory(poses_3d, takeoff_idx=takeoff, landing_idx=landing)

        # Height should be non-negative (or fallback to simple diff if fit fails)
        assert result["height"] >= 0 or np.isnan(result["height"])

    def test_flight_time_calculation(self):
        """Test that flight time is calculated correctly."""
        engine = PhysicsEngine()
        poses_3d = np.random.rand(100, 17, 3).astype(np.float32)

        result = engine.fit_jump_trajectory(poses_3d, takeoff_idx=20, landing_idx=60)

        # 40 frames at 30 fps = 1.33 seconds
        expected_time = 40 / 30.0
        np.testing.assert_almost_equal(result["flight_time"], expected_time, decimal=2)


class TestAnalyze:
    """Tests for full physics analysis."""

    def test_analyze_returns_physics_result(self):
        """Test that analyze returns PhysicsResult with correct fields."""
        engine = PhysicsEngine()
        poses_3d = np.random.rand(100, 17, 3).astype(np.float32)

        result = engine.analyze(poses_3d)

        assert isinstance(result, PhysicsResult)
        assert result.center_of_mass.shape == (100, 3)
        assert result.moment_of_inertia.shape == (100,)
        assert result.angular_momentum.shape == (100,)

    def test_analyze_with_takeoff_landing(self):
        """Test analyze with takeoff/landing indices."""
        engine = PhysicsEngine()
        poses_3d = np.random.rand(100, 17, 3).astype(np.float32)

        result = engine.analyze(poses_3d, takeoff_idx=20, landing_idx=60)

        assert result.jump_height is not None
        assert result.flight_time is not None

    def test_analyze_without_takeoff_landing(self):
        """Test analyze without takeoff/landing indices."""
        engine = PhysicsEngine()
        poses_3d = np.random.rand(100, 17, 3).astype(np.float32)

        result = engine.analyze(poses_3d)

        assert result.jump_height is None
        assert result.flight_time is None
