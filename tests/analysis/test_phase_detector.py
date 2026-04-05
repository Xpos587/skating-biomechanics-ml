"""Tests for automatic phase detection."""

import numpy as np

from src.analysis.metrics import PhaseDetectionResult
from src.analysis.phase_detector import PhaseDetector
from src.types import ElementPhase, H36Key


class TestPhaseDetector:
    """Test PhaseDetector."""

    def test_detector_initialization(self):
        """Should initialize without errors."""
        detector = PhaseDetector()

        assert detector is not None

    def test_detect_jump_phases_simple(self):
        """Should detect jump phases from simple trajectory."""
        detector = PhaseDetector()

        # Create simple jump trajectory: baseline -> peak -> baseline
        poses = np.zeros((50, 17, 2), dtype=np.float32)

        # Set hip Y to simulate jump with clear phases
        for i in range(50):
            if i < 10:
                poses[i, H36Key.LHIP, 1] = 0.3  # left_hip baseline
                poses[i, H36Key.RHIP, 1] = 0.3  # right_hip baseline
            elif i < 20:
                # Rising phase (10 frames)
                progress = (i - 10) / 10
                poses[i, H36Key.LHIP, 1] = 0.3 - 0.2 * progress
                poses[i, H36Key.RHIP, 1] = 0.3 - 0.2 * progress
            elif i < 30:
                # Hang time at peak (10 frames)
                poses[i, H36Key.LHIP, 1] = 0.1  # peak height
                poses[i, H36Key.RHIP, 1] = 0.1
            else:
                # Landing phase (20 frames)
                progress = (i - 30) / 20
                poses[i, H36Key.LHIP, 1] = 0.1 + 0.2 * progress
                poses[i, H36Key.RHIP, 1] = 0.1 + 0.2 * progress

        result = detector.detect_jump_phases(poses, fps=30.0)

        assert isinstance(result, PhaseDetectionResult)
        assert result.phases.takeoff < result.phases.peak
        assert result.phases.peak < result.phases.landing
        assert result.confidence > 0

    def test_detect_three_turn_phases(self):
        """Should detect three-turn phases."""
        detector = PhaseDetector()

        # Create simple trajectory with edge change
        # H3.6M 17 format
        poses = np.zeros((50, 17, 2), dtype=np.float32)

        result = detector.detect_three_turn_phases(poses, fps=30.0)

        assert isinstance(result, PhaseDetectionResult)
        assert result.phases.end < len(poses)

    def test_detect_phases_jump(self):
        """Should route jump elements correctly."""
        detector = PhaseDetector()

        poses = np.zeros((50, 17, 2), dtype=np.float32)

        result = detector.detect_phases(poses, fps=30.0, element_type="waltz_jump")

        assert isinstance(result, PhaseDetectionResult)
        assert result.phases.name == "jump"

    def test_detect_phases_three_turn(self):
        """Should route step elements correctly."""
        detector = PhaseDetector()

        poses = np.zeros((50, 17, 2), dtype=np.float32)

        result = detector.detect_phases(poses, fps=30.0, element_type="three_turn")

        assert isinstance(result, PhaseDetectionResult)
        assert result.phases.name == "three_turn"

    def test_detect_phases_unknown_element(self):
        """Should handle unknown elements gracefully."""
        detector = PhaseDetector()

        poses = np.zeros((50, 17, 2), dtype=np.float32)

        result = detector.detect_phases(poses, fps=30.0, element_type="unknown")

        assert isinstance(result, PhaseDetectionResult)
        # Should return default phases with low confidence
        assert result.confidence == 0

    def test_find_takeoff_before_peak(self):
        """Should find takeoff frame before peak."""
        detector = PhaseDetector()

        derivative = np.array([0, 0, -0.1, -0.1, -0.05, 0, 0])
        peak_idx = 5

        takeoff = detector._find_takeoff(derivative, peak_idx)

        assert takeoff < peak_idx
        assert takeoff >= 0

    def test_find_landing_after_peak(self):
        """Should find landing frame after peak."""
        detector = PhaseDetector()

        hip_y = np.array([0.3, 0.2, 0.1, 0.2, 0.3, 0.3])
        peak_idx = 2
        takeoff_idx = 0

        landing = detector._find_landing(hip_y, peak_idx, takeoff_idx)

        assert landing > peak_idx
        assert landing < len(hip_y)


class TestPhaseDetectionResult:
    """Test PhaseDetectionResult dataclass."""

    def test_phase_detection_result_creation(self):
        """Should create result correctly."""
        phases = ElementPhase(
            name="test",
            start=0,
            takeoff=10,
            peak=20,
            landing=30,
            end=40,
        )

        result = PhaseDetectionResult(phases=phases, confidence=0.8)

        assert result.phases == phases
        assert result.confidence == 0.8


class TestParabolicFlightDetector:
    """Test parabolic CoM fitting for jump phase detection."""

    @staticmethod
    def _make_jump_poses(
        n_frames: int,
        takeoff_frame: int,
        landing_frame: int,
        peak_y: float = 0.1,
        baseline_y: float = 0.4,
    ) -> np.ndarray:
        """Create normalized poses with a parabolic CoM trajectory.

        Uses image coordinates: lower Y = higher person.
        During flight the CoM follows a parabola opening upward.
        """
        poses = np.full((n_frames, 17, 2), baseline_y, dtype=np.float32)
        # Put all keypoints at baseline so CoM ≈ baseline
        # During jump, shift PELVIS/HIP/KNEE to simulate CoM rising

        flight_start = takeoff_frame
        flight_end = landing_frame
        peak_frame = (flight_start + flight_end) // 2

        for f in range(flight_start, flight_end + 1):
            # Parabolic: y = a*(t-t0)^2 + peak_y, opens upward
            t = f - peak_frame
            half_span = max(1, (flight_end - flight_start) / 2.0)
            a = (baseline_y - peak_y) / (half_span**2)
            y = a * (t**2) + peak_y
            # Set all keypoints to this y to make CoM ≈ y
            poses[f, :, 1] = y
            # Keep x at baseline (doesn't affect CoM Y)
            poses[f, :, 0] = 0.5

        return poses

    def test_parabolic_detects_clean_jump(self):
        """Parabolic method should detect takeoff < peak < landing for clean jump."""
        detector = PhaseDetector()

        # 60-frame sequence with jump in frames 15-45
        poses = self._make_jump_poses(
            n_frames=60,
            takeoff_frame=15,
            landing_frame=45,
            peak_y=0.1,
            baseline_y=0.4,
        )

        result = detector._detect_jump_phases_parabolic(poses, fps=30.0)

        assert isinstance(result, PhaseDetectionResult)
        assert result.phases.takeoff < result.phases.peak
        assert result.phases.peak < result.phases.landing
        # Peak should be roughly in the middle of the jump
        assert 20 <= result.phases.peak <= 40

    def test_parabolic_ignores_prep_movement(self):
        """Parabolic method should ignore prep movements and find real jump.

        Frames 10-25: shoulder lean (flat CoM, not parabolic).
        Frames 40-55: real parabolic jump.
        Takeoff should be well past the prep (>= 30).
        """
        n = 70
        poses = np.full((n, 17, 2), 0.4, dtype=np.float32)
        poses[:, :, 0] = 0.5  # constant x

        # Preparation: frames 10-25 — slight shoulder lean but flat CoM
        for f in range(10, 26):
            # Move shoulders but keep hips (majority mass) at baseline
            poses[f, H36Key.LSHOULDER, 1] = 0.38
            poses[f, H36Key.RSHOULDER, 1] = 0.38
            poses[f, H36Key.LELBOW, 1] = 0.37
            poses[f, H36Key.RELBOW, 1] = 0.37
            # Hips, knees, feet stay at baseline → CoM barely moves

        # Real jump: frames 40-55 — parabolic CoM
        peak_y = 0.1
        baseline_y = 0.4
        flight_start, flight_end = 40, 55
        peak_frame = (flight_start + flight_end) // 2
        for f in range(flight_start, flight_end + 1):
            t = f - peak_frame
            half_span = max(1, (flight_end - flight_start) / 2.0)
            a = (baseline_y - peak_y) / (half_span**2)
            y = a * (t**2) + peak_y
            poses[f, :, 1] = y

        detector = PhaseDetector()
        result = detector._detect_jump_phases_parabolic(poses, fps=30.0)

        assert isinstance(result, PhaseDetectionResult)
        # Takeoff should be well past prep (frame 25)
        assert result.phases.takeoff >= 30
        assert result.phases.takeoff < result.phases.peak
        assert result.phases.peak < result.phases.landing

    def test_parabolic_fallback_on_no_jump(self):
        """Flat poses should fallback to velocity method and still return a result."""
        detector = PhaseDetector()

        # Completely flat poses — no jump at all
        poses = np.full((50, 17, 2), 0.4, dtype=np.float32)
        poses[:, :, 0] = 0.5

        result = detector._detect_jump_phases_parabolic(poses, fps=30.0)

        assert isinstance(result, PhaseDetectionResult)
        # Should still return a valid result (fallback kicks in)
        assert result.phases is not None

    def test_parabolic_short_sequence(self):
        """Very short sequence should not crash."""
        detector = PhaseDetector()

        # Only 10 frames — too short for any real jump
        poses = np.full((10, 17, 2), 0.4, dtype=np.float32)
        poses[:, :, 0] = 0.5

        result = detector._detect_jump_phases_parabolic(poses, fps=30.0)

        assert isinstance(result, PhaseDetectionResult)
        assert result.phases is not None
