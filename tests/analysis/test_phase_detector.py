"""Tests for automatic phase detection."""

import numpy as np
import pytest

from src.phase_detector import PhaseDetector
from src.metrics import PhaseDetectionResult
from src.types import H36Key, ElementPhase


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
