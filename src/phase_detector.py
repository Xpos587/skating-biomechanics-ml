"""Automatic phase detection for skating elements.

This module detects key phases like takeoff, peak height, and landing
from pose sequences using biomechanical cues.
"""

import numpy as np
from scipy.signal import find_peaks

from .metrics import BiomechanicsAnalyzer, PhaseDetectionResult
from .element_defs import ElementDef
from .types import ElementPhase, NormalizedPose
from .geometry import calculate_com_trajectory

# BladeEdgeDetector is optional (requires 3D poses)
try:
    from . import blade_edge_detector

    BladeEdgeDetector = blade_edge_detector.BladeEdgeDetector
    BLADE_DETECTOR_AVAILABLE = True
except Exception:
    BLADE_DETECTOR_AVAILABLE = False


class PhaseDetector:
    """Detect phases of skating elements from pose sequences."""

    def detect_phases(
        self,
        poses: NormalizedPose,
        fps: float,
        element_type: str,
    ) -> PhaseDetectionResult:
        """Detect phases for a skating element.

        Args:
            poses: NormalizedPose (num_frames, 33, 2).
            fps: Frame rate.
            element_type: Type of element (jump or step).

        Returns:
            PhaseDetectionResult with detected boundaries and confidence.
        """
        if element_type in ("waltz_jump", "toe_loop", "flip"):
            return self.detect_jump_phases(poses, fps)
        elif element_type == "three_turn":
            return self.detect_three_turn_phases(poses, fps)
        else:
            # Default: entire sequence as one phase
            return PhaseDetectionResult(
                phases=ElementPhase(
                    name=element_type,
                    start=0,
                    takeoff=0,
                    peak=0,
                    landing=0,
                    end=len(poses) - 1,
                ),
                confidence=0.0,
            )

    def detect_jump_phases(self, poses: NormalizedPose, fps: float) -> PhaseDetectionResult:
        """Detect jump phases: takeoff, peak, landing.

        Uses improved Center of Mass (CoM) trajectory with velocity-based detection
        and adaptive sigma-based thresholds for better accuracy across different
        video qualities and jump types. Falls back to blade detection if CoM fails.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            fps: Frame rate.

        Returns:
            PhaseDetectionResult with jump phase boundaries.
        """
        # Try improved CoM-based detection first (adaptive thresholds)
        com_result = self._detect_jump_phases_com_improved(poses, fps)

        # If low confidence and blade detector is available, try blade detection as backup
        if com_result.confidence < 0.5 and BLADE_DETECTOR_AVAILABLE:
            blade_result = self._detect_jump_phases_blade(poses, fps)
            # Use the result with higher confidence
            if blade_result.confidence > com_result.confidence:
                return blade_result

        return com_result

    def _detect_jump_phases_com(self, poses: NormalizedPose, fps: float) -> PhaseDetectionResult:
        """Detect jump phases using Center of Mass trajectory.

        Uses acceleration spikes to detect takeoff (upward acceleration)
        and landing (downward acceleration/impact). This is the physics-accurate
        method that fixes the 60% error in hip-only methods.

        Args:
            poses: NormalizedPose (num_frames, 33, 2).
            fps: Frame rate.

        Returns:
            PhaseDetectionResult with jump phase boundaries.
        """
        # Calculate CoM trajectory
        com_y = calculate_com_trajectory(poses)

        # Find peaks (local minima in Y = maxima in height)
        peaks, properties = find_peaks(-com_y, prominence=0.02, distance=10)

        if len(peaks) == 0:
            # No clear jump detected
            return PhaseDetectionResult(
                phases=ElementPhase(
                    name="jump",
                    start=0,
                    takeoff=0,
                    peak=len(poses) // 2,
                    landing=len(poses) - 1,
                    end=len(poses) - 1,
                ),
                confidence=0.0,
            )

        # Use highest peak
        peak_idx = peaks[np.argmax(-properties["prominences"])]

        # Detect takeoff using acceleration spike
        takeoff_idx = self._find_takeoff_accel(com_y, fps, peak_idx)

        # Detect landing using negative acceleration (impact)
        landing_idx = self._find_landing_accel(com_y, fps, peak_idx, takeoff_idx)

        # Validate phases
        if takeoff_idx >= peak_idx:
            takeoff_idx = max(0, peak_idx - 10)
        if landing_idx <= peak_idx:
            landing_idx = min(len(poses) - 1, peak_idx + 10)

        # Set boundaries
        start_idx = max(0, takeoff_idx - 10)
        end_idx = min(len(poses) - 1, landing_idx + 10)

        phases = ElementPhase(
            name="jump",
            start=start_idx,
            takeoff=takeoff_idx,
            peak=peak_idx,
            landing=landing_idx,
            end=end_idx,
        )

        # Confidence based on peak prominence
        prominence = float(properties["prominences"][np.argmax(-properties["prominences"])])
        confidence = min(1.0, prominence / 0.1)

        return PhaseDetectionResult(phases=phases, confidence=confidence)

    def _detect_jump_phases_com_improved(
        self, poses: NormalizedPose, fps: float
    ) -> PhaseDetectionResult:
        """Improved jump phase detection using CoM velocity with adaptive thresholds.

        Uses vertical CoM velocity with adaptive sigma-based thresholds instead of
        fixed prominence values. This provides better detection across different
        video qualities and jump types.

        Args:
            poses: NormalizedPose (num_frames, 33, 2).
            fps: Frame rate.

        Returns:
            PhaseDetectionResult with improved jump phase boundaries.
        """
        # Calculate CoM trajectory
        com_y = calculate_com_trajectory(poses)

        # Calculate vertical velocity (positive = upward, negative = downward)
        vy = np.gradient(com_y) * fps

        # Calculate standard deviation for adaptive thresholds
        vy_std = np.std(vy)

        # Detect takeoff: positive velocity peak (skater pushing upward)
        # Use 2-sigma threshold for sensitivity
        takeoff_candidates, takeoff_props = find_peaks(vy, height=2 * vy_std, distance=10)

        # Detect landing: negative velocity spike (impact)
        # Use 3-sigma threshold for robustness against false positives
        landing_candidates, landing_props = find_peaks(-vy, height=3 * vy_std, distance=10)

        # Find peak: minimum CoM Y (maximum height)
        if len(takeoff_candidates) > 0 and len(landing_candidates) > 0:
            # Peak should be between takeoff and landing
            first_takeoff = takeoff_candidates[0]
            first_landing = landing_candidates[0]

            # Search for peak in the expected region
            search_start = max(0, first_takeoff)
            search_end = min(len(poses), first_landing + 1)

            com_y_search = com_y[search_start:search_end]
            if len(com_y_search) > 0:
                peak_offset = np.argmin(com_y_search)
                peak_idx = search_start + peak_offset
            else:
                peak_idx = len(poses) // 2
        else:
            # Fallback to simple peak detection
            peaks, properties = find_peaks(-com_y, prominence=0.02, distance=10)
            if len(peaks) == 0:
                peak_idx = len(poses) // 2
            else:
                peak_idx = peaks[np.argmax(-properties["prominences"])]

        # Set takeoff and landing indices
        if len(takeoff_candidates) > 0:
            takeoff_idx = takeoff_candidates[0]
        else:
            takeoff_idx = max(0, peak_idx - 10)

        if len(landing_candidates) > 0:
            # Find first landing after peak
            valid_landings = landing_candidates[landing_candidates > peak_idx]
            if len(valid_landings) > 0:
                landing_idx = valid_landings[0]
            else:
                landing_idx = min(len(poses) - 1, peak_idx + 10)
        else:
            landing_idx = min(len(poses) - 1, peak_idx + 10)

        # Validate physical plausibility
        airtime = (landing_idx - takeoff_idx) / fps

        # Minimum airtime validation (0.3 seconds)
        if airtime < 0.3:
            # Airtime too short, likely false positive
            return PhaseDetectionResult(
                phases=ElementPhase(
                    name="jump",
                    start=0,
                    takeoff=0,
                    peak=len(poses) // 2,
                    landing=len(poses) - 1,
                    end=len(poses) - 1,
                ),
                confidence=0.0,
            )

        # Validate order: takeoff < peak < landing
        if takeoff_idx >= peak_idx:
            takeoff_idx = max(0, peak_idx - 5)

        if landing_idx <= peak_idx:
            landing_idx = min(len(poses) - 1, peak_idx + 5)

        # Set boundaries (include preparation and recovery)
        start_idx = max(0, takeoff_idx - 10)
        end_idx = min(len(poses) - 1, landing_idx + 10)

        phases = ElementPhase(
            name="jump",
            start=start_idx,
            takeoff=takeoff_idx,
            peak=peak_idx,
            landing=landing_idx,
            end=end_idx,
        )

        # Confidence based on multiple factors
        # 1. Peak prominence (how distinct the jump is)
        if takeoff_idx < peak_idx < landing_idx:
            flight_com = com_y[takeoff_idx : landing_idx + 1]
            prominence = float(np.max(flight_com) - np.min(flight_com))
        else:
            prominence = 0.01

        # 2. Velocity signal strength (how clear the takeoff/landing is)
        takeoff_signal = abs(vy[takeoff_idx]) if takeoff_idx < len(vy) else 0
        landing_signal = abs(vy[landing_idx]) if landing_idx < len(vy) else 0

        # Combine factors
        confidence = min(
            1.0,
            (
                min(1.0, prominence / 0.05) * 0.5  # Peak prominence (max 0.05)
                + min(1.0, takeoff_signal / (2 * vy_std)) * 0.3  # Takeoff clarity
                + min(1.0, landing_signal / (3 * vy_std)) * 0.2  # Landing clarity
            ),
        )

        return PhaseDetectionResult(phases=phases, confidence=confidence)

    def _detect_jump_phases_blade(self, poses: NormalizedPose, fps: float) -> PhaseDetectionResult:
        """Detect jump phases using blade edge detection as backup.

        Uses blade state transitions (edge → toe pick → edge) to detect
        takeoff and landing.

        Args:
            poses: NormalizedPose (num_frames, 33, 2).
            fps: Frame rate.

        Returns:
            PhaseDetectionResult with jump phase boundaries.
        """
        detector = BladeEdgeDetector(smoothing_window=3)

        # Detect blade states for both feet
        left_states = detector.detect_sequence(poses, fps, foot="left")
        right_states = detector.detect_sequence(poses, fps, foot="right")

        # Use left foot as primary (takeoff foot for most jumps)
        takeoff, landing = detector.detect_takeoff_landing(left_states, fps)

        # If left foot detection failed, try right foot
        if takeoff is None or landing is None:
            takeoff, landing = detector.detect_takeoff_landing(right_states, fps)

        # Fallback values
        if takeoff is None:
            takeoff = 0
        if landing is None:
            landing = len(poses) - 1

        # Find peak between takeoff and landing
        com_y = calculate_com_trajectory(poses)
        flight_y = com_y[takeoff : landing + 1]
        peak_offset = np.argmin(flight_y)  # Minimum Y = maximum height
        peak_idx = takeoff + peak_offset

        # Set boundaries
        start_idx = max(0, takeoff - 10)
        end_idx = min(len(poses) - 1, landing + 10)

        phases = ElementPhase(
            name="jump",
            start=start_idx,
            takeoff=takeoff,
            peak=peak_idx,
            landing=landing,
            end=end_idx,
        )

        # Confidence based on blade detection quality
        confidence = 0.6 if takeoff > 0 and landing < len(poses) - 1 else 0.3

        return PhaseDetectionResult(phases=phases, confidence=confidence)

    def detect_three_turn_phases(
        self,
        poses: NormalizedPose,
        fps: float,  # noqa: ARG002
    ) -> PhaseDetectionResult:
        """Detect three-turn phases by edge change.

        Args:
            poses: NormalizedPose (num_frames, 33, 2).
            fps: Frame rate.

        Returns:
            PhaseDetectionResult with turn phase boundaries.
        """
        # Compute edge indicator
        dummy_def = ElementDef(
            name="three_turn",
            name_ru="тройка",
            rotations=0,
            has_toe_pick=False,
            key_joints=[],
            ideal_metrics={},
        )

        analyzer = BiomechanicsAnalyzer(dummy_def)
        edge_ind = analyzer.compute_edge_indicator(poses, side="left")

        # Find edge change point (zero crossing)
        # Compute derivative to find rapid changes
        edge_derivative = np.gradient(edge_ind)

        # Find peak in derivative (maximum rate of change)
        change_points, _ = find_peaks(np.abs(edge_derivative), prominence=0.1, distance=5)

        if len(change_points) == 0:
            # No clear turn detected
            return PhaseDetectionResult(
                phases=ElementPhase(
                    name="three_turn",
                    start=0,
                    takeoff=0,
                    peak=0,
                    landing=0,
                    end=len(poses) - 1,
                ),
                confidence=0.0,
            )

        # Use most prominent change point as turn center
        turn_center = change_points[np.argmax(np.abs(edge_derivative[change_points]))]

        # Set boundaries around turn
        start_idx = max(0, turn_center - 15)
        end_idx = min(len(poses) - 1, turn_center + 15)

        phases = ElementPhase(
            name="three_turn",
            start=start_idx,
            takeoff=0,  # No takeoff for steps
            peak=turn_center,  # Use peak as turn center
            landing=0,  # No landing for steps
            end=end_idx,
        )

        # Confidence based on edge change magnitude
        max_change = float(np.max(np.abs(edge_derivative)))
        confidence = min(1.0, max_change / 0.5)

        return PhaseDetectionResult(phases=phases, confidence=confidence)

    def _find_takeoff_accel(
        self,
        com_y: np.ndarray,
        fps: float,
        peak_idx: int,
    ) -> int:
        """Find takeoff using vertical acceleration spike.

        Detects the impulse moment when skater pushes off the ice by
        finding the positive acceleration spike in CoM trajectory.

        Args:
            com_y: CoM Y trajectory (lower = higher).
            fps: Frame rate.
            peak_idx: Peak frame index.

        Returns:
            Takeoff frame index.
        """
        # Calculate acceleration (second derivative)
        accel_y = np.gradient(np.gradient(com_y))

        # Look backward from peak for acceleration spike
        search_start = max(0, peak_idx - 30)
        search_end = peak_idx

        # Use 3-sigma rule for threshold
        accel_std = float(np.std(accel_y[search_start:search_end]))
        threshold = accel_std * 3.0

        # Find first significant positive acceleration before peak
        for i in range(search_end - 1, search_start, -1):
            if accel_y[i] > threshold:
                # Verify it's the start of sustained acceleration
                window = min(3, search_end - i)
                if np.all(accel_y[i : i + window] > 0):
                    return i

        # Fallback: use derivative-based method
        derivative = np.gradient(com_y)
        return self._find_takeoff_derivative(derivative, peak_idx)

    def _find_landing_accel(
        self,
        com_y: np.ndarray,
        fps: float,
        peak_idx: int,
        takeoff_idx: int,
    ) -> int:
        """Find landing using negative acceleration spike (impact).

        Detects the moment when skater returns to ice by finding the
        negative acceleration spike (impact deceleration).

        Args:
            com_y: CoM Y trajectory (lower = higher).
            fps: Frame rate.
            peak_idx: Peak frame index.
            takeoff_idx: Takeoff frame index.

        Returns:
            Landing frame index.
        """
        # Calculate acceleration (second derivative)
        accel_y = np.gradient(np.gradient(com_y))

        # Look forward from peak for negative spike
        search_start = peak_idx
        search_end = min(len(com_y), peak_idx + 40)

        # Use 2-sigma rule for threshold (more sensitive for landing)
        accel_std = float(np.std(accel_y[search_start:search_end]))
        threshold = -accel_std * 2.0

        # Find first significant negative acceleration after peak
        for i in range(search_start, search_end):
            if accel_y[i] < threshold:
                # Verify it's followed by sustained low acceleration
                window = min(5, search_end - i)
                if np.mean(accel_y[i : i + window]) < 0:
                    return i

        # Fallback: use baseline return method
        return self._find_landing_baseline(com_y, peak_idx, takeoff_idx)

    def _find_takeoff_derivative(self, derivative: np.ndarray, peak_idx: int) -> int:
        """Find takeoff frame using derivative method (fallback).

        Args:
            derivative: CoM Y derivative.
            peak_idx: Peak frame index.

        Returns:
            Takeoff frame index.
        """
        # Look backward from peak for sustained negative derivative
        search_start = max(0, peak_idx - 30)
        search_end = peak_idx

        # Find where derivative becomes consistently negative
        for i in range(search_end, search_start, -1):
            if derivative[i] < -0.01:
                window = min(5, search_end - i)
                if np.all(derivative[i : i + window] < 0):
                    return i

        return max(0, peak_idx - 10)

    def _find_landing_baseline(
        self,
        com_y: np.ndarray,
        peak_idx: int,
        takeoff_idx: int,
    ) -> int:
        """Find landing frame using baseline return method (fallback).

        Args:
            com_y: CoM Y trajectory.
            peak_idx: Peak frame index.
            takeoff_idx: Takeoff frame index.

        Returns:
            Landing frame index.
        """
        # Get CoM at takeoff (baseline)
        baseline_y = com_y[takeoff_idx]

        # Look forward from peak for return to baseline
        search_start = peak_idx
        search_end = min(len(com_y), peak_idx + 30)

        for i in range(search_start, search_end):
            if abs(com_y[i] - baseline_y) < 0.02:
                window = min(5, search_end - i)
                if np.all(np.abs(com_y[i : i + window] - baseline_y) < 0.03):
                    return i

        return min(len(com_y) - 1, peak_idx + 10)

    def _find_takeoff(self, derivative: np.ndarray, peak_idx: int) -> int:
        """Find takeoff frame before peak (deprecated, use _find_takeoff_accel).

        Args:
            derivative: Hip Y derivative.
            peak_idx: Peak frame index.

        Returns:
            Takeoff frame index.
        """
        # Look backward from peak for sustained negative derivative
        search_start = max(0, peak_idx - 30)
        search_end = peak_idx

        # Find where derivative becomes consistently negative
        for i in range(search_end, search_start, -1):
            # Check if derivative is negative and significant
            if derivative[i] < -0.01:
                # Check if sustained for next few frames
                window = min(5, search_end - i)
                if np.all(derivative[i : i + window] < 0):
                    return i

        # Fallback: fixed frames before peak
        return max(0, peak_idx - 10)

    def _find_landing(
        self,
        hip_y: np.ndarray,
        peak_idx: int,
        takeoff_idx: int,
    ) -> int:
        """Find landing frame after peak (deprecated, use _find_landing_accel).

        Args:
            hip_y: Hip Y coordinates.
            peak_idx: Peak frame index.
            takeoff_idx: Takeoff frame index.

        Returns:
            Landing frame index.
        """
        # Get hip Y at takeoff (baseline)
        baseline_y = hip_y[takeoff_idx]

        # Look forward from peak for return to baseline
        search_start = peak_idx
        search_end = min(len(hip_y), peak_idx + 30)

        for i in range(search_start, search_end):
            # Check if hip Y has returned to near baseline
            if abs(hip_y[i] - baseline_y) < 0.02:
                # Verify it stays near baseline
                window = min(5, search_end - i)
                if np.all(np.abs(hip_y[i : i + window] - baseline_y) < 0.03):
                    return i

        # Fallback: fixed frames after peak
        return min(len(hip_y) - 1, peak_idx + 10)
