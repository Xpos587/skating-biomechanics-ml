"""Automatic phase detection for skating elements.

This module detects key phases like takeoff, peak height, and landing
from pose sequences using biomechanical cues.
"""

import numpy as np
from scipy.signal import find_peaks

from ..types import ElementPhase, NormalizedPose
from ..utils.geometry import calculate_com_trajectory
from .element_defs import ElementDef
from .metrics import BiomechanicsAnalyzer, PhaseDetectionResult

# NOTE: 2D blade detector removed in 3D-only migration
# Phase detection now uses CoM-based method only


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
        if element_type in ("waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"):
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

        Uses parabolic CoM fitting to identify true flight phases.
        Falls back to velocity-based detection when no parabola passes quality checks.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            fps: Frame rate.

        Returns:
            PhaseDetectionResult with jump phase boundaries.
        """
        return self._detect_jump_phases_parabolic(poses, fps)

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
        takeoff_candidates, _takeoff_props = find_peaks(vy, height=2 * vy_std, distance=10)

        # Detect landing: negative velocity spike (impact)
        # Use 3-sigma threshold for robustness against false positives
        landing_candidates, _landing_props = find_peaks(-vy, height=3 * vy_std, distance=10)

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

    def _detect_jump_phases_parabolic(
        self, poses: NormalizedPose, fps: float
    ) -> PhaseDetectionResult:
        """Detect jump phases by fitting parabolas to CoM trajectory segments.

        During true flight the CoM follows a parabolic arc (gravity only).
        Preparation movements (leg swings, crouches) produce flat or noisy CoM
        that does not fit a parabola well. By sliding over elevated segments
        and fitting y(t) = at² + bt + c, we discriminate real flight from prep.

        Falls back to :meth:`_detect_jump_phases_com_improved` when no segment
        passes the quality checks.

        Args:
            poses: NormalizedPose (num_frames, 17, 2).
            fps: Frame rate.

        Returns:
            PhaseDetectionResult with jump phase boundaries.
        """
        from scipy.ndimage import median_filter as _median_filter

        N = len(poses)
        if N < 12:
            # Too short for any meaningful analysis — fallback
            return self._detect_jump_phases_com_improved(poses, fps)

        # 1. Compute CoM trajectory
        com_y = calculate_com_trajectory(poses)

        # 2. Smooth with median filter (remove spikes)
        com_smooth = _median_filter(com_y.astype(np.float64), size=5)

        # 3. Compute baseline via large-window median
        baseline_win = min(61, max(21, N // 3))
        baseline = _median_filter(com_smooth, size=baseline_win)

        # 4. Threshold: standard deviation of excursion
        excursion = com_smooth - baseline
        threshold = float(np.std(excursion))
        if threshold < 1e-6:
            # Essentially flat — no jump present
            return self._detect_jump_phases_com_improved(poses, fps)

        # 5. Find contiguous elevated segments (com_smooth < baseline - threshold)
        #    In image coords, lower Y means person is higher.
        elevated = com_smooth < baseline - threshold

        # 6. Extract segments, merge close ones (gap < 3 frames)
        segments: list[tuple[int, int]] = []
        seg_start = None
        for i in range(N):
            if elevated[i]:
                if seg_start is None:
                    seg_start = i
            else:
                if seg_start is not None:
                    # Check if gap to previous segment is small → merge
                    if segments and (i - segments[-1][1]) < 3:
                        segments[-1] = (segments[-1][0], i - 1)
                    else:
                        segments.append((seg_start, i - 1))
                    seg_start = None
        # Handle segment that runs to end
        if seg_start is not None:
            if segments and (N - 1 - segments[-1][1]) < 3:
                segments[-1] = (segments[-1][0], N - 1)
            else:
                segments.append((seg_start, N - 1))

        # 7. Filter by minimum duration
        min_dur = max(5, int(0.2 * fps))
        segments = [(s, e) for s, e in segments if (e - s + 1) >= min_dur]

        if not segments:
            return self._detect_jump_phases_com_improved(poses, fps)

        # 8. For each segment, extend, fit parabola, score
        best_score = -1.0
        best_result: PhaseDetectionResult | None = None

        for seg_start, seg_end in segments:
            # Extend by 3 frames each side for better fit
            ext_start = max(0, seg_start - 3)
            ext_end = min(N - 1, seg_end + 3)

            t_local = np.arange(ext_start, ext_end + 1, dtype=np.float64)
            y_local = com_smooth[ext_start:ext_end + 1]

            if len(t_local) < 4:
                continue

            # Fit parabola: y(t) = a*t^2 + b*t + c
            coeffs, residuals, _rank, _sv, _rcond = np.polyfit(
                t_local, y_local, 2, full=True
            )
            a_coeff = coeffs[0]
            b_coeff = coeffs[1]

            # Compute R²
            y_pred = np.polyval(coeffs, t_local)
            ss_res = float(np.sum((y_local - y_pred) ** 2))
            ss_tot = float(np.sum((y_local - np.mean(y_local)) ** 2))
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

            # Peak of parabola
            parabola_peak_t = -b_coeff / (2.0 * a_coeff) if abs(a_coeff) > 1e-12 else -1.0

            # 9. Quality checks
            #    a > 0: parabola opens upward (image coords: lower y = higher person)
            #    Peak inside segment
            #    R² > 0.80
            peak_inside = seg_start <= parabola_peak_t <= seg_end
            if a_coeff > 0 and peak_inside and r_squared > 0.80:
                peak_frame = int(round(parabola_peak_t))

                # 11. Find takeoff/landing by scanning to baseline crossing
                takeoff_idx = self._scan_to_baseline(
                    com_smooth, baseline, peak_frame, direction=-1
                )
                landing_idx = self._scan_to_baseline(
                    com_smooth, baseline, peak_frame, direction=+1
                )

                # Validate order
                if takeoff_idx >= peak_frame:
                    takeoff_idx = max(0, peak_frame - 5)
                if landing_idx <= peak_frame:
                    landing_idx = min(N - 1, peak_frame + 5)

                # Validate physical plausibility
                airtime = (landing_idx - takeoff_idx) / fps
                if airtime < 0.3:
                    continue

                # Score = R² × excursion magnitude
                seg_excursion = float(
                    baseline[peak_frame] - com_smooth[peak_frame]
                )
                score = r_squared * seg_excursion

                if score > best_score:
                    best_score = score

                    # Build phase boundaries
                    start_idx = max(0, takeoff_idx - 10)
                    end_idx = min(N - 1, landing_idx + 10)

                    # Confidence from R² and excursion
                    confidence = min(1.0, r_squared * min(1.0, seg_excursion / 0.1))

                    best_result = PhaseDetectionResult(
                        phases=ElementPhase(
                            name="jump",
                            start=start_idx,
                            takeoff=takeoff_idx,
                            peak=peak_frame,
                            landing=landing_idx,
                            end=end_idx,
                        ),
                        confidence=confidence,
                    )

        # 12. Fallback if no good parabola found
        if best_result is None:
            return self._detect_jump_phases_com_improved(poses, fps)

        return best_result

    @staticmethod
    def _scan_to_baseline(
        com_smooth: np.ndarray,
        baseline: np.ndarray,
        peak_frame: int,
        direction: int,
    ) -> int:
        """Scan backward or forward from peak to find baseline crossing.

        Args:
            com_smooth: Smoothed CoM trajectory.
            baseline: Baseline trajectory.
            peak_frame: Peak frame index.
            direction: -1 for backward (takeoff), +1 for forward (landing).

        Returns:
            Frame index where CoM returns to baseline.
        """
        N = len(com_smooth)
        i = peak_frame
        while 0 <= i < N:
            if com_smooth[i] >= baseline[i]:
                return i
            i += direction
        # Reached boundary
        return i - direction  # last valid index in scan direction

    def detect_three_turn_phases(
        self,
        poses: NormalizedPose,
        fps: float,
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
