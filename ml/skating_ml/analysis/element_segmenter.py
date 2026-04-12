"""Automatic element segmentation for skating videos.

Detects element boundaries using motion energy analysis and classifies
each segment using rule-based heuristics.
"""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_opening

from ..types import (
    ElementPhase,
    ElementSegment,
    NormalizedPose,
    SegmentationResult,
)
from ..utils.geometry import get_mid_hip, smooth_signal

if TYPE_CHECKING:
    from ..utils.video import VideoMeta


class ElementSegmenter:
    """Automatic segmentation of skating video into elements.

    Uses motion energy analysis to detect stillness periods between elements,
    then classifies each segment using rule-based heuristics.
    """

    def __init__(
        self,
        stillness_threshold: float | None = None,
        min_still_duration: float = 0.5,
        min_segment_duration: float = 0.5,
        boundary_window: int = 10,
    ) -> None:
        """Initialize element segmenter.

        Args:
            stillness_threshold: Motion energy threshold for stillness (auto if None).
            min_still_duration: Minimum seconds for stillness period (default 0.5s).
            min_segment_duration: Minimum seconds for valid segment (default 0.5s).
            boundary_window: Frames to search for boundary refinement (default 10).
        """
        self._stillness_threshold = stillness_threshold
        self._min_still_duration = min_still_duration
        self._min_segment_duration = min_segment_duration
        self._boundary_window = boundary_window

    def segment(
        self,
        poses: NormalizedPose,
        video_path: Path,
        video_meta: "VideoMeta",
        method: str = "adaptive",
    ) -> SegmentationResult:
        """Detect element boundaries in video.

        Args:
            poses: NormalizedPose sequence (num_frames, 33, 2).
            video_path: Path to original video file.
            video_meta: Video metadata.
            method: Segmentation strategy ("adaptive", "motion_energy").

        Returns:
            SegmentationResult with detected segments.
        """
        # Stage 1: Compute motion energy signal
        motion_energy = self._compute_motion_energy(poses)

        # Stage 2: Detect stillness periods (element separators)
        stillness_mask = self._detect_stillness(motion_energy, video_meta.fps)

        # Stage 3: Extract active segments
        active_segments = self._extract_active_segments(stillness_mask)

        # Stage 4: Filter short segments
        min_frames = int(self._min_segment_duration * video_meta.fps)
        filtered_segments = [(s, e) for s, e in active_segments if e - s >= min_frames]

        # Stage 5: Refine boundaries using velocity/pose changes
        refined_segments = self._refine_boundaries(poses, filtered_segments)

        # Stage 6: Classify each segment
        classified_segments = self._classify_segments(poses, refined_segments, video_meta.fps)

        # Stage 7: Compute overall confidence
        overall_confidence = self._compute_overall_confidence(classified_segments)

        return SegmentationResult(
            segments=classified_segments,
            video_path=video_path,
            video_meta=video_meta,
            method=method,
            confidence=overall_confidence,
        )

    def _compute_motion_energy(self, poses: NormalizedPose) -> NDArray[np.float32]:
        """Compute per-frame motion energy from pose differences.

        Args:
            poses: NormalizedPose sequence (num_frames, 33, 2).

        Returns:
            Motion energy signal (num_frames,).
        """
        # Frame-to-frame difference
        diff = np.diff(poses, axis=0)  # (num_frames-1, 33, 2)

        # L2 norm per frame (sum of all joint movements)
        energy = np.linalg.norm(diff, axis=(1, 2))  # (num_frames-1,)

        # Pad to match original length
        energy = np.pad(energy, (1, 0), mode="edge")

        # Smooth with moving average to reduce noise
        energy = smooth_signal(energy, window=5)

        return energy.astype(np.float32)

    def _detect_stillness(
        self,
        motion_energy: NDArray[np.float32],
        fps: float,
    ) -> NDArray[np.bool_]:
        """Detect stillness periods (element separators).

        Args:
            motion_energy: Per-frame energy signal.
            fps: Frame rate.

        Returns:
            Boolean mask where True = stillness.
        """
        # Adaptive threshold if not provided
        if self._stillness_threshold is None:
            # Use 25th percentile as threshold
            threshold = np.percentile(motion_energy, 25)
        else:
            threshold = self._stillness_threshold

        # Binary mask: energy below threshold = stillness
        still = motion_energy < threshold

        # Morphological opening to remove short noise bursts
        min_frames = int(self._min_still_duration * fps)
        if min_frames > 1:
            still = cast(
                "NDArray[np.bool_]",
                binary_opening(still, structure=np.ones(min_frames, dtype=bool)),
            )

        return still

    def _extract_active_segments(self, stillness_mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
        """Extract active segments from stillness mask.

        Args:
            stillness_mask: Boolean mask where True = stillness.

        Returns:
            List of (start_frame, end_frame) tuples.
        """
        # Find transitions
        transitions = np.diff(stillness_mask.astype(int))
        starts = np.where(transitions == -1)[0] + 1  # still -> active
        ends = np.where(transitions == 1)[0] + 1  # active -> still

        # Handle edge cases
        segments: list[tuple[int, int]] = []

        # If video starts with active segment
        if len(stillness_mask) > 0 and not stillness_mask[0]:
            start = 0
            if len(ends) > 0:
                segments.append((start, ends[0]))

        # Process interior segments
        for i, (s, e) in enumerate(zip(starts, ends, strict=False)):
            if i + 1 < len(starts) and s < e:
                segments.append((int(s), int(e)))

        # If video ends with active segment
        if len(starts) > 0 and len(ends) > 0 and starts[-1] > ends[-1]:
            segments.append((int(starts[-1]), len(stillness_mask)))

        return segments

    def _refine_boundaries(
        self,
        poses: NormalizedPose,
        segments: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Refine segment boundaries using velocity minima.

        Args:
            poses: NormalizedPose sequence.
            segments: List of (start, end) tuples.

        Returns:
            Refined list of (start, end) tuples.
        """
        if not segments:
            return segments

        refined = []

        # Compute hip velocity for boundary refinement
        hip_y = get_mid_hip(poses)[:, 1]
        velocity = np.gradient(hip_y)
        velocity_mag = np.abs(velocity)

        for start, end in segments:
            # Refine start boundary
            if start > self._boundary_window:
                search_start = start - self._boundary_window
                search_end = start + self._boundary_window
            else:
                search_start = 0
                search_end = min(start + self._boundary_window, len(poses))

            # Find velocity minimum in search window
            window_vel = velocity_mag[search_start:search_end]
            if len(window_vel) > 0:
                min_idx = np.argmin(window_vel)
                new_start = search_start + min_idx
            else:
                new_start = start

            # Refine end boundary
            if end + self._boundary_window < len(poses):
                search_start = max(0, end - self._boundary_window)
                search_end = end + self._boundary_window
            else:
                search_start = max(0, end - self._boundary_window)
                search_end = len(poses)

            window_vel = velocity_mag[search_start:search_end]
            if len(window_vel) > 0:
                min_idx = np.argmin(window_vel)
                new_end = search_start + min_idx
            else:
                new_end = end

            # Ensure segment is valid
            if new_start < new_end:
                refined.append((int(new_start), int(new_end)))

        return refined

    def _classify_segments(
        self,
        poses: NormalizedPose,
        segments: list[tuple[int, int]],
        fps: float,
    ) -> list[ElementSegment]:
        """Classify each segment as an element type.

        Args:
            poses: Full video poses.
            segments: List of (start, end) tuples.
            fps: Frame rate.

        Returns:
            List of ElementSegment objects.
        """
        classified = []

        for start, end in segments:
            segment_poses = poses[start:end]

            # Extract features
            features = self._extract_segment_features(segment_poses, fps)

            # Classify by rules
            element_type, confidence = self._classify_by_rules(features)

            # Detect phases for this segment
            phases = self._detect_segment_phases(segment_poses, fps, element_type, start)

            # Create segment
            segment = ElementSegment(
                element_type=element_type,
                start=start,
                end=end,
                confidence=confidence,
                phases=phases,
                metadata=features,
            )

            classified.append(segment)

        return classified

    def _extract_segment_features(
        self,
        poses: NormalizedPose,
        fps: float,
    ) -> dict[str, float | int | bool]:
        """Extract features for classification.

        Args:
            poses: Segment poses.
            fps: Frame rate.

        Returns:
            Dictionary of features.
        """
        features: dict[str, float | int | bool] = {}

        # Duration
        num_frames = len(poses)
        features["duration_sec"] = round(num_frames / fps, 3)
        features["duration_frames"] = num_frames

        # Motion energy
        motion_energy = self._compute_motion_energy(poses)
        features["motion_energy_mean"] = float(np.mean(motion_energy))
        features["motion_energy_std"] = float(np.std(motion_energy))
        features["motion_energy_max"] = float(np.max(motion_energy))

        # Hip Y trajectory (for jumps)
        hip_y = get_mid_hip(poses)[:, 1]
        features["hip_y_range"] = float(np.max(hip_y) - np.min(hip_y))
        features["hip_y_min_idx"] = int(np.argmin(hip_y))

        # Detect jump-like pattern
        hip_y_derivative = np.gradient(hip_y)
        has_takeoff = np.any(hip_y_derivative < -0.02)  # Rapid rise (negative Y is up)
        has_landing = np.any(hip_y_derivative > 0.02)  # Rapid descent
        features["has_jump_pattern"] = bool(has_takeoff and has_landing)

        # Edge indicator (for steps/turns)
        edge_ind = self._compute_edge_indicator(poses)
        features["edge_change_count"] = int(np.sum(np.abs(np.diff(edge_ind)) > 0.3))
        features["edge_indicator_mean"] = float(np.mean(np.abs(edge_ind)))

        # Rotation speed (shoulder axis)
        shoulder_angles = self._compute_shoulder_rotation(poses)
        if len(shoulder_angles) > 0:
            rot_velocity = np.abs(np.gradient(shoulder_angles)) * fps
            features["rotation_speed_max"] = float(np.max(rot_velocity))
            features["rotation_speed_mean"] = float(np.mean(rot_velocity))
        else:
            features["rotation_speed_max"] = 0.0
            features["rotation_speed_mean"] = 0.0

        # Knee angle range
        knee_angles = self._compute_knee_angle_series(poses, side="left")
        if len(knee_angles) > 0:
            features["knee_angle_min"] = float(np.min(knee_angles))
            features["knee_angle_max"] = float(np.max(knee_angles))
            features["knee_angle_range"] = features["knee_angle_max"] - features["knee_angle_min"]
        else:
            features["knee_angle_min"] = 0.0
            features["knee_angle_max"] = 0.0
            features["knee_angle_range"] = 0.0

        return features

    def _classify_by_rules(self, features: dict[str, float | int | bool]) -> tuple[str, float]:
        """Rule-based element classification.

        Args:
            features: Extracted segment features.

        Returns:
            (element_type, confidence) tuple.
        """
        # Decision tree based on features

        # Check for jump (hip Y pattern + rotation)
        if features.get("has_jump_pattern", False):
            rotation_max = features.get("rotation_speed_max", 0)
            if isinstance(rotation_max, (int, float)) and rotation_max > 200:
                # Classify jump type by rotation speed
                if rotation_max > 500:
                    return "flip", 0.7
                elif rotation_max > 350:
                    return "toe_loop", 0.7
                else:
                    return "waltz_jump", 0.65
            elif isinstance(rotation_max, (int, float)) and rotation_max > 100:
                return "waltz_jump", 0.6

        # Check for turn/step (edge changes)
        edge_changes = features.get("edge_change_count", 0)
        if isinstance(edge_changes, (int, float)) and edge_changes > 0:
            return "three_turn", 0.7

        return "unknown", 0.3

    def _detect_segment_phases(
        self,
        poses: NormalizedPose,
        fps: float,
        element_type: str,
        global_start: int,
    ) -> ElementPhase | None:
        """Detect phases for a segment.

        Args:
            poses: Segment poses.
            fps: Frame rate.
            element_type: Element type.
            global_start: Global start frame offset.

        Returns:
            ElementPhase with adjusted frame indices, or None.
        """
        try:
            from . import phase_detector

            PhaseDetector = phase_detector.PhaseDetector

            detector = PhaseDetector()
            result = detector.detect_phases(poses, fps, element_type)

            # Adjust frame indices to global coordinates
            phases = result.phases
            return ElementPhase(
                name=element_type,
                start=phases.start + global_start,
                takeoff=phases.takeoff + global_start if phases.takeoff > 0 else 0,
                peak=phases.peak + global_start if phases.peak > 0 else 0,
                landing=phases.landing + global_start if phases.landing > 0 else 0,
                end=phases.end + global_start,
            )
        except Exception:
            # If phase detection fails, return None
            return None

    def _compute_overall_confidence(self, segments: list[ElementSegment]) -> float:
        """Compute overall segmentation confidence.

        Args:
            segments: List of classified segments.

        Returns:
            Overall confidence score [0, 1].
        """
        if not segments:
            return 0.0

        # Average of segment confidences
        return float(np.mean([s.confidence for s in segments]))

    def _compute_edge_indicator(self, poses: NormalizedPose) -> NDArray[np.float32]:
        """Compute edge indicator for step/turn detection.

        Uses foot velocity direction to estimate edge (inside/outside/flat).

        Args:
            poses: NormalizedPose sequence (H3.6M 17kp format).

        Returns:
            Edge indicator signal (+1=inside, -1=outside, 0=flat).
        """
        # Simplified: use foot velocity direction as edge indicator
        # For H3.6M format, use LFOOT and RFOOT keypoints
        from ..types import H36Key

        left_foot = poses[:, H36Key.LFOOT, :]
        right_foot = poses[:, H36Key.RFOOT, :]

        # Compute foot velocity (difference between consecutive frames)
        left_vel = np.diff(left_foot, axis=0, prepend=left_foot[:1])
        right_vel = np.diff(right_foot, axis=0, prepend=right_foot[:1])

        # Use x-component of velocity as edge indicator
        # (positive = outside edge, negative = inside edge)
        edge_left = np.sign(left_vel[:, 0])
        edge_right = np.sign(right_vel[:, 0])

        # Average both feet
        edge = (edge_left + edge_right) / 2

        return edge.astype(np.float32)

    def _compute_shoulder_rotation(self, poses: NormalizedPose) -> NDArray[np.float32]:
        """Compute shoulder rotation angle over time.

        Args:
            poses: NormalizedPose sequence (H3.6M 17kp format).

        Returns:
            Shoulder rotation angles in radians.
        """
        from ..types import H36Key

        left_shoulder = poses[:, H36Key.LSHOULDER, :]
        right_shoulder = poses[:, H36Key.RSHOULDER, :]

        # Compute shoulder axis vector
        shoulder_vector = right_shoulder - left_shoulder

        # Compute angle relative to horizontal
        angles = np.arctan2(shoulder_vector[:, 1], shoulder_vector[:, 0])

        return angles.astype(np.float32)

    def _compute_knee_angle_series(
        self,
        poses: NormalizedPose,
        side: str = "left",
    ) -> NDArray[np.float32]:
        """Compute knee angle series.

        Args:
            poses: NormalizedPose sequence (H3.6M 17kp format).
            side: "left" or "right".

        Returns:
            Knee angles in degrees.
        """
        from ..types import H36Key
        from ..utils.geometry import angle_3pt

        if side == "left":
            hip_idx = H36Key.LHIP
            knee_idx = H36Key.LKNEE
            ankle_idx = H36Key.LFOOT
        else:
            hip_idx = H36Key.RHIP
            knee_idx = H36Key.RKNEE
            ankle_idx = H36Key.RFOOT

        angles = []
        for i in range(len(poses)):
            hip = poses[i, hip_idx]
            knee = poses[i, knee_idx]
            ankle = poses[i, ankle_idx]

            # Skip if any point is at origin (missing data)
            if np.allclose(hip, 0) or np.allclose(knee, 0) or np.allclose(ankle, 0):
                angles.append(0.0)
            else:
                angle = angle_3pt(hip, knee, ankle)
                angles.append(angle)

        return np.array(angles, dtype=np.float32)
