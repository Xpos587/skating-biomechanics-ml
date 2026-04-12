"""Motion alignment using Dynamic Time Warping (DTW).

This module aligns pose sequences of different lengths and tempos
for fair comparison between user and reference performances.
"""

from typing import TYPE_CHECKING

import numpy as np
from dtw import dtw  # type: ignore[import-untyped]

from ..types import ElementPhase, NormalizedPose

if TYPE_CHECKING:
    from dtw import DTW


class MotionAligner:
    """Align motion sequences using DTW with Sakoe-Chiba window."""

    def __init__(self, window_type: str = "sakoechiba", window_size: float = 0.2) -> None:
        """Initialize motion aligner.

        Args:
            window_type: Type of DTW constraint ("sakoechiba", "itakura", or None).
            window_size: Window size ratio (0.2 = 20% of max sequence length).
        """
        self._window_type = window_type
        self._window_size = window_size

    def align(
        self,
        user: NormalizedPose,
        reference: NormalizedPose,
        joints: list[int] | None = None,
    ) -> tuple[NormalizedPose, np.ndarray]:
        """Align user pose sequence to reference using DTW.

        Args:
            user: User pose sequence (num_user_frames, 33, 2).
            reference: Reference pose sequence (num_ref_frames, 33, 2).
            joints: List of joint indices to use for alignment (None = all 33).

        Returns:
            Tuple of (aligned_user, warp_path):
            - aligned_user: User poses warped to reference timeline (num_ref_frames, 33, 2).
            - warp_path: DTW path (N, 2) mapping user frames to reference frames.
        """
        # Select joints for alignment
        if joints is None:
            joints = list(range(33))

        # Flatten to 2D sequences: (num_frames, num_joints * 2)
        user_flat = user[:, joints, :].reshape(len(user), -1)
        ref_flat = reference[:, joints, :].reshape(len(reference), -1)

        # Compute DTW
        alignment = self._compute_dtw(user_flat, ref_flat)

        # Warp user sequence to reference timeline
        aligned_user = self._warp_sequence(
            user,
            alignment.index1,  # type: ignore[attr-defined]
            alignment.index2,  # type: ignore[attr-defined]
        )

        # Build warp path as array
        warp_path = np.column_stack(
            [alignment.index1, alignment.index2]  # type: ignore[attr-defined]
        )

        return aligned_user, warp_path

    def compute_distance(
        self,
        user: NormalizedPose,
        reference: NormalizedPose,
        joints: list[int] | None = None,
    ) -> float:
        """Compute DTW distance between sequences.

        Args:
            user: User pose sequence (num_user_frames, 33, 2).
            reference: Reference pose sequence (num_ref_frames, 33, 2).
            joints: List of joint indices to use for alignment (None = all 33).

        Returns:
            DTW distance (normalized).
        """
        if joints is None:
            joints = list(range(33))

        # Flatten to 2D sequences
        user_flat = user[:, joints, :].reshape(len(user), -1)
        ref_flat = reference[:, joints, :].reshape(len(reference), -1)

        # Compute DTW
        alignment = self._compute_dtw(user_flat, ref_flat)

        # Normalize distance by path length
        return float(
            alignment.distance / max(len(user), len(reference))  # type: ignore[attr-defined]
        )

    def align_phases(
        self,
        user: NormalizedPose,
        user_phases: ElementPhase,
        reference: NormalizedPose,
        ref_phases: ElementPhase,
        joints: list[int] | None = None,
    ) -> dict[str, float]:
        """Compute DTW distance per phase separately.

        Args:
            user: User pose sequence.
            user_phases: User phase boundaries.
            reference: Reference pose sequence.
            ref_phases: Reference phase boundaries.
            joints: List of joint indices to use for alignment.

        Returns:
            Dict mapping phase names to DTW distances.
        """
        # Extract phases
        user_entry = user[user_phases.start : user_phases.takeoff]
        user_flight = user[user_phases.takeoff : user_phases.landing]
        user_landing = user[user_phases.landing : user_phases.end]

        ref_entry = reference[ref_phases.start : ref_phases.takeoff]
        ref_flight = reference[ref_phases.takeoff : ref_phases.landing]
        ref_landing = reference[ref_phases.landing : ref_phases.end]

        distances: dict[str, float] = {}

        # Only compute distances for non-empty phases
        if len(user_entry) > 0 and len(ref_entry) > 0:
            distances["entry"] = self.compute_distance(user_entry, ref_entry, joints)

        if len(user_flight) > 0 and len(ref_flight) > 0:
            distances["flight"] = self.compute_distance(user_flight, ref_flight, joints)

        if len(user_landing) > 0 and len(ref_landing) > 0:
            distances["landing"] = self.compute_distance(user_landing, ref_landing, joints)

        return distances

    def extract_phase(
        self,
        poses: NormalizedPose,
        phase: ElementPhase,
    ) -> NormalizedPose:
        """Extract poses for a specific phase.

        Args:
            poses: Full pose sequence.
            phase: Phase boundaries.

        Returns:
            Pose sequence for the phase.
        """
        start = phase.start
        end = min(phase.end, len(poses))

        return poses[start:end]

    def _compute_dtw(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> "DTW":  # type: ignore[valid-type]
        """Compute DTW alignment with configured window.

        Args:
            x: First sequence (num_samples, num_features).
            y: Second sequence (num_samples, num_features).

        Returns:
            DTW alignment object.
        """
        # Configure window
        window_args = {}
        if self._window_type == "sakoechiba":
            window_size = int(self._window_size * max(len(x), len(y)))
            window_args = {
                "window_type": self._window_type,
                "window_args": {"window_size": window_size},
            }
        elif self._window_type == "itakura":
            window_args = {"window_type": "itakura"}

        # Compute DTW
        return dtw(
            x,
            y,
            keep_internals=True,
            distance_only=False,
            open_end=False,
            open_begin=False,
            **window_args,
        )

    def _warp_sequence(
        self,
        sequence: NormalizedPose,
        index1: np.ndarray,
        index2: np.ndarray,
    ) -> NormalizedPose:
        """Warp sequence to match reference timeline.

        Args:
            sequence: Original sequence (num_frames, 33, 2).
            index1: DTW path indices for sequence.
            index2: DTW path indices for reference.

        Returns:
            Warped sequence (len(index2), 33, 2).
        """
        # Create mapping from reference frames to user frames
        # For each reference frame, find the corresponding user frame(s)
        warped_length = len(np.unique(index2))
        warped = np.zeros((warped_length, 33, 2), dtype=np.float32)

        for i, ref_idx in enumerate(np.unique(index2)):
            # Find all user frames mapped to this reference frame
            user_indices = index1[index2 == ref_idx]

            if len(user_indices) == 0:
                # No mapping, use nearest
                warped[i] = sequence[int(ref_idx * len(sequence) / warped_length)]
            elif len(user_indices) == 1:
                warped[i] = sequence[int(user_indices[0])]
            else:
                # Multiple user frames, average them
                frames = sequence[user_indices.astype(int)]
                warped[i] = np.mean(frames, axis=0)

        return warped
