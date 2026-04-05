"""Motion DTW with keyframe-aware alignment.

This module improves standard DTW by using keyframes (takeoff, peak, landing)
as anchors to prevent pathological warping and ensure biomechanically correct alignment.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from dtw import dtw  # type: ignore[import-untyped]

from ..types import ElementPhase, NormalizedPose

if TYPE_CHECKING:
    from dtw import DTW


@dataclass
class KeyFrame:
    """A keyframe that must be aligned between sequences.

    Attributes:
        name: Keyframe identifier (takeoff, peak, landing).
        user_idx: Frame index in user sequence.
        ref_idx: Frame index in reference sequence.
    """

    name: str
    user_idx: int
    ref_idx: int


@dataclass
class PhaseAlignment:
    """Alignment result for a single phase.

    Attributes:
        name: Phase name.
        distance: DTW distance for this phase.
        warp_path: Alignment path (N, 2) mapping user to reference frames.
    """

    name: str
    distance: float
    warp_path: np.ndarray


@dataclass
class MotionDTWResult:
    """Complete motion alignment result.

    Attributes:
        total_distance: Combined distance across all phases.
        phase_alignments: Alignment results for each phase.
        full_warp_path: Complete warp path for entire sequence.
        aligned_user: User poses warped to reference timeline.
    """

    total_distance: float
    phase_alignments: list[PhaseAlignment]
    full_warp_path: np.ndarray
    aligned_user: NormalizedPose


class MotionDTWAligner:
    """Keyframe-aware DTW aligner for motion sequences.

    Uses biomechanically important keyframes (takeoff, peak, landing) as anchors
    to prevent pathological warping and ensure phase-aware alignment.

    Example:
        ```python
        aligner = MotionDTWAligner()

        # Align with keyframe constraints
        result = aligner.align_with_keyframes(
            user_poses,
            user_phases,
            ref_poses,
            ref_phases,
        )

        # Access phase-specific distances
        for phase_align in result.phase_alignments:
            print(f"{phase_align.name}: {phase_align.distance:.4f}")
        ```
    """

    def __init__(
        self,
        window_type: str = "sakoechiba",
        window_size: float = 0.2,
        phase_weight: float = 1.0,
    ) -> None:
        """Initialize MotionDTW aligner.

        Args:
            window_type: DTW window type ("sakoechiba", "itakura", or None).
            window_size: Window size ratio (0.2 = 20% of phase length).
            phase_weight: Weight for phase distance in total score.
        """
        self._window_type = window_type
        self._window_size = window_size
        self._phase_weight = phase_weight

    def align_with_keyframes(
        self,
        user: NormalizedPose,
        user_phases: ElementPhase,
        reference: NormalizedPose,
        ref_phases: ElementPhase,
        joints: list[int] | None = None,
    ) -> MotionDTWResult:
        """Align sequences using keyframe-constrained DTW.

        Splits sequences into phases at keyframes and aligns each phase separately.
        This prevents pathological warping across biomechanical boundaries.

        Args:
            user: User pose sequence (num_user_frames, 33, 2).
            user_phases: User phase boundaries with keyframes.
            reference: Reference pose sequence (num_ref_frames, 33, 2).
            ref_phases: Reference phase boundaries with keyframes.
            joints: Joint indices to use for alignment (None = all 33).

        Returns:
            MotionDTWResult with phase-wise alignments and combined result.
        """
        if joints is None:
            joints = list(range(33))

        # Determine phase segments
        phases = self._split_into_phases(user_phases, ref_phases)

        # Align each phase separately
        phase_alignments: list[PhaseAlignment] = []
        phase_paths: list[np.ndarray] = []

        for phase in phases:
            user_segment = user[phase["user_start"] : phase["user_end"]]
            ref_segment = reference[phase["ref_start"] : phase["ref_end"]]

            if len(user_segment) == 0 or len(ref_segment) == 0:
                # Empty phase, skip
                continue

            # Align this phase
            distance, warp_path = self._align_phase(
                user_segment, ref_segment, joints, phase["name"]
            )

            # Adjust warp path to global coordinates
            if len(warp_path) > 0:
                warp_path[:, 0] += phase["user_start"]
                warp_path[:, 1] += phase["ref_start"]

            phase_alignments.append(
                PhaseAlignment(name=phase["name"], distance=distance, warp_path=warp_path)
            )
            phase_paths.append(warp_path)

        # Combine phase alignments
        full_warp_path = self._combine_phase_paths(phase_paths)

        # Warp user sequence to reference timeline
        aligned_user = self._warp_with_path(user, full_warp_path, len(reference))

        # Compute total distance
        total_distance = sum(p.distance for p in phase_alignments) / max(len(phase_alignments), 1)

        return MotionDTWResult(
            total_distance=total_distance,
            phase_alignments=phase_alignments,
            full_warp_path=full_warp_path,
            aligned_user=aligned_user,
        )

    def _extract_keyframes(
        self, user_phases: ElementPhase, ref_phases: ElementPhase
    ) -> list[KeyFrame]:
        """Extract keyframes that must align between sequences.

        Args:
            user_phases: User phase boundaries.
            ref_phases: Reference phase boundaries.

        Returns:
            List of KeyFrame objects with indices in both sequences.
        """
        keyframes: list[KeyFrame] = []

        # Takeoff (for jumps)
        if user_phases.takeoff > 0 and ref_phases.takeoff > 0:
            keyframes.append(
                KeyFrame(name="takeoff", user_idx=user_phases.takeoff, ref_idx=ref_phases.takeoff)
            )

        # Peak (maximum height or turn center)
        if user_phases.peak > 0 and ref_phases.peak > 0:
            keyframes.append(
                KeyFrame(name="peak", user_idx=user_phases.peak, ref_idx=ref_phases.peak)
            )

        # Landing
        if user_phases.landing > 0 and ref_phases.landing > 0:
            keyframes.append(
                KeyFrame(name="landing", user_idx=user_phases.landing, ref_idx=ref_phases.landing)
            )

        return keyframes

    def _split_into_phases(self, user_phases: ElementPhase, ref_phases: ElementPhase) -> list[dict]:
        """Split sequences into biomechanically meaningful phases.

        For jumps: entry -> flight -> landing
        For steps: full sequence (no takeoff/landing)

        Args:
            user_phases: User phase boundaries.
            ref_phases: Reference phase boundaries.

        Returns:
            List of phase dicts with start/end indices for both sequences.
        """
        phases: list[dict] = []

        # Check if this is a jump (has takeoff/landing)
        is_jump = user_phases.takeoff > 0 and user_phases.landing > 0

        if is_jump:
            # Entry phase: start -> takeoff
            phases.append(
                {
                    "name": "entry",
                    "user_start": user_phases.start,
                    "user_end": user_phases.takeoff,
                    "ref_start": ref_phases.start,
                    "ref_end": ref_phases.takeoff,
                }
            )

            # Flight phase: takeoff -> landing
            phases.append(
                {
                    "name": "flight",
                    "user_start": user_phases.takeoff,
                    "user_end": user_phases.landing,
                    "ref_start": ref_phases.takeoff,
                    "ref_end": ref_phases.landing,
                }
            )

            # Landing phase: landing -> end
            phases.append(
                {
                    "name": "landing",
                    "user_start": user_phases.landing,
                    "user_end": user_phases.end,
                    "ref_start": ref_phases.landing,
                    "ref_end": ref_phases.end,
                }
            )
        else:
            # Single phase for steps/turns
            phases.append(
                {
                    "name": "full",
                    "user_start": user_phases.start,
                    "user_end": user_phases.end,
                    "ref_start": ref_phases.start,
                    "ref_end": ref_phases.end,
                }
            )

        return phases

    def _align_phase(
        self,
        user_segment: NormalizedPose,
        ref_segment: NormalizedPose,
        joints: list[int],
        phase_name: str,
    ) -> tuple[float, np.ndarray]:
        """Align a single phase using DTW.

        Args:
            user_segment: User poses for this phase.
            ref_segment: Reference poses for this phase.
            joints: Joint indices to use.
            phase_name: Phase identifier (reserved for future use).

        Returns:
            Tuple of (distance, warp_path).
        """
        # Flatten to 2D
        user_flat = user_segment[:, joints, :].reshape(len(user_segment), -1)
        ref_flat = ref_segment[:, joints, :].reshape(len(ref_segment), -1)

        # Compute DTW with phase-appropriate window
        # Use smaller window for short phases
        phase_window = min(self._window_size, 0.5)  # Max 50% for short phases
        alignment = self._compute_dtw(user_flat, ref_flat, phase_window)

        # Build warp path
        warp_path = np.column_stack(
            [alignment.index1, alignment.index2]  # type: ignore[attr-defined]
        )

        # Normalize distance by path length
        distance = float(
            alignment.distance / max(len(user_segment), len(ref_segment))  # type: ignore[attr-defined]
        )

        return distance, warp_path

    def _compute_dtw(self, x: np.ndarray, y: np.ndarray, window_size: float) -> "DTW":  # type: ignore[valid-type]
        """Compute DTW with specified window.

        Args:
            x: First sequence.
            y: Second sequence.
            window_size: Window size ratio.

        Returns:
            DTW alignment object.
        """
        window_args = {}
        if self._window_type == "sakoechiba":
            win_size = int(window_size * max(len(x), len(y)))
            window_args = {
                "window_type": self._window_type,
                "window_args": {"window_size": max(win_size, 1)},
            }
        elif self._window_type == "itakura":
            window_args = {"window_type": "itakura"}

        return dtw(
            x,
            y,
            keep_internals=True,
            distance_only=False,
            open_end=False,
            open_begin=False,
            **window_args,
        )

    def _combine_phase_paths(self, phase_paths: list[np.ndarray]) -> np.ndarray:
        """Combine phase warp paths into a single path.

        Args:
            phase_paths: List of warp paths for each phase.

        Returns:
            Combined warp path for full sequence.
        """
        if not phase_paths:
            return np.array([], dtype=np.int32).reshape(0, 2)

        # Filter out empty paths
        valid_paths = [p for p in phase_paths if len(p) > 0]

        if not valid_paths:
            return np.array([], dtype=np.int32).reshape(0, 2)

        return np.vstack(valid_paths)

    def _warp_with_path(
        self, sequence: NormalizedPose, warp_path: np.ndarray, target_length: int
    ) -> NormalizedPose:
        """Warp sequence using computed warp path.

        Args:
            sequence: Original sequence.
            warp_path: DTW warp path (N, 2).
            target_length: Length of output sequence.

        Returns:
            Warped sequence.
        """
        if len(warp_path) == 0:
            # No path, return sequence as-is (padded/truncated)
            if len(sequence) >= target_length:
                return sequence[:target_length]
            else:
                # Pad with last frame
                padding = np.zeros((target_length - len(sequence), 33, 2), dtype=np.float32)
                padding[:] = sequence[-1]
                return np.vstack([sequence, padding])

        # Create mapping from reference frames to user frames
        warped = np.zeros((target_length, 33, 2), dtype=np.float32)

        for ref_idx in range(target_length):
            # Find all user frames mapped to this reference frame
            user_indices = warp_path[:, 0][warp_path[:, 1] == ref_idx]

            if len(user_indices) == 0:
                # No mapping, use nearest neighbor
                if ref_idx > 0:
                    warped[ref_idx] = warped[ref_idx - 1]
                else:
                    warped[ref_idx] = sequence[0]
            elif len(user_indices) == 1:
                warped[ref_idx] = sequence[int(user_indices[0])]
            else:
                # Multiple mappings, average
                frames = sequence[user_indices.astype(int)]
                warped[ref_idx] = np.mean(frames, axis=0)

        return warped

    def compute_phase_distances(
        self,
        user: NormalizedPose,
        user_phases: ElementPhase,
        reference: NormalizedPose,
        ref_phases: ElementPhase,
        joints: list[int] | None = None,
    ) -> dict[str, float]:
        """Compute DTW distance per phase separately.

        This is useful for identifying which phases need improvement.

        Args:
            user: User pose sequence.
            user_phases: User phase boundaries.
            reference: Reference pose sequence.
            ref_phases: Reference phase boundaries.
            joints: Joint indices to use.

        Returns:
            Dict mapping phase names to normalized distances.
        """
        result = self.align_with_keyframes(user, user_phases, reference, ref_phases, joints)

        return {pa.name: pa.distance for pa in result.phase_alignments}

    def compute_distance(
        self,
        user: NormalizedPose,
        reference: NormalizedPose,
        joints: list[int] | None = None,
    ) -> float:
        """Compute DTW distance between sequences.

        Computes the DTW distance between two pose sequences.
        For phase-aware analysis, use align_with_keyframes instead.

        Args:
            user: User pose sequence (num_frames, 17, 2) - H3.6M format.
            reference: Reference pose sequence (num_frames, 17, 2) - H3.6M format.
            joints: Joint indices to use (None = all 17).

        Returns:
            Normalized DTW distance.
        """
        if joints is None:
            joints = list(range(17))

        # Create default phases (full sequence)
        user_phases = ElementPhase(
            name="full", start=0, takeoff=0, peak=0, landing=0, end=len(user) - 1
        )
        ref_phases = ElementPhase(
            name="full", start=0, takeoff=0, peak=0, landing=0, end=len(reference) - 1
        )

        # Use phase-aware alignment
        result = self.align_with_keyframes(user, user_phases, reference, ref_phases, joints)

        return result.total_distance
