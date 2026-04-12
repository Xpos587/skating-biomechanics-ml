"""Gap filling for pose arrays with missing (NaN) frames.

Provides three-tier interpolation strategy:
- Short gaps (<10 frames): Linear interpolation
- Medium gaps (10-30 frames): Velocity-based extrapolation
- Long gaps (>30 frames): Sequence split at gap boundaries

Phase-aware mode prevents interpolation across phase boundaries
(takeoff, peak, landing), treating each inter-boundary segment
independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class GapReport:
    """Report of gap filling operations.

    Attributes:
        gaps: List of (start, end) frame indices for each gap filled.
        strategy_used: Strategy applied to each gap ('linear', 'extrapolation', 'split').
    """

    gaps: list[tuple[int, int]]
    strategy_used: list[str]


class GapFiller:
    """Fills NaN gaps in pose arrays using a three-tier strategy.

    Pose arrays have shape (N, 17, 3) where NaN rows indicate frames
    where the target person was not detected. This class fills those
    gaps so downstream components receive continuous pose data.

    Args:
        fps: Video frame rate (for time-based threshold computation).
        short_gap_threshold: Max frames for linear interpolation (default 10).
        medium_gap_threshold: Max frames for extrapolation (default 30).
    """

    def __init__(
        self,
        fps: float = 30.0,
        short_gap_threshold: int = 10,
        medium_gap_threshold: int = 30,
    ) -> None:
        self._fps = fps
        self._short_threshold = short_gap_threshold
        self._medium_threshold = medium_gap_threshold

    def fill_gaps(
        self,
        poses: NDArray[np.floating],
        valid_mask: NDArray[np.bool_],
        phase_boundaries: list[int] | None = None,
    ) -> tuple[NDArray[np.floating], GapReport]:
        """Fill NaN gaps in the pose array.

        Args:
            poses: Pose array of shape (N, 17, 3). NaN rows are gaps.
            valid_mask: Boolean array of shape (N,). True where pose is valid.
            phase_boundaries: Optional list of frame indices (e.g. [takeoff, peak,
                landing]). When provided, interpolation never crosses a boundary.

        Returns:
            Tuple of (filled_poses, gap_report). filled_poses has the same shape
            as input. For long gaps that trigger a split, the returned array
            contains only the longest valid segment.
        """
        filled = poses.copy()
        gaps = self._find_gaps(valid_mask)

        if not gaps:
            return filled, GapReport(gaps=[], strategy_used=[])

        if phase_boundaries is not None:
            return self._fill_phase_aware(filled, valid_mask, gaps, phase_boundaries)

        return self._fill_no_phases(filled, gaps)

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_gaps(valid_mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
        """Return list of (start, end) indices for each gap.

        A gap is a contiguous run of False in valid_mask.
        Both start and end are inclusive indices within the gap.
        """
        gaps: list[tuple[int, int]] = []
        n = len(valid_mask)
        i = 0
        while i < n:
            if not valid_mask[i]:
                start = i
                while i < n and not valid_mask[i]:
                    i += 1
                gaps.append((start, i - 1))
            else:
                i += 1
        return gaps

    # ------------------------------------------------------------------
    # Phase-aware filling
    # ------------------------------------------------------------------

    def _fill_phase_aware(
        self,
        poses: NDArray[np.floating],
        valid_mask: NDArray[np.bool_],
        gaps: list[tuple[int, int]],
        phase_boundaries: list[int],
    ) -> tuple[NDArray[np.floating], GapReport]:
        """Fill gaps respecting phase boundaries.

        Each phase boundary splits the timeline. Gaps that span a boundary
        are split into sub-gaps that are filled independently within each
        inter-boundary segment.
        """
        all_gaps: list[tuple[int, int]] = []
        all_strategies: list[str] = []

        # Build sorted unique boundary set plus sequence edges
        boundaries = sorted(set(phase_boundaries))
        segments = self._build_phase_segments(len(poses), boundaries)

        for seg_start, seg_end in segments:
            seg_valid = valid_mask[seg_start : seg_end + 1]
            seg_gaps = self._find_gaps(seg_valid)

            # Remap gap indices back to full-array coordinates
            for gs, ge in seg_gaps:
                abs_start = seg_start + gs
                abs_end = seg_start + ge
                gap_len = abs_end - abs_start + 1

                if gap_len <= self._short_threshold:
                    self._fill_linear(poses, abs_start, abs_end)
                    all_strategies.append("linear")
                elif gap_len <= self._medium_threshold:
                    seg_mask = valid_mask[seg_start : seg_end + 1]
                    # Map to segment-local indices for velocity lookback
                    self._fill_extrapolation(poses, abs_start, abs_end, seg_start, seg_mask)
                    all_strategies.append("extrapolation")
                else:
                    # Long gap within a phase segment: still fill linearly
                    # rather than splitting the returned array, but warn.
                    logger.warning(
                        "Long gap (%d frames) within phase segment [%d, %d]. "
                        "Falling back to linear interpolation.",
                        gap_len,
                        abs_start,
                        abs_end,
                    )
                    self._fill_linear(poses, abs_start, abs_end)
                    all_strategies.append("linear")

                all_gaps.append((abs_start, abs_end))

        return poses, GapReport(gaps=all_gaps, strategy_used=all_strategies)

    @staticmethod
    def _build_phase_segments(total_frames: int, boundaries: list[int]) -> list[tuple[int, int]]:
        """Build (start, end) inclusive segment ranges from boundaries.

        Segments are the intervals between consecutive phase boundaries,
        including the regions before the first and after the last boundary.
        """
        if not boundaries:
            return [(0, total_frames - 1)]

        segments: list[tuple[int, int]] = []
        prev = 0
        for b in boundaries:
            if b > prev:
                segments.append((prev, b - 1))
            prev = b
        if prev < total_frames:
            segments.append((prev, total_frames - 1))
        return segments

    # ------------------------------------------------------------------
    # Non-phase-aware filling
    # ------------------------------------------------------------------

    def _fill_no_phases(
        self,
        poses: NDArray[np.floating],
        gaps: list[tuple[int, int]],
    ) -> tuple[NDArray[np.floating], GapReport]:
        """Fill gaps when no phase boundaries are provided."""
        all_gaps: list[tuple[int, int]] = []
        strategies: list[str] = []

        for start, end in gaps:
            gap_len = end - start + 1

            if gap_len <= self._short_threshold:
                self._fill_linear(poses, start, end)
                strategies.append("linear")
            elif gap_len <= self._medium_threshold:
                # Build valid mask for the whole array for lookback
                vm = ~np.isnan(poses[:, 0, 0])
                self._fill_extrapolation(poses, start, end, 0, vm)
                strategies.append("extrapolation")
            else:
                # Long gap: split sequence, keep longest valid segment
                poses_new, actual_start, actual_end = self._split_at_long_gap(poses, start, end)
                # After split, we must re-run on remaining gaps
                # For simplicity, handle this one long gap and adjust
                # remaining poses in-place. The caller gets the trimmed array.
                logger.warning(
                    "Long gap (%d frames) at [%d, %d]. "
                    "Returning longest valid segment [%d, %d] (%d frames).",
                    gap_len,
                    start,
                    end,
                    actual_start,
                    actual_end,
                    actual_end - actual_start + 1,
                )
                # Re-run fill on the trimmed array
                trimmed_mask = ~np.isnan(poses_new[:, 0, 0])
                remaining_gaps = self._find_gaps(trimmed_mask)
                # Fill remaining short/medium gaps in trimmed array
                for gs, ge in remaining_gaps:
                    gl = ge - gs + 1
                    if gl <= self._short_threshold:
                        self._fill_linear(poses_new, gs, ge)
                        strategies.append("linear")
                    elif gl <= self._medium_threshold:
                        vm = ~np.isnan(poses_new[:, 0, 0])
                        self._fill_extrapolation(poses_new, gs, ge, 0, vm)
                        strategies.append("extrapolation")
                    all_gaps.append((gs, ge))

                strategies.insert(0, "split")
                all_gaps.insert(0, (start, end))
                return poses_new, GapReport(gaps=all_gaps, strategy_used=strategies)

            all_gaps.append((start, end))

        return poses, GapReport(gaps=all_gaps, strategy_used=strategies)

    # ------------------------------------------------------------------
    # Filling strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_linear(poses: NDArray[np.floating], gap_start: int, gap_end: int) -> None:
        """Fill gap with linear interpolation between bounding valid frames.

        For gaps at the array edges (no valid frame on one side), copies
        the nearest valid frame across the gap.
        Sets confidence (channel 2) to 0 for all interpolated frames.
        """
        n = len(poses)

        # Find nearest valid frame before gap
        left_idx: int | None = None
        for i in range(gap_start - 1, -1, -1):
            if not np.isnan(poses[i, 0, 0]):
                left_idx = i
                break

        # Find nearest valid frame after gap
        right_idx: int | None = None
        for i in range(gap_end + 1, n):
            if not np.isnan(poses[i, 0, 0]):
                right_idx = i
                break

        num_gap_frames = gap_end - gap_start + 1

        if left_idx is not None and right_idx is not None:
            # Interpolate between left and right
            left_pose = poses[left_idx].copy()
            right_pose = poses[right_idx].copy()
            for t in range(num_gap_frames):
                alpha = (t + 1) / (num_gap_frames + 1)
                poses[gap_start + t] = left_pose * (1 - alpha) + right_pose * alpha
                poses[gap_start + t, :, 2] = 0.0  # zero confidence
        elif left_idx is not None:
            # Gap at end: repeat last valid frame
            poses[gap_start : gap_end + 1] = poses[left_idx]
            poses[gap_start : gap_end + 1, :, 2] = 0.0
        elif right_idx is not None:
            # Gap at start: repeat first valid frame
            poses[gap_start : gap_end + 1] = poses[right_idx]
            poses[gap_start : gap_end + 1, :, 2] = 0.0
        else:
            # Entire array is NaN: cannot fill
            logger.warning("No valid frames in array; cannot fill gaps.")

    @staticmethod
    def _fill_extrapolation(
        poses: NDArray[np.floating],
        gap_start: int,
        gap_end: int,
        seg_offset: int,
        seg_valid_mask: NDArray[np.bool_],
    ) -> None:
        """Fill gap using velocity-based extrapolation.

        Computes velocity from the last 3 valid frames before the gap
        using np.gradient, then extrapolates forward. Falls back to
        linear interpolation if insufficient history exists.

        Sets confidence (channel 2) to 0 for all filled frames.
        """
        gap_len = gap_end - gap_start + 1

        # Find last 3 valid frames before gap (relative to seg_offset)
        local_start = gap_start - seg_offset
        prior_indices: list[int] = []
        for i in range(local_start - 1, -1, -1):
            if seg_valid_mask[i]:
                prior_indices.append(seg_offset + i)  # absolute index
                if len(prior_indices) == 3:
                    break

        if len(prior_indices) < 2:
            # Not enough history for velocity; fall back to linear
            GapFiller._fill_linear(poses, gap_start, gap_end)
            return

        prior_indices.reverse()  # chronological order
        last_valid_idx = prior_indices[-1]

        # Compute per-keypoint per-coordinate velocity from recent frames
        recent_poses = poses[prior_indices]  # (K, 17, 3), K=2 or 3
        velocities = np.gradient(recent_poses, axis=0)  # (K, 17, 3)
        avg_velocity = velocities.mean(axis=0)  # (17, 3)

        # Extrapolate from last known pose
        last_pose = poses[last_valid_idx].copy()

        for t in range(gap_len):
            dt = t + 1
            poses[gap_start + t] = last_pose + avg_velocity * dt
            poses[gap_start + t, :, 2] = 0.0  # zero confidence

    @staticmethod
    def _split_at_long_gap(
        poses: NDArray[np.floating],
        gap_start: int,
        gap_end: int,
    ) -> tuple[NDArray[np.floating], int, int]:
        """Split array at a long gap and return the longest valid segment.

        Returns:
            Tuple of (trimmed_poses, seg_start, seg_end) where seg_start
            and seg_end are the inclusive frame indices within the trimmed array.
        """
        # Segment before gap: [0, gap_start-1]
        # Segment after gap: [gap_end+1, end]
        left_end = gap_start  # exclusive
        right_start = gap_end + 1
        n = len(poses)

        left_len = left_end
        right_len = n - right_start

        if left_len >= right_len and left_len > 0:
            return poses[:left_end].copy(), 0, left_len - 1
        elif right_len > 0:
            return poses[right_start:].copy(), 0, right_len - 1
        else:
            # Both empty (shouldn't happen in practice)
            return poses.copy(), 0, n - 1
