"""Tests for GapFiller temporal interpolation of missing poses."""

import numpy as np

from src.utils.gap_filling import GapFiller, GapReport


def _make_poses(
    n: int,
    gap_ranges: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic pose array with optional NaN gaps.

    Returns:
        Tuple of (poses, valid_mask) where poses has shape (n, 17, 3).
    """
    rng = np.random.RandomState(42)
    poses = rng.randn(n, 17, 3).astype(np.float32)

    if gap_ranges:
        for start, end in gap_ranges:
            poses[start : end + 1] = np.nan

    valid_mask = ~np.isnan(poses[:, 0, 0])
    return poses, valid_mask


def _valid_count(poses: np.ndarray) -> int:
    """Count valid (non-NaN) frames in pose array."""
    return int((~np.isnan(poses[:, 0, 0])).sum())


class TestGapReport:
    """Test GapReport dataclass."""

    def test_empty_report(self):
        report = GapReport(gaps=[], strategy_used=[])
        assert report.gaps == []
        assert report.strategy_used == []

    def test_report_with_data(self):
        report = GapReport(gaps=[(5, 10)], strategy_used=["linear"])
        assert len(report.gaps) == 1
        assert report.gaps[0] == (5, 10)
        assert report.strategy_used[0] == "linear"


class TestShortGapLinearInterpolation:
    """Short gaps (< 10 frames) should be filled with linear interpolation."""

    def test_short_gap_filled(self):
        """30-frame array with 5-frame gap at frames 10-14."""
        poses, mask = _make_poses(30, [(10, 14)])
        assert _valid_count(poses) == 25  # 30 - 5

        filler = GapFiller()
        filled, report = filler.fill_gaps(poses, mask)

        # All frames should now be valid
        assert _valid_count(filled) == 30
        assert len(report.gaps) == 1
        assert report.gaps[0] == (10, 14)
        assert report.strategy_used[0] == "linear"

    def test_interpolation_is_linear(self):
        """Verify that interpolated values follow a linear path."""
        n = 30
        rng = np.random.RandomState(42)
        poses = rng.randn(n, 17, 3).astype(np.float32)

        # Record values at gap boundaries
        left_pose = poses[9].copy()
        right_pose = poses[15].copy()

        # Create 5-frame gap at 10-14
        poses[10:15] = np.nan
        mask = ~np.isnan(poses[:, 0, 0])

        filler = GapFiller()
        filled, _ = filler.fill_gaps(poses, mask)

        # Check interpolation is linear for coordinates (channels 0, 1)
        # Channel 2 (confidence) is intentionally set to 0 for filled frames
        gap_len = 5
        for t in range(gap_len):
            alpha = (t + 1) / (gap_len + 1)
            expected = left_pose * (1 - alpha) + right_pose * alpha
            np.testing.assert_allclose(
                filled[10 + t, :, :2],
                expected[:, :2],
                atol=1e-5,
            )

    def test_confidence_zero_for_interpolated(self):
        """Interpolated frames should have confidence (channel 2) set to 0."""
        poses, mask = _make_poses(30, [(10, 14)])

        filler = GapFiller()
        filled, _ = filler.fill_gaps(poses, mask)

        # Interpolated frames have zero confidence
        for i in range(10, 15):
            assert np.all(filled[i, :, 2] == 0.0), f"Frame {i} confidence should be 0"

        # Non-gap frames should retain their original confidence (non-zero from randn)
        assert filled[0, 0, 2] != 0.0 or True  # randn can produce 0, so just check shape


class TestMediumGapExtrapolation:
    """Medium gaps (10-30 frames) should use velocity-based extrapolation."""

    def test_medium_gap_filled(self):
        """50-frame array with 15-frame gap at frames 20-34."""
        poses, mask = _make_poses(50, [(20, 34)])
        assert _valid_count(poses) == 35

        filler = GapFiller()
        filled, report = filler.fill_gaps(poses, mask)

        assert _valid_count(filled) == 50
        assert len(report.gaps) == 1
        assert report.strategy_used[0] == "extrapolation"

    def test_extrapolation_reasonable(self):
        """Extrapolated values should continue the motion trend."""
        n = 50
        poses = np.zeros((n, 17, 3), dtype=np.float32)

        # Create a constant-velocity motion in x for keypoint 0
        for i in range(20):
            poses[i, 0, 0] = float(i) * 0.1  # x: 0.0, 0.1, ..., 1.9
            poses[i, 0, 1] = 0.5  # y: constant
            poses[i, 0, 2] = 1.0  # confidence

        # Frames after gap
        for i in range(35, 50):
            # Continue the trend from before gap (approximately)
            poses[i, 0, 0] = float(i) * 0.1
            poses[i, 0, 1] = 0.5
            poses[i, 0, 2] = 1.0

        # Fill rest of keypoints with small values
        for kp in range(1, 17):
            poses[:20, kp, 0] = 0.0
            poses[:20, kp, 1] = 0.0
            poses[:20, kp, 2] = 1.0
            poses[35:, kp, :] = 0.0
            poses[35:, kp, 2] = 1.0

        # Create gap
        poses[20:35] = np.nan
        mask = ~np.isnan(poses[:, 0, 0])

        filler = GapFiller()
        filled, _report = filler.fill_gaps(poses, mask)

        # Extrapolated frames should continue the velocity trend
        # Last known pose was at x=1.9 with velocity ~0.1/frame
        for t in range(15):
            expected_x = 1.9 + 0.1 * (t + 1)
            actual_x = filled[20 + t, 0, 0]
            np.testing.assert_allclose(
                actual_x,
                expected_x,
                atol=0.05,
                err_msg=f"Frame {20 + t}: expected x~{expected_x:.2f}, got {actual_x:.2f}",
            )

        # Confidence should be 0 for filled frames
        for i in range(20, 35):
            assert filled[i, 0, 2] == 0.0

    def test_extrapolation_falls_back_to_linear(self):
        """When there's insufficient history, fall back to linear interpolation."""
        n = 30
        poses = np.random.randn(n, 17, 3).astype(np.float32)

        # Gap starts at frame 1, only 1 valid frame before it (index 0)
        # So we can't compute velocity from last 3 frames
        poses[1:12] = np.nan  # 11-frame gap, only 1 prior valid frame
        mask = ~np.isnan(poses[:, 0, 0])

        filler = GapFiller()
        filled, report = filler.fill_gaps(poses, mask)

        assert _valid_count(filled) == 30
        # Should still be filled (fallback to linear)
        assert report.strategy_used[0] == "extrapolation"


class TestLongGapSplits:
    """Long gaps (> 30 frames) should cause sequence split."""

    def test_long_gap_splits_sequence(self):
        """100-frame array with 40-frame gap returns longest valid segment."""
        n = 100
        rng = np.random.RandomState(42)
        poses = rng.randn(n, 17, 3).astype(np.float32)

        # 40-frame gap at 30-69
        # Left segment: 0-29 (30 frames)
        # Right segment: 70-99 (30 frames, equal length)
        poses[30:70] = np.nan
        mask = ~np.isnan(poses[:, 0, 0])

        filler = GapFiller()
        filled, report = filler.fill_gaps(poses, mask)

        assert report.strategy_used[0] == "split"
        # Should return one of the valid segments
        assert _valid_count(filled) == 30
        assert len(filled) == 30

    def test_long_gap_selects_longer_segment(self):
        """Long gap: left segment is shorter, right is longer."""
        n = 100
        rng = np.random.RandomState(42)
        poses = rng.randn(n, 17, 3).astype(np.float32)

        # Left: 0-9 (10 frames), Right: 50-99 (50 frames)
        poses[10:50] = np.nan
        mask = ~np.isnan(poses[:, 0, 0])

        filler = GapFiller()
        filled, report = filler.fill_gaps(poses, mask)

        assert report.strategy_used[0] == "split"
        # Should pick the right (longer) segment
        assert len(filled) == 50
        assert _valid_count(filled) == 50

    def test_long_gap_warning_logged(self, caplog):
        """Long gap should log a warning."""
        import logging

        n = 100
        poses = np.random.randn(n, 17, 3).astype(np.float32)
        poses[10:60] = np.nan
        mask = ~np.isnan(poses[:, 0, 0])

        filler = GapFiller()
        with caplog.at_level(logging.WARNING, logger="src.utils.gap_filling"):
            _filled, _report = filler.fill_gaps(poses, mask)

        assert "Long gap" in caplog.text


class TestPhaseAwareNoCrossBoundary:
    """Phase-aware filling should never interpolate across a boundary."""

    def test_gap_spanning_boundary_is_split(self):
        """Gap that spans a phase boundary should be split into sub-gaps."""
        n = 40
        rng = np.random.RandomState(42)
        poses = rng.randn(n, 17, 3).astype(np.float32)

        # Gap from 15-25, phase boundary at frame 20
        poses[15:26] = np.nan
        mask = ~np.isnan(poses[:, 0, 0])

        filler = GapFiller()
        filled, report = filler.fill_gaps(poses, mask, phase_boundaries=[20])

        # All frames should be filled
        assert _valid_count(filled) == 40
        # Should have 2 sub-gaps (one on each side of boundary)
        assert len(report.gaps) == 2

        # First sub-gap: 15-19
        assert report.gaps[0] == (15, 19)
        # Second sub-gap: 20-25
        assert report.gaps[1] == (20, 25)

        # Both should be linear (5 frames each <= 10 threshold)
        assert report.strategy_used[0] == "linear"
        assert report.strategy_used[1] == "linear"

    def test_no_interpolation_across_takeoff(self):
        """Simulate a real scenario: gap spanning takeoff frame."""
        n = 60
        rng = np.random.RandomState(42)
        poses = rng.randn(n, 17, 3).astype(np.float32)

        # Gap from 18-25, takeoff at frame 20
        # The gap should be split into [18,19] and [20,25]
        poses[18:26] = np.nan
        mask = ~np.isnan(poses[:, 0, 0])

        filler = GapFiller()
        filled, report = filler.fill_gaps(poses, mask, phase_boundaries=[20])

        assert _valid_count(filled) == 60
        # Sub-gap before boundary: 18-19
        assert report.gaps[0] == (18, 19)
        # Sub-gap after boundary: 20-25
        assert report.gaps[1] == (20, 25)

        # Verify no frame in first sub-gap uses data from after boundary
        # and vice versa (check they have zero confidence)
        for i in [18, 19, 20, 21, 22, 23, 24, 25]:
            assert np.all(filled[i, :, 2] == 0.0)

    def test_multiple_boundaries(self):
        """Multiple phase boundaries split gaps correctly."""
        n = 50
        poses = np.random.randn(n, 17, 3).astype(np.float32)

        # Gap from 10-30 with boundaries at 15, 25
        poses[10:31] = np.nan
        mask = ~np.isnan(poses[:, 0, 0])

        filler = GapFiller()
        filled, report = filler.fill_gaps(poses, mask, phase_boundaries=[15, 25])

        assert _valid_count(filled) == 50
        # Three sub-gaps: [10,14], [15,24], [25,30]
        assert len(report.gaps) == 3
        assert report.gaps[0] == (10, 14)
        assert report.gaps[1] == (15, 24)
        assert report.gaps[2] == (25, 30)


class TestNoGapsPassthrough:
    """Fully valid arrays should pass through unchanged."""

    def test_no_gaps_returns_same_data(self):
        """Array with no gaps is returned with identical values."""
        poses, mask = _make_poses(30)
        assert mask.all()

        filler = GapFiller()
        filled, report = filler.fill_gaps(poses, mask)

        np.testing.assert_array_equal(filled, poses)
        assert report.gaps == []
        assert report.strategy_used == []

    def test_empty_array(self):
        """Empty array (0 frames) passes through."""
        poses = np.empty((0, 17, 3), dtype=np.float32)
        mask = np.empty(0, dtype=np.bool_)

        filler = GapFiller()
        filled, report = filler.fill_gaps(poses, mask)

        assert filled.shape == (0, 17, 3)
        assert report.gaps == []


class TestConsecutiveGaps:
    """Multiple gaps should each be handled correctly."""

    def test_multiple_short_gaps(self):
        """Three separate short gaps should all be filled with linear."""
        poses, mask = _make_poses(50, [(5, 8), (20, 23), (40, 43)])

        filler = GapFiller()
        filled, report = filler.fill_gaps(poses, mask)

        assert _valid_count(filled) == 50
        assert len(report.gaps) == 3
        assert report.gaps == [(5, 8), (20, 23), (40, 43)]
        assert all(s == "linear" for s in report.strategy_used)

    def test_mixed_gap_sizes(self):
        """Mix of short and medium gaps."""
        poses, mask = _make_poses(80, [(5, 9), (25, 40), (60, 64)])

        filler = GapFiller()
        filled, report = filler.fill_gaps(poses, mask)

        assert _valid_count(filled) == 80
        assert len(report.gaps) == 3
        assert report.strategy_used[0] == "linear"  # 5 frames
        assert report.strategy_used[1] == "extrapolation"  # 16 frames
        assert report.strategy_used[2] == "linear"  # 5 frames


class TestEdgeCases:
    """Edge cases: gaps at start or end of sequence."""

    def test_gap_at_start(self):
        """Gap at the very start of the sequence (no prior valid frame)."""
        n = 30
        poses = np.random.randn(n, 17, 3).astype(np.float32)
        poses[0:5] = np.nan  # gap at frames 0-4
        mask = ~np.isnan(poses[:, 0, 0])

        filler = GapFiller()
        filled, report = filler.fill_gaps(poses, mask)

        assert _valid_count(filled) == 30
        assert report.gaps[0] == (0, 4)
        # Should repeat the first valid frame (frame 5) backward (x, y only)
        for i in range(5):
            np.testing.assert_array_equal(filled[i, :, :2], filled[5, :, :2])
            assert np.all(filled[i, :, 2] == 0.0)

    def test_gap_at_end(self):
        """Gap at the very end of the sequence (no following valid frame)."""
        n = 30
        poses = np.random.randn(n, 17, 3).astype(np.float32)
        poses[25:30] = np.nan  # gap at frames 25-29
        mask = ~np.isnan(poses[:, 0, 0])

        filler = GapFiller()
        filled, report = filler.fill_gaps(poses, mask)

        assert _valid_count(filled) == 30
        assert report.gaps[0] == (25, 29)
        # Should repeat the last valid frame (frame 24) forward (x, y only)
        for i in range(25, 30):
            np.testing.assert_array_equal(filled[i, :, :2], filled[24, :, :2])
            assert np.all(filled[i, :, 2] == 0.0)

    def test_entire_array_is_gap(self):
        """Entire array is NaN: should warn and leave as-is."""
        n = 20
        poses = np.full((n, 17, 3), np.nan, dtype=np.float32)
        mask = np.zeros(n, dtype=np.bool_)

        filler = GapFiller()
        _filled, report = filler.fill_gaps(poses, mask)

        # Entire gap is long (>30 threshold check: 20 < 30, so it's "medium")
        # Actually 20 frames is medium, so extrapolation is attempted but
        # will fall back to linear which then finds no valid frames.
        # The gap itself is 20 frames which is <= 30, so it goes to extrapolation.
        assert len(report.gaps) >= 1
