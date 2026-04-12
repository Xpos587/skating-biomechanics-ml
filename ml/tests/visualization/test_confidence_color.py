"""Tests for RdYlGn confidence coloring."""

from skating_ml.visualization.skeleton.joints import get_confidence_color_rdygn


class TestRdYlGnConfidenceColor:
    def test_high_confidence_is_green(self):
        """Confidence=1.0 should return green."""
        color = get_confidence_color_rdygn(1.0)
        # Green in BGR: (0, ~255, 0)
        assert color[1] > 200  # G channel high

    def test_low_confidence_is_red(self):
        """Confidence=0.0 should return red."""
        color = get_confidence_color_rdygn(0.0)
        # Red in BGR: (0, 0, ~255)
        assert color[2] > 200  # R channel high

    def test_mid_confidence_is_yellowish(self):
        """Confidence=0.5 should return yellow-ish."""
        color = get_confidence_color_rdygn(0.5)
        # Yellow in BGR: (0, ~255, ~255)
        assert color[1] > 100  # G present
        assert color[2] > 100  # R present

    def test_clamps_out_of_range(self):
        """Should clamp confidence to [0, 1]."""
        c_low = get_confidence_color_rdygn(-0.5)
        c_high = get_confidence_color_rdygn(1.5)
        assert c_low == get_confidence_color_rdygn(0.0)
        assert c_high == get_confidence_color_rdygn(1.0)
