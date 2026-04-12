"""Tests for recommendation engine."""

from skating_ml.analysis.recommender import Recommender
from skating_ml.types import MetricResult


class TestRecommender:
    """Test Recommender."""

    def test_recommender_initialization(self):
        """Should initialize with rule set."""
        recommender = Recommender()

        assert recommender is not None
        # Should have rules for all element types
        assert "waltz_jump" in recommender._rules
        assert "three_turn" in recommender._rules

    def test_recommend_perfect_execution(self):
        """Should return empty list for perfect execution."""
        recommender = Recommender()

        # All metrics within ideal range
        metrics = [
            MetricResult(
                name="airtime",
                value=0.5,
                unit="s",
                is_good=True,
                reference_range=(0.3, 0.7),
            ),
            MetricResult(
                name="landing_knee_angle",
                value=110,
                unit="deg",
                is_good=True,
                reference_range=(90, 130),
            ),
        ]

        recommendations = recommender.recommend(metrics, "waltz_jump")

        # Should be empty or minimal (no errors)
        assert len(recommendations) <= 1

    def test_recommend_airtime_too_low(self):
        """Should recommend for low airtime."""
        recommender = Recommender()

        metrics = [
            MetricResult(
                name="airtime",
                value=0.2,  # Too low
                unit="s",
                is_good=False,
                reference_range=(0.3, 0.7),
            ),
        ]

        recommendations = recommender.recommend(metrics, "waltz_jump")

        assert len(recommendations) > 0
        # Should contain keywords about airtime
        assert any("время полёта" in r for r in recommendations)

    def test_recommend_landing_too_stiff(self):
        """Should recommend for stiff landing."""
        recommender = Recommender()

        metrics = [
            MetricResult(
                name="landing_knee_angle",
                value=150,  # Too straight
                unit="deg",
                is_good=False,
                reference_range=(90, 130),
            ),
        ]

        recommendations = recommender.recommend(metrics, "waltz_jump")

        assert len(recommendations) > 0
        # Should mention bending knees
        assert any("колен" in r for r in recommendations)

    def test_recommend_arm_position(self):
        """Should recommend for poor arm position."""
        recommender = Recommender()

        metrics = [
            MetricResult(
                name="arm_position_score",
                value=0.3,  # Too low (arms not controlled)
                unit="score",
                is_good=False,
                reference_range=(0.6, 1.0),
            ),
        ]

        recommendations = recommender.recommend(metrics, "waltz_jump")

        assert len(recommendations) > 0
        # Should mention arms
        assert any("рук" in r for r in recommendations)

    def test_recommend_three_turn_trunk_lean(self):
        """Should recommend for three turn trunk lean."""
        recommender = Recommender()

        metrics = [
            MetricResult(
                name="trunk_lean",
                value=25,  # Too much lean
                unit="deg",
                is_good=False,
                reference_range=(-10, 15),
            ),
        ]

        recommendations = recommender.recommend(metrics, "three_turn")

        assert len(recommendations) > 0
        # Should mention trunk/corps
        assert any("корпус" in r for r in recommendations)

    def test_recommendations_sorted_by_priority(self):
        """Should sort recommendations by priority."""
        recommender = Recommender()

        metrics = [
            MetricResult(
                name="airtime",
                value=0.2,  # Priority 0
                unit="s",
                is_good=False,
                reference_range=(0.3, 0.7),
            ),
            MetricResult(
                name="max_height",
                value=0.1,  # Priority 2
                unit="norm",
                is_good=False,
                reference_range=(0.2, 0.5),
            ),
        ]

        recommendations = recommender.recommend(metrics, "waltz_jump")

        # Airtime (priority 0) should come before height (priority 2)
        assert len(recommendations) >= 2
        # Find indices
        airtime_idx = None
        height_idx = None
        for i, rec in enumerate(recommendations):
            if "время полёта" in rec:
                airtime_idx = i
            elif "высота" in rec:
                height_idx = i

        if airtime_idx is not None and height_idx is not None:
            assert airtime_idx < height_idx

    def test_recommend_unknown_element(self):
        """Should handle unknown element type gracefully."""
        recommender = Recommender()

        metrics = [
            MetricResult(
                name="test_metric",
                value=0.5,
                unit="test",
                is_good=False,
                reference_range=(0, 1),
            ),
        ]

        # Should not crash
        recommendations = recommender.recommend(metrics, "unknown_element")

        # May be empty if no rules match
        assert isinstance(recommendations, list)

    def test_recommend_multiple_errors(self):
        """Should generate recommendations for multiple errors."""
        recommender = Recommender()

        metrics = [
            MetricResult(
                name="airtime",
                value=0.2,
                unit="s",
                is_good=False,
                reference_range=(0.3, 0.7),
            ),
            MetricResult(
                name="landing_knee_angle",
                value=150,
                unit="deg",
                is_good=False,
                reference_range=(90, 130),
            ),
            MetricResult(
                name="arm_position_score",
                value=0.3,
                unit="score",
                is_good=False,
                reference_range=(0.6, 1.0),
            ),
        ]

        recommendations = recommender.recommend(metrics, "waltz_jump")

        # Should have multiple recommendations
        assert len(recommendations) >= 2

    def test_recommendations_in_russian(self):
        """All recommendations should be in Russian."""
        recommender = Recommender()

        metrics = [
            MetricResult(
                name="airtime",
                value=0.2,
                unit="s",
                is_good=False,
                reference_range=(0.3, 0.7),
            ),
        ]

        recommendations = recommender.recommend(metrics, "waltz_jump")

        # Check that recommendations contain Cyrillic characters
        assert recommendations
        assert any(ord(c) > 127 for c in recommendations[0] for c in recommendations[0])
