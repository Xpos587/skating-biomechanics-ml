"""Rule-based recommendation engine for skating technique.

This module generates specific, actionable recommendations in Russian
based on biomechanics metrics analysis.
"""

from . import jump_rules, three_turn_rules
from .types import MetricResult, RecommendationRule


class Recommender:
    """Generate recommendations based on biomechanics metrics.

    Uses a rule-based system where each metric that falls outside
    the ideal range triggers specific recommendations.
    """

    def __init__(self) -> None:
        """Initialize recommender with rule set."""
        self._rules: dict[str, list[RecommendationRule]] = {}
        self._build_rules()

    def recommend(
        self,
        metrics: list[MetricResult],
        element_type: str,
    ) -> list[str]:
        """Generate recommendations based on metrics.

        Args:
            metrics: List of computed MetricResult.
            element_type: Type of skating element.

        Returns:
            List of recommendation strings in Russian, sorted by priority.
        """
        recommendations: list[tuple[int, str]] = []

        # Get rules for element type
        element_rules = self._rules.get(element_type, [])

        # Check each metric against rules
        for metric in metrics:
            for rule in element_rules:
                if rule.metric_name != metric.name:
                    continue

                # Check if metric triggers rule
                if rule.condition(metric.value, metric.reference_range):
                    # Generate recommendation
                    severity = self._determine_severity(metric.value, metric.reference_range)
                    template = rule.templates.get(severity, rule.templates.get("default", ""))

                    # Format template with values
                    recommendation = template.format(
                        value=metric.value,
                        unit=metric.unit,
                        target_min=metric.reference_range[0],
                        target_max=metric.reference_range[1],
                    )

                    # Store with priority
                    recommendations.append((rule.priority, recommendation))

        # Sort by priority (lower = more critical) and return strings
        recommendations.sort(key=lambda x: x[0])
        return [rec for _, rec in recommendations]

    def _determine_severity(
        self,
        value: float,
        reference_range: tuple[float, float],
    ) -> str:
        """Determine severity level for recommendation.

        Args:
            value: Metric value.
            reference_range: (min_good, max_good) range.

        Returns:
            Severity key: "too_low", "too_high", or "default".
        """
        min_good, max_good = reference_range

        if value < min_good:
            return "too_low"
        elif value > max_good:
            return "too_high"
        else:
            return "default"

    def _build_rules(self) -> None:
        """Build rule set for all element types."""
        # Add rules for each element type
        self._rules["waltz_jump"] = jump_rules.WALTZ_JUMP_RULES
        self._rules["toe_loop"] = jump_rules.TOE_LOOP_RULES
        self._rules["flip"] = jump_rules.FLIP_RULES
        self._rules["three_turn"] = three_turn_rules.THREE_TURN_RULES
