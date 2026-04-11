"""Tests for metrics_registry.py."""

import pytest

from src.backend.metrics_registry import (
    ALL_ELEMENTS,
    JUMP_ELEMENTS,
    METRIC_REGISTRY,
    get_metrics_for_element,
)


class TestMetricRegistry:
    """Test the metric registry structure and contents."""

    def test_registry_has_all_known_metrics(self):
        """Verify all 12 expected metrics are present."""
        expected_metrics = {
            "airtime",
            "max_height",
            "relative_jump_height",
            "landing_knee_angle",
            "landing_knee_stability",
            "landing_trunk_recovery",
            "arm_position_score",
            "rotation_speed",
            "knee_angle",
            "trunk_lean",
            "edge_change_smoothness",
            "symmetry",
        }
        actual_metrics = set(METRIC_REGISTRY.keys())
        assert actual_metrics == expected_metrics, (
            f"Expected {len(expected_metrics)} metrics, "
            f"found {len(actual_metrics)}. Missing: {expected_metrics - actual_metrics}, "
            f"Extra: {actual_metrics - expected_metrics}"
        )

    def test_metric_def_fields(self):
        """Validate all required fields are present on each MetricDef."""
        for metric_name, metric_def in METRIC_REGISTRY.items():
            # Check all fields exist and have correct types
            assert hasattr(metric_def, "name"), f"{metric_name}: missing 'name'"
            assert hasattr(metric_def, "label_ru"), f"{metric_name}: missing 'label_ru'"
            assert hasattr(metric_def, "unit"), f"{metric_name}: missing 'unit'"
            assert hasattr(metric_def, "format"), f"{metric_name}: missing 'format'"
            assert hasattr(metric_def, "direction"), f"{metric_name}: missing 'direction'"
            assert hasattr(metric_def, "element_types"), f"{metric_name}: missing 'element_types'"
            assert hasattr(metric_def, "ideal_range"), f"{metric_name}: missing 'ideal_range'"

            # Type validation
            assert isinstance(metric_def.name, str), f"{metric_name}: name must be str"
            assert isinstance(metric_def.label_ru, str), f"{metric_name}: label_ru must be str"
            assert isinstance(metric_def.unit, str), f"{metric_name}: unit must be str"
            assert isinstance(metric_def.format, str), f"{metric_name}: format must be str"
            assert metric_def.direction in {"higher", "lower"}, (
                f"{metric_name}: direction must be 'higher' or 'lower'"
            )
            assert isinstance(metric_def.element_types, tuple), (
                f"{metric_name}: element_types must be tuple"
            )
            assert isinstance(metric_def.ideal_range, tuple), (
                f"{metric_name}: ideal_range must be tuple"
            )
            assert len(metric_def.ideal_range) == 2, (
                f"{metric_name}: ideal_range must have 2 values (min, max)"
            )

            # Valid unit values
            valid_units = {"s", "deg", "score", "norm", "ratio", "deg/s"}
            assert metric_def.unit in valid_units, (
                f"{metric_name}: unit '{metric_def.unit}' not in {valid_units}"
            )

    def test_jump_metrics_not_on_three_turn(self):
        """Jump-specific metrics should not apply to three_turn."""
        jump_only_metrics = {"airtime", "max_height", "rotation_speed"}

        three_turn_metrics = get_metrics_for_element("three_turn")
        three_turn_metric_names = set(three_turn_metrics.keys())

        for metric in jump_only_metrics:
            assert metric not in three_turn_metric_names, (
                f"{metric} should not apply to three_turn element"
            )

    def test_symmetry_on_all_elements(self):
        """Symmetry metric should apply to all 8 element types."""
        symmetry_def = METRIC_REGISTRY["symmetry"]
        assert set(symmetry_def.element_types) == set(ALL_ELEMENTS), (
            f"symmetry should apply to all {len(ALL_ELEMENTS)} elements. "
            f"Got: {symmetry_def.element_types}"
        )

    def test_get_metrics_for_element_jump(self):
        """Test get_metrics_for_element for a jump element."""
        waltz_jump_metrics = get_metrics_for_element("waltz_jump")

        # Should have all jump-specific metrics plus symmetry
        expected_jump_metrics = {
            "airtime",
            "max_height",
            "relative_jump_height",
            "landing_knee_angle",
            "landing_knee_stability",
            "landing_trunk_recovery",
            "arm_position_score",
            "rotation_speed",
            "symmetry",
        }
        assert set(waltz_jump_metrics.keys()) == expected_jump_metrics

        # Verify step-specific metrics are NOT included
        step_only_metrics = {"knee_angle", "trunk_lean", "edge_change_smoothness"}
        for metric in step_only_metrics:
            assert metric not in waltz_jump_metrics

    def test_get_metrics_for_element_step(self):
        """Test get_metrics_for_element for a step element."""
        three_turn_metrics = get_metrics_for_element("three_turn")

        # Should have step-specific metrics plus symmetry
        expected_step_metrics = {
            "knee_angle",
            "trunk_lean",
            "edge_change_smoothness",
            "symmetry",
        }
        assert set(three_turn_metrics.keys()) == expected_step_metrics

        # Verify jump-specific metrics are NOT included
        jump_only_metrics = {
            "airtime",
            "max_height",
            "relative_jump_height",
            "landing_knee_angle",
            "landing_knee_stability",
            "landing_trunk_recovery",
            "arm_position_score",
            "rotation_speed",
        }
        for metric in jump_only_metrics:
            assert metric not in three_turn_metrics

    def test_get_metrics_for_element_invalid(self):
        """Test get_metrics_for_element with invalid element type."""
        with pytest.raises(ValueError, match="Unknown element type"):
            get_metrics_for_element("invalid_element")

    def test_metric_def_is_frozen(self):
        """MetricDef should be immutable (frozen dataclass)."""
        from dataclasses import FrozenInstanceError

        metric_def = METRIC_REGISTRY["airtime"]
        with pytest.raises(FrozenInstanceError):
            metric_def.name = "changed"

    def test_direction_values(self):
        """Verify direction field has valid values."""
        for metric_name, metric_def in METRIC_REGISTRY.items():
            assert metric_def.direction in {"higher", "lower"}, (
                f"{metric_name}: direction '{metric_def.direction}' is invalid"
            )

    def test_ideal_range_ordering(self):
        """Verify ideal_range has min <= max for all metrics."""
        for metric_name, metric_def in METRIC_REGISTRY.items():
            min_val, max_val = metric_def.ideal_range
            assert min_val <= max_val, (
                f"{metric_name}: ideal_range min ({min_val}) > max ({max_val})"
            )

    def test_element_types_constants(self):
        """Verify JUMP_ELEMENTS and ALL_ELEMENTS constants are correct."""
        # JUMP_ELEMENTS should have 7 jump types
        assert len(JUMP_ELEMENTS) == 7, f"Expected 7 jump elements, got {len(JUMP_ELEMENTS)}"

        # ALL_ELEMENTS should be jumps + three_turn
        assert len(ALL_ELEMENTS) == 8, f"Expected 8 total elements, got {len(ALL_ELEMENTS)}"
        assert ALL_ELEMENTS == JUMP_ELEMENTS + ("three_turn",)

        # All element types in registry should be in ALL_ELEMENTS
        for metric_def in METRIC_REGISTRY.values():
            for element_type in metric_def.element_types:
                assert element_type in ALL_ELEMENTS, (
                    f"Element type '{element_type}' not in ALL_ELEMENTS constant"
                )
