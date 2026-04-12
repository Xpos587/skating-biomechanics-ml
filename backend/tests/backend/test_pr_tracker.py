"""Tests for PR tracker service."""

import pytest

from backend.app.services.pr_tracker import check_pr


@pytest.mark.parametrize(
    "direction,current_best,new_value,expected_is_pr",
    [
        ("higher", 0.38, 0.42, True),
        ("higher", 0.42, 0.38, False),
        ("higher", None, 0.42, True),
        ("lower", 0.10, 0.05, True),
        ("lower", 0.05, 0.10, False),
        ("lower", None, 0.05, True),
    ],
)
def test_check_pr(direction, current_best, new_value, expected_is_pr):
    is_pr, prev_best = check_pr(direction, current_best, new_value)
    assert is_pr == expected_is_pr
    if expected_is_pr:
        assert prev_best == current_best
    else:
        assert prev_best is None
