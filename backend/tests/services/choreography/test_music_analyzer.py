# backend/tests/services/choreography/test_music_analyzer.py
"""Tests for music analyzer."""

from unittest.mock import patch

import pytest
from app.services.choreography.music_analyzer import (
    analyze_music_sync,
    extract_features_for_csp,
)


def test_extract_features_for_csp_basic():
    analysis = {
        "bpm": 120.0,
        "duration_sec": 180.0,
        "peaks": [10.0, 25.0, 40.0, 60.0],
        "structure": [{"type": "verse", "start": 0.0, "end": 30.0}],
    }
    features = extract_features_for_csp(analysis)
    assert features["duration"] == 180.0
    assert features["peaks"] == [10.0, 25.0, 40.0, 60.0]
    assert features["structure"] == [{"type": "verse", "start": 0.0, "end": 30.0}]


def test_extract_features_handles_missing_peaks():
    analysis = {"bpm": 120.0, "duration_sec": 180.0}
    features = extract_features_for_csp(analysis)
    assert features["duration"] == 180.0
    assert features["peaks"] == []


def test_analyze_music_sync_returns_expected_keys():
    """analyze_music_sync returns all required keys."""
    with patch("app.services.choreography.music_analyzer._run_analysis") as mock_run:
        mock_run.return_value = {
            "bpm": 120.0,
            "duration_sec": 180.0,
            "peaks": [30.0, 60.0],
            "structure": [{"type": "verse", "start": 0.0, "end": 30.0}],
            "energy_curve": {"timestamps": [0.0, 0.5], "values": [0.1, 0.2]},
        }
        result = analyze_music_sync("/fake/path.mp3")

    assert "bpm" in result
    assert "duration_sec" in result
    assert "peaks" in result
    assert "structure" in result
    assert "energy_curve" in result


def test_extract_features_for_csp():
    """extract_features_for_csp extracts only CSP-relevant fields."""
    full = {
        "bpm": 120.0,
        "duration_sec": 180.0,
        "peaks": [30.0, 60.0],
        "structure": [{"type": "verse", "start": 0.0, "end": 30.0}],
        "energy_curve": {"timestamps": [0.0, 0.5], "values": [0.1, 0.2]},
    }
    features = extract_features_for_csp(full)

    assert features["duration"] == 180.0
    assert features["peaks"] == [30.0, 60.0]
    assert features["structure"] == [{"type": "verse", "start": 0.0, "end": 30.0}]
    assert "bpm" not in features
    assert "energy_curve" not in features
