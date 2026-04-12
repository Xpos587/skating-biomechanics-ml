# backend/tests/services/choreography/test_music_analyzer.py
"""Tests for music analyzer."""

from backend.app.services.choreography.music_analyzer import (
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
