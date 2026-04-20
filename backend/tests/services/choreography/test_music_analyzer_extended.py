"""Extended tests for music_analyzer with real audio files.

Tests _run_analysis end-to-end with a synthetic WAV file and
extract_features_for_csp edge cases.
"""

from __future__ import annotations

import struct
import sys
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

librosa = pytest.importorskip("librosa", reason="librosa not installed")


@pytest.fixture
def sample_audio(tmp_path):
    """Create a minimal 2-second WAV file (silence) for analysis."""
    path = str(tmp_path / "test.wav")
    sr = 22050
    duration = 2
    n_samples = sr * duration
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for _ in range(n_samples):
            wf.writeframes(struct.pack("<h", 0))
    return path


@pytest.fixture
def sample_audio_with_tone(tmp_path):
    """Create a 2-second WAV file with a 440Hz sine tone for better BPM detection."""
    path = str(tmp_path / "tone.wav")
    sr = 22050
    duration = 2
    n_samples = sr * duration

    # Generate a simple sine wave at 440Hz
    t = np.arange(n_samples) / sr
    y = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y.tobytes())
    return path


class TestAnalyzeMusicSyncRealAudio:
    """Test analyze_music_sync with real audio files."""

    def test_analyze_music_sync_returns_expected_keys(self, sample_audio):
        """_run_analysis via analyze_music_sync returns all required keys."""
        from app.services.choreography.music_analyzer import analyze_music_sync

        result = analyze_music_sync(sample_audio)

        expected_keys = {"bpm", "duration_sec", "peaks", "structure", "energy_curve"}
        assert expected_keys.issubset(set(result.keys()))

    def test_analyze_music_sync_bpm_is_numeric(self, sample_audio):
        """BPM should be a non-negative float (silence may yield 0.0 from librosa fallback)."""
        from app.services.choreography.music_analyzer import analyze_music_sync

        result = analyze_music_sync(sample_audio)

        assert isinstance(result["bpm"], float)
        assert result["bpm"] >= 0

    def test_analyze_music_sync_duration(self, sample_audio):
        """Duration should be approximately 2 seconds."""
        from app.services.choreography.music_analyzer import analyze_music_sync

        result = analyze_music_sync(sample_audio)

        assert isinstance(result["duration_sec"], float)
        assert 1.5 < result["duration_sec"] < 3.0  # allow some tolerance

    def test_analyze_music_sync_energy_curve_format(self, sample_audio):
        """Energy curve should have matching timestamps and values lists."""
        from app.services.choreography.music_analyzer import analyze_music_sync

        result = analyze_music_sync(sample_audio)

        ec = result["energy_curve"]
        assert "timestamps" in ec
        assert "values" in ec
        assert len(ec["timestamps"]) == len(ec["values"])
        assert len(ec["timestamps"]) > 0
        # All timestamps should be non-negative
        assert all(t >= 0 for t in ec["timestamps"])

    def test_analyze_music_sync_peaks_is_list(self, sample_audio):
        """Peaks should be a list of floats (may be empty for silence)."""
        from app.services.choreography.music_analyzer import analyze_music_sync

        result = analyze_music_sync(sample_audio)

        assert isinstance(result["peaks"], list)
        for peak in result["peaks"]:
            assert isinstance(peak, float)

    def test_analyze_music_sync_structure_is_list(self, sample_audio):
        """Structure should be a list (may be empty if MSAF fails)."""
        from app.services.choreography.music_analyzer import analyze_music_sync

        result = analyze_music_sync(sample_audio)

        assert isinstance(result["structure"], list)

    def test_analyze_music_sync_tone_file(self, sample_audio_with_tone):
        """Analysis works with a sine tone audio file."""
        from app.services.choreography.music_analyzer import analyze_music_sync

        result = analyze_music_sync(sample_audio_with_tone)

        assert isinstance(result["bpm"], float)
        assert result["bpm"] >= 0  # librosa fallback may return 0 for simple tones
        assert isinstance(result["duration_sec"], float)
        assert 1.5 < result["duration_sec"] < 3.0


class TestExtractFeaturesForCspExtended:
    """Extended tests for extract_features_for_csp."""

    def test_extract_features_for_csp_defaults_missing_keys(self):
        """When keys are missing, defaults are used."""
        from app.services.choreography.music_analyzer import extract_features_for_csp

        # Empty analysis dict
        features = extract_features_for_csp({})

        assert features["duration"] == 180.0  # default
        assert features["peaks"] == []  # default
        assert features["structure"] == []  # default

    def test_extract_features_for_csp_no_bpm_in_output(self):
        """BPM is not included in CSP features."""
        from app.services.choreography.music_analyzer import extract_features_for_csp

        analysis = {
            "bpm": 140.0,
            "duration_sec": 200.0,
            "peaks": [10.0],
            "structure": [{"type": "chorus", "start": 0, "end": 30}],
            "energy_curve": {"timestamps": [0.0], "values": [0.5]},
        }
        features = extract_features_for_csp(analysis)

        assert "bpm" not in features
        assert "energy_curve" not in features

    def test_extract_features_for_csp_preserves_structure(self):
        """Structure entries are passed through unchanged."""
        from app.services.choreography.music_analyzer import extract_features_for_csp

        structure = [
            {"type": "intro", "start": 0.0, "end": 10.0},
            {"type": "verse", "start": 10.0, "end": 40.0},
            {"type": "chorus", "start": 40.0, "end": 60.0},
        ]
        analysis = {"duration_sec": 60.0, "peaks": [], "structure": structure}
        features = extract_features_for_csp(analysis)

        assert features["structure"] == structure
        assert len(features["structure"]) == 3

    def test_extract_features_for_csp_peaks_passthrough(self):
        """Peaks list is passed through as-is."""
        from app.services.choreography.music_analyzer import extract_features_for_csp

        peaks = [5.0, 10.0, 15.0, 20.0, 25.0]
        analysis = {"duration_sec": 30.0, "peaks": peaks, "structure": []}
        features = extract_features_for_csp(analysis)

        assert features["peaks"] == peaks
        assert len(features["peaks"]) == 5

    def test_extract_features_for_csp_duration_from_analysis(self):
        """Duration is taken from duration_sec key."""
        from app.services.choreography.music_analyzer import extract_features_for_csp

        analysis = {"duration_sec": 245.5, "peaks": [], "structure": []}
        features = extract_features_for_csp(analysis)

        assert features["duration"] == 245.5

    def test_extract_features_for_csp_returns_only_three_keys(self):
        """Output dict has exactly 3 keys: duration, peaks, structure."""
        from app.services.choreography.music_analyzer import extract_features_for_csp

        analysis = {
            "bpm": 120.0,
            "duration_sec": 180.0,
            "peaks": [1.0, 2.0],
            "structure": [],
            "energy_curve": {},
            "extra_key": "should not appear",
        }
        features = extract_features_for_csp(analysis)

        assert set(features.keys()) == {"duration", "peaks", "structure"}


class TestMadmomBeatTracking:
    """Cover lines 40-44: madmom DBNBeatTracker paths."""

    def test_madmom_single_beat_frame_falls_back_to_librosa(self, tmp_path):
        """When madmom returns only 1 beat frame (len <= 1), falls back to librosa."""
        import wave

        path = str(tmp_path / "test_single_beat.wav")
        sr = 22050
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b"\x00\x00" * sr * 2)  # 2 seconds of silence

        # Build a mock madmom module whose DBNBeatTracker returns a single beat frame
        mock_tracker_cls = MagicMock()
        mock_tracker_cls.preprocess.return_value = np.zeros((50, 84))
        mock_tracker_cls.detect.return_value = np.array([5.0])  # Only 1 beat!

        mock_beats_mod = MagicMock()
        mock_beats_mod.DBNBeatTracker = mock_tracker_cls

        mock_madmom_mod = MagicMock()
        mock_madmom_mod.features = MagicMock()
        mock_madmom_mod.features.beats = mock_beats_mod

        from app.services.choreography.music_analyzer import analyze_music_sync

        with patch.dict(
            sys.modules,
            {
                "madmom": mock_madmom_mod,
                "madmom.features": mock_madmom_mod.features,
                "madmom.features.beats": mock_beats_mod,
            },
        ):
            result = analyze_music_sync(path)

        # BPM should come from librosa fallback (not None, not from madmom)
        assert result["bpm"] is not None
        assert isinstance(result["bpm"], float)
        # Verify madmom preprocess was called (lines 40-41 executed)
        mock_tracker_cls.preprocess.assert_called_once()
        mock_tracker_cls.detect.assert_called_once()

    def test_madmom_multiple_beats_computes_bpm(self, tmp_path):
        """When madmom returns 2+ beat frames, BPM is computed from intervals (lines 43-44)."""
        import wave

        path = str(tmp_path / "test_multi_beat.wav")
        sr = 22050
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b"\x00\x00" * sr * 2)  # 2 seconds

        # Return enough beat frames to trigger lines 42-44
        # BPM = 60 / median(diff(beat_frames) * 512 / sr)
        # For 120 BPM: interval_seconds = 0.5, interval_frames = 0.5 * 22050 / 512 ≈ 21.5
        beat_frames = np.array([10.0, 31.5, 53.0, 74.5, 96.0])

        mock_tracker_cls = MagicMock()
        mock_tracker_cls.preprocess.return_value = np.zeros((200, 84))
        mock_tracker_cls.detect.return_value = beat_frames

        mock_beats_mod = MagicMock()
        mock_beats_mod.DBNBeatTracker = mock_tracker_cls

        mock_madmom_mod = MagicMock()
        mock_madmom_mod.features = MagicMock()
        mock_madmom_mod.features.beats = mock_beats_mod

        from app.services.choreography.music_analyzer import analyze_music_sync

        with patch.dict(
            sys.modules,
            {
                "madmom": mock_madmom_mod,
                "madmom.features": mock_madmom_mod.features,
                "madmom.features.beats": mock_beats_mod,
            },
        ):
            result = analyze_music_sync(path)

        # BPM should be computed from madmom beat intervals (~120 BPM)
        assert result["bpm"] is not None
        assert isinstance(result["bpm"], float)
        assert 100.0 < result["bpm"] < 140.0  # roughly 120


class TestFindPeaksExceptionFallback:
    """Cover lines 68-69: scipy.signal.find_peaks raises an exception."""

    def test_find_peaks_exception_returns_empty_peaks(self, tmp_path):
        """When scipy.signal.find_peaks raises, peaks list is empty."""
        import wave

        path = str(tmp_path / "test_peaks_exc.wav")
        sr = 22050
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b"\x00\x00" * sr)  # 1 second

        from app.services.choreography.music_analyzer import analyze_music_sync

        with patch("scipy.signal.find_peaks", side_effect=ValueError("bad peak input")):
            result = analyze_music_sync(path)

        # peaks should be empty list (exception caught at line 68-69)
        assert isinstance(result["peaks"], list)
        assert result["peaks"] == []
        # energy_curve should still be populated (computed before find_peaks)
        assert "timestamps" in result["energy_curve"]
        assert "values" in result["energy_curve"]


class TestMsafStructureAnalysis:
    """Cover lines 79-81 and 88-89: msaf.process success and exception paths."""

    def test_msaf_process_exception_returns_empty_structure(self, tmp_path):
        """When msaf.process raises, structure is an empty list."""
        import wave

        path = str(tmp_path / "test_msaf_exc.wav")
        sr = 22050
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b"\x00\x00" * sr)  # 1 second

        mock_msaf = MagicMock()
        mock_msaf.process = MagicMock(side_effect=RuntimeError("msaf crashed"))

        from app.services.choreography.music_analyzer import analyze_music_sync

        with patch.dict(sys.modules, {"msaf": mock_msaf}):
            result = analyze_music_sync(path)

        # structure should be empty (exception caught at lines 88-89)
        assert result["structure"] == []
        # Other fields should still be populated
        assert isinstance(result["bpm"], float)
        assert isinstance(result["peaks"], list)

    def test_msaf_process_succeeds_builds_structure(self, tmp_path):
        """When msaf.process returns boundaries and labels, structure is built (lines 80-81)."""
        import wave

        path = str(tmp_path / "test_msaf_ok.wav")
        sr = 22050
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b"\x00\x00" * sr)  # 1 second

        mock_msaf = MagicMock()
        # Return 4 boundaries (3 segments) and 3 labels
        mock_msaf.process = MagicMock(
            return_value=(
                np.array([0.0, 10.0, 20.0, 30.0]),
                ["intro", "chorus", "outro"],
            )
        )

        from app.services.choreography.music_analyzer import analyze_music_sync

        with patch.dict(sys.modules, {"msaf": mock_msaf}):
            result = analyze_music_sync(path)

        # structure should have 3 entries (boundaries - 1)
        assert len(result["structure"]) == 3
        assert result["structure"][0]["type"] == "intro"
        assert result["structure"][0]["start"] == 0.0
        assert result["structure"][0]["end"] == 10.0
        assert result["structure"][1]["type"] == "chorus"
        assert result["structure"][2]["type"] == "outro"
        assert result["structure"][2]["end"] == 30.0
