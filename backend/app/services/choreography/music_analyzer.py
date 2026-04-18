# backend/app/services/choreography/music_analyzer.py
"""Music analysis: BPM, structure, energy peaks using madmom + librosa.

This module is called from the arq worker (ml/skating_ml/worker.py),
NOT from the backend directly. The backend only stores/retrieves cached results.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _run_analysis(audio_path: str) -> dict:
    """Run music analysis pipeline.

    Steps:
    1. madmom BeatTracker -> BPM
    2. librosa RMS energy -> energy curve
    3. scipy.signal.find_peaks -> energy peaks
    4. MSAF -> structure boundaries (optional, can fail gracefully)

    Returns dict with: bpm, duration_sec, peaks, structure, energy_curve.
    """
    import librosa  # type: ignore[import-untyped]
    import numpy as np

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration_sec = float(len(y) / sr)

    # --- BPM via madmom ---
    bpm = None
    try:
        from madmom.features.beats import DBNBeatTracker  # type: ignore[import-untyped]

        act = DBNBeatTracker.preprocess(y, sr=sr)
        beat_frames = DBNBeatTracker.detect(act, fps=sr / 512)
        if len(beat_frames) > 1:
            intervals = np.diff(beat_frames) * 512 / sr
            bpm = float(60.0 / np.median(intervals))
    except Exception:  # noqa: BLE001
        logger.warning("madmom beat tracking failed, using librosa fallback")

    if bpm is None:
        bpm = float(librosa.beat.beat_track(y=y, sr=sr)[0])

    # --- Energy curve (RMS per 0.5s window) ---
    hop_length = int(sr * 0.5)
    rms = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]
    timestamps = [float(i * 0.5) for i in range(len(rms))]
    energy_curve = {"timestamps": timestamps, "values": [float(v) for v in rms]}

    # --- Energy peaks ---
    peaks: list[float] = []
    try:
        from scipy.signal import find_peaks

        rms_normalized = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
        peak_indices, _ = find_peaks(rms_normalized, height=0.6, distance=4)
        peaks = [timestamps[i] for i in peak_indices]
    except Exception:  # noqa: BLE001
        logger.warning("Peak detection failed")

    # --- Structure analysis (MSAF, optional) ---
    structure: list[dict] = []
    try:
        import msaf  # type: ignore[import-untyped]

        boundaries, labels = msaf.process(audio_path, boundaries_id="sf", labels_id="foote")
        for i in range(len(boundaries) - 1):
            structure.append(
                {
                    "type": labels[i] if i < len(labels) else "unknown",
                    "start": float(boundaries[i]),
                    "end": float(boundaries[i + 1]),
                }
            )
    except Exception:  # noqa: BLE001
        logger.warning("MSAF structure analysis failed -- using empty structure")

    return {
        "bpm": round(bpm, 1),
        "duration_sec": round(duration_sec, 1),
        "peaks": peaks,
        "structure": structure,
        "energy_curve": energy_curve,
    }


def analyze_music_sync(audio_path: str) -> dict:
    """Analyze a music file synchronously. Called from arq worker."""
    return _run_analysis(audio_path)


def extract_features_for_csp(analysis: dict) -> dict:
    """Extract CSP-relevant features from a full analysis result."""
    return {
        "duration": analysis.get("duration_sec", 180.0),
        "peaks": analysis.get("peaks", []),
        "structure": analysis.get("structure", []),
    }
