"""Tests for audio fingerprinting."""

import wave
from pathlib import Path

from app.services.choreography.fingerprint import compute_fingerprint


def _create_wav(path: Path, duration_sec: float = 1.0, sample_rate: int = 22050) -> None:
    n_frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def test_compute_fingerprint_returns_32char_hex():
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = Path(f.name)
    try:
        _create_wav(tmp)
        fp = compute_fingerprint(str(tmp))
        assert fp is not None
        assert len(fp) == 32
        assert all(c in "0123456789abcdef" for c in fp)
    finally:
        tmp.unlink(missing_ok=True)


def test_compute_fingerprint_same_file_same_hash():
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp1 = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp2 = Path(f.name)
    try:
        _create_wav(tmp1)
        _create_wav(tmp2)
        fp1 = compute_fingerprint(str(tmp1))
        fp2 = compute_fingerprint(str(tmp2))
        assert fp1 == fp2
    finally:
        tmp1.unlink(missing_ok=True)
        tmp2.unlink(missing_ok=True)


def test_compute_fingerprint_missing_file_returns_none():
    result = compute_fingerprint("/nonexistent/path.wav")
    assert result is None
