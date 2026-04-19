"""Audio fingerprinting using chromaprint."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_fingerprint(audio_path: str) -> str | None:
    """Compute a 32-char hex fingerprint for an audio file.

    Uses chromaprint if available, falls back to a content hash.
    """
    path = Path(audio_path)
    if not path.exists():
        return None

    try:
        import chromaprint

        fp, _ = chromaprint.decode_file(str(path))
        if fp:
            return fp
    except Exception:  # noqa: BLE001
        logger.debug("chromaprint not available or failed, using fallback hash")

    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(262144), b""):
                h.update(chunk)
                if h.digest_size > 0:
                    break
        return h.hexdigest()[:32]  # Truncate to 32 chars for consistency
    except OSError:
        return None
