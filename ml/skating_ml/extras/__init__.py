"""Optional ML models for enhanced analysis.

Models are lazy-loaded via ModelRegistry and individually toggleable.
No model runs unless explicitly requested via CLI flags or API params.
"""

from skating_ml.extras.model_registry import ModelRegistry

__all__ = ["ModelRegistry"]
