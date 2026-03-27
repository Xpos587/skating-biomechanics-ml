"""Reference dataset module for expert skating elements."""

from skating_biomechanics_ml.references.element_defs import (
    ELEMENT_DEFS,
    ElementDef,
    get_element_def,
    is_jump,
    list_supported_elements,
)
from skating_biomechanics_ml.references.reference_builder import ReferenceBuilder
from skating_biomechanics_ml.references.reference_store import ReferenceStore

__all__ = [
    "ElementDef",
    "ELEMENT_DEFS",
    "get_element_def",
    "list_supported_elements",
    "is_jump",
    "ReferenceBuilder",
    "ReferenceStore",
]
