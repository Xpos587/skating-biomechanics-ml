"""Reference data storage and retrieval.

This module provides a file-based storage system for reference skating elements,
organized by element type (three_turn, waltz_jump, etc.).
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .reference_builder import ReferenceBuilder  # type: ignore[import-untyped]
    from .types import ReferenceData  # type: ignore[import-untyped]


class ReferenceStore:
    """Storage for reference skating elements.

    References are stored as .npz files organized by element type:
        store_dir/
            three_turn/
                expert_01.npz
                expert_02.npz
            waltz_jump/
                expert_01.npz
                ...
    """

    def __init__(self, store_dir: Path) -> None:
        """Initialize reference store.

        Args:
            store_dir: Directory containing reference .npz files.
        """
        self._store_dir = store_dir
        self._builder: ReferenceBuilder | None = None

    def set_builder(self, builder: "ReferenceBuilder") -> None:  # type: ignore[valid-type]
        """Set reference builder for loading .npz files.

        Args:
            builder: ReferenceBuilder instance.
        """
        self._builder = builder

    def add(self, ref: "ReferenceData") -> Path:  # type: ignore[valid-type]
        """Add reference to store.

        Args:
            ref: ReferenceData to add.

        Returns:
            Path to saved .npz file.

        Raises:
            RuntimeError: If ReferenceBuilder not set.
        """
        if self._builder is None:
            raise RuntimeError("ReferenceBuilder not set. Use set_builder() first.")

        element_dir = self._store_dir / ref.element_type
        return self._builder.save_reference(ref, element_dir)

    def get(self, element_type: str) -> list["ReferenceData"]:  # type: ignore[valid-type]
        """Get all references for an element type.

        Args:
            element_type: Element identifier (e.g., 'three_turn').

        Returns:
            List of ReferenceData. Empty list if element type not found.

        Raises:
            RuntimeError: If ReferenceBuilder not set.
        """
        if self._builder is None:
            raise RuntimeError("ReferenceBuilder not set. Use set_builder() first.")

        element_dir = self._store_dir / element_type

        if not element_dir.exists():
            return []

        references: list[ReferenceData] = []  # type: ignore[valid-type]

        for npz_file in element_dir.glob("*.npz"):
            try:
                ref = self._builder.load_reference(npz_file)
                references.append(ref)
            except Exception as e:
                # Skip invalid files
                print(f"Warning: Failed to load {npz_file}: {e}")

        return references

    def list_elements(self) -> list[str]:
        """List all element types in store.

        Returns:
            List of element type identifiers.
        """
        if not self._store_dir.exists():
            return []

        element_dirs = [d.name for d in self._store_dir.iterdir() if d.is_dir()]
        return sorted(element_dirs)

    def get_best_match(self, element_type: str) -> "ReferenceData | None":  # type: ignore[valid-type]
        """Get best reference match for an element type.

        Args:
            element_type: Element identifier.

        Returns:
            First available ReferenceData, or None if not found.

        Note:
            For MVP, returns the first reference. Future versions could
            implement more sophisticated matching (e.g., by athlete height).
        """
        references = self.get(element_type)

        if not references:
            return None

        # Return first reference (MVP: no sophisticated matching)
        return references[0]

    def ensure_store_dir(self) -> None:
        """Create store directory if it doesn't exist."""
        self._store_dir.mkdir(parents=True, exist_ok=True)
