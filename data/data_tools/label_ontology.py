"""Unified label ontology for figure skating elements.

Maps FSC (64 classes), MCFS (132 classes), and SkatingVerse (28 classes)
to a unified element specification.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ElementSpec:
    """Unified specification for a figure skating element.

    Attributes:
        name: Canonical name like "3Lutz" or "CamelSpin"
        category: Element category - "jump" | "spin" | "sequence" | "combination"
        edge: Edge type for jumps - "inside" | "outside" | None
        rotation: Number of rotations for jumps (1-4), None for spins
        position: Spin position - "camel" | "sit" | "upright" | "layback" | None
        is_combination: True if this is a multi-element combination
        fsc_id: Figure-Skating-Classification class ID (0-63)
        mcfs_id: MCFS (MMFS) class ID (1-132)
        sv_id: SkatingVerse class ID (0-27)
    """
    name: str
    category: str
    edge: Optional[str] = None
    rotation: Optional[int] = None
    position: Optional[str] = None
    is_combination: bool = False
    fsc_id: Optional[int] = None
    mcfs_id: Optional[int] = None
    sv_id: Optional[int] = None

    def __post_init__(self):
        """Validate element specification."""
        valid_categories = {"jump", "spin", "sequence", "combination", "unknown"}
        if self.category not in valid_categories:
            raise ValueError(f"Invalid category: {self.category}")

        if self.edge is not None and self.edge not in {"inside", "outside"}:
            raise ValueError(f"Invalid edge: {self.edge}")

        if self.rotation is not None and not (1 <= self.rotation <= 4):
            raise ValueError(f"Invalid rotation: {self.rotation}")

        if self.position is not None:
            valid_positions = {"camel", "sit", "upright", "layback"}
            if self.position not in valid_positions:
                raise ValueError(f"Invalid position: {self.position}")

        # Validate IDs
        if self.fsc_id is not None and not (0 <= self.fsc_id < 64):
            raise ValueError(f"Invalid fsc_id: {self.fsc_id}")

        if self.sv_id is not None and not (0 <= self.sv_id <= 27):
            raise ValueError(f"Invalid sv_id: {self.sv_id}")


class LabelOntology:
    """Unified label ontology for figure skating elements.

    Maps element names and IDs across three datasets:
    - FSC: Figure-Skating-Classification (64 classes)
    - MCFS: MMFS dataset (132 classes)
    - SV: SkatingVerse (28 classes)
    """

    def __init__(self, data_root: Optional[Path] = None):
        """Initialize the label ontology.

        Args:
            data_root: Path to data directory containing datasets.
                      Defaults to ../../../datasets from this file.
        """
        if data_root is None:
            # Default to data/datasets relative to this file
            data_root = Path(__file__).parent.parent.parent / "datasets"

        self._data_root = Path(data_root)
        self._fsc_elements: Dict[int, ElementSpec] = {}
        self._mcfs_elements: Dict[int, ElementSpec] = {}
        self._sv_elements: Dict[int, ElementSpec] = {}
        self._name_to_element: Dict[str, ElementSpec] = {}

        # Load all mappings
        self._load_fsc_mapping()
        self._load_mcfs_mapping()
        self._load_sv_mapping()

    def _load_fsc_mapping(self):
        """Load Figure-Skating-Classification 64-class mapping.

        Based on README structure:
        - 0-20: Single jumps (1Axel, 2Axel, ..., 4Toeloop, 4Salchow, 4Flip)
        - 21-30: Combinations (1A+3T, 1A+3A, 2A+3T, etc.)
        - 31-34: FCSp (Foot Change Camel Spin) grades 1-4
        - 35-38: CCoSp grades 1-4
        - 39-42: ChCamelSp grades 1-4
        - 43-46: ChComboSp grades 1-4
        - 47-50: ChSitSp grades 1-4
        - 51-54: FlySitSp grades 1-4
        - 55-58: LaybackSp grades 1-4
        - 59-62: StepSeq grades 1-4
        - 63: ChoreoSeq
        """
        # Single jumps (0-20)
        single_jumps = [
            ("1Axel", 1, None), ("2Axel", 2, None), ("3Axel", 3, None),
            ("1Flip", 1, "inside"), ("2Flip", 2, "inside"), ("3Flip", 3, "inside"),
            ("1Lutz", 1, "outside"), ("2Lutz", 2, "outside"), ("3Lutz", 3, "outside"),
            ("1Loop", 1, None), ("2Loop", 2, None), ("3Loop", 3, None),
            ("1Salchow", 1, None), ("2Salchow", 2, None), ("3Salchow", 3, None),
            ("1Toeloop", 1, None), ("2Toeloop", 2, None), ("3Toeloop", 3, None),
            ("4Toeloop", 4, None), ("4Salchow", 4, None), ("4Flip", 4, "inside"),
        ]

        for idx, (name, rotation, edge) in enumerate(single_jumps):
            elem = ElementSpec(
                name=name,
                category="jump",
                edge=edge,
                rotation=rotation,
                is_combination=False,
                fsc_id=idx
            )
            self._fsc_elements[idx] = elem
            self._name_to_element[name] = elem

        # Combinations (21-30)
        combinations = [
            "1A+3T", "1A+3A", "2A+3T", "2A+3A", "2A+1Eu+3S",
            "3F+3T", "3F+2T+2Lo", "3Lz+3T", "3Lz+3Lo", "Comb"
        ]

        for idx, name in enumerate(combinations, start=21):
            elem = ElementSpec(
                name=name,
                category="combination",
                is_combination=True,
                fsc_id=idx
            )
            self._fsc_elements[idx] = elem
            self._name_to_element[name] = elem

        # Spins (31-58)
        spin_types = [
            ("FCSp", "camel", 31, 34),      # Foot Change Camel Spin
            ("CCoSp", "sit", 35, 38),       # Catch Foot Combination Spin
            ("ChCamelSp", "camel", 39, 42), # Change Camel Spin
            ("ChComboSp", "camel", 43, 46), # Change Combination Spin
            ("ChSitSp", "sit", 47, 50),     # Change Sit Spin
            ("FlySitSp", "sit", 51, 54),    # Fly Sit Spin
            ("LaybackSp", "layback", 55, 58), # Layback Spin
        ]

        for prefix, position, start_idx, end_idx in spin_types:
            for grade in range(1, 5):  # grades 1-4
                idx = start_idx + (grade - 1)
                name = f"{prefix}{grade}"
                elem = ElementSpec(
                    name=name,
                    category="spin",
                    position=position,
                    is_combination=False,
                    fsc_id=idx
                )
                self._fsc_elements[idx] = elem
                self._name_to_element[name] = elem

        # Step sequences (59-62)
        for grade in range(1, 5):
            idx = 58 + grade
            name = f"StepSeq{grade}"
            elem = ElementSpec(
                name=name,
                category="sequence",
                is_combination=False,
                fsc_id=idx
            )
            self._fsc_elements[idx] = elem
            self._name_to_element[name] = elem

        # Choreo sequence (63)
        elem = ElementSpec(
            name="ChoreoSeq",
            category="sequence",
            is_combination=False,
            fsc_id=63
        )
        self._fsc_elements[63] = elem
        self._name_to_element["ChoreoSeq"] = elem

    def _load_mcfs_mapping(self):
        """Load MCFS (MMFS) 132-class mapping.

        Reads from data/datasets/mcfs/mapping.txt
        Format: "ID NAME" per line
        Skips "NONE" class (ID 0)
        """
        mapping_file = self._data_root / "mcfs" / "mapping.txt"

        if not mapping_file.exists():
            # Fall back to absolute path from main repo
            mapping_file = Path("/home/michael/Github/skating-biomechanics-ml/data/datasets/mcfs/mapping.txt")

        if not mapping_file.exists():
            raise FileNotFoundError(f"MCFS mapping file not found: {mapping_file}")

        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Format: "ID NAME" - split on first space
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    continue

                idx = int(parts[0])
                name = parts[1]

                # Skip NONE class
                if name == "NONE":
                    continue

                # Try to match to existing FSC element
                elem = self._match_element(name, mcfs_id=idx)

                self._mcfs_elements[idx] = elem
                if elem.name not in self._name_to_element:
                    self._name_to_element[elem.name] = elem

    def _load_sv_mapping(self):
        """Load SkatingVerse 28-class mapping.

        Reads from data/datasets/skatingverse/mapping.txt
        Format: "NAME ID" per line (reversed from MCFS!)
        Skips "No Basic" class (ID 12)
        """
        mapping_file = self._data_root / "skatingverse" / "mapping.txt"

        if not mapping_file.exists():
            # Fall back to absolute path from main repo
            mapping_file = Path("/home/michael/Github/skating-biomechanics-ml/data/datasets/skatingverse/mapping.txt")

        if not mapping_file.exists():
            raise FileNotFoundError(f"SkatingVerse mapping file not found: {mapping_file}")

        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Format: "NAME ID" - but NAME can have spaces (e.g., "Camel Spin 3")
                # So we need to split from the right
                parts = line.rsplit(maxsplit=1)
                if len(parts) != 2:
                    continue

                name = parts[0]
                idx = int(parts[1])

                # Skip NoBasic class
                if name == "No Basic":
                    continue

                # Normalize name (remove spaces)
                normalized_name = name.replace(" ", "")

                # Try to match to existing FSC element
                elem = self._match_element(normalized_name, sv_id=idx)

                self._sv_elements[idx] = elem
                if elem.name not in self._name_to_element:
                    self._name_to_element[elem.name] = elem

    def _match_element(self, name: str, fsc_id: Optional[int] = None,
                       mcfs_id: Optional[int] = None, sv_id: Optional[int] = None) -> ElementSpec:
        """Match a dataset element to an existing FSC element or create new one.

        Args:
            name: Element name from dataset
            fsc_id: FSC class ID (if known)
            mcfs_id: MCFS class ID (if known)
            sv_id: SV class ID (if known)

        Returns:
            ElementSpec for this element
        """
        # Try multiple name variants to find existing element
        # 1. Exact name (with spaces/underscores removed)
        exact = name.replace(" ", "").replace("_", "")

        # 2. Normalized name (with abbreviations)
        normalized = self._normalize_name(name)

        # Find existing element
        existing = None
        for variant in [exact, normalized, name]:
            if variant in self._name_to_element:
                existing = self._name_to_element[variant]
                break

        if existing:
            # Create new ElementSpec with additional IDs
            return ElementSpec(
                name=existing.name,
                category=existing.category,
                edge=existing.edge,
                rotation=existing.rotation,
                position=existing.position,
                is_combination=existing.is_combination,
                fsc_id=fsc_id if fsc_id is not None else existing.fsc_id,
                mcfs_id=mcfs_id if mcfs_id is not None else existing.mcfs_id,
                sv_id=sv_id if sv_id is not None else existing.sv_id
            )

        # Parse element to create new spec
        category, edge, rotation, position, is_combination = self._parse_element(name)

        return ElementSpec(
            name=name,
            category=category,
            edge=edge,
            rotation=rotation,
            position=position,
            is_combination=is_combination,
            fsc_id=fsc_id,
            mcfs_id=mcfs_id,
            sv_id=sv_id
        )

    def _normalize_name(self, name: str) -> str:
        """Normalize element name for matching.

        Removes spaces, underscores, and standardizes abbreviations.
        """
        normalized = name.replace(" ", "").replace("_", "")

        # Standardize abbreviations
        abbrev_map = {
            "Axel": "A",
            "Lutz": "Lz",
            "Flip": "F",
            "Loop": "Lo",
            "Salchow": "S",
            "Toeloop": "T",
            "Spin": "Sp",
            "Sequence": "Seq",
        }

        for full, abbrev in abbrev_map.items():
            normalized = normalized.replace(full, abbrev)

        return normalized

    def _parse_element(self, name: str) -> tuple:
        """Parse element name to extract properties.

        Returns:
            (category, edge, rotation, position, is_combination)
        """
        # Check for combination (contains + or _)
        if "+" in name or "_" in name:
            return ("combination", None, None, None, True)

        # Check for spin
        if "Spin" in name or "Sp" in name:
            position = None
            if "Camel" in name:
                position = "camel"
            elif "Sit" in name:
                position = "sit"
            elif "Upright" in name:
                position = "upright"
            elif "Layback" in name:
                position = "layback"
            return ("spin", None, None, position, False)

        # Check for sequence
        if "Sequence" in name or "Seq" in name or "Step" in name:
            return ("sequence", None, None, None, False)

        # Parse jumps
        # Match rotation (1-4 at start)
        import re
        jump_match = re.match(r'^(\d)([A-Za-z]+)', name)
        if jump_match:
            rotation = int(jump_match.group(1))
            jump_name = jump_match.group(2)

            # Determine edge from jump type
            edge = None
            if "Flip" in jump_name or "F" in jump_name:
                edge = "inside"
            elif "Lutz" in jump_name or "Lz" in jump_name:
                edge = "outside"

            return ("jump", edge, rotation, None, False)

        # Default to unknown
        return ("unknown", None, None, None, False)

    # Public API

    def num_fsc_classes(self) -> int:
        """Return number of FSC classes."""
        return 64

    def num_mcfs_classes(self) -> int:
        """Return number of MCFS classes."""
        return len(self._mcfs_elements)

    def num_sv_classes(self) -> int:
        """Return number of SV classes (excluding NoBasic)."""
        return len(self._sv_elements)

    def get_element(self, name: str) -> Optional[ElementSpec]:
        """Get element specification by name.

        Args:
            name: Element name (e.g., "3Lutz", "CamelSpin")

        Returns:
            ElementSpec or None if not found
        """
        # Try exact match first (with spaces removed)
        exact = name.replace(" ", "").replace("_", "")
        if exact in self._name_to_element:
            return self._name_to_element[exact]

        # Try normalized match
        normalized = self._normalize_name(name)
        return self._name_to_element.get(normalized)

    def map_fsc_label(self, fsc_id: int) -> Optional[ElementSpec]:
        """Map FSC class ID to element specification.

        Args:
            fsc_id: FSC class ID (0-63)

        Returns:
            ElementSpec or None if not found
        """
        return self._fsc_elements.get(fsc_id)

    def map_sv_to_fsc(self, sv_id: int) -> Optional[int]:
        """Map SkatingVerse class ID to FSC class ID.

        Args:
            sv_id: SV class ID (0-27)

        Returns:
            FSC class ID or None if no mapping exists
        """
        if sv_id not in self._sv_elements:
            return None

        sv_elem = self._sv_elements[sv_id]
        return sv_elem.fsc_id

    def count_sv_fsc_overlap(self) -> int:
        """Count number of SV classes that map to FSC classes.

        Returns:
            Number of SV elements with valid FSC ID mapping
        """
        count = 0
        for sv_elem in self._sv_elements.values():
            if sv_elem.fsc_id is not None:
                count += 1
        return count

    def list_all_elements(self) -> List[ElementSpec]:
        """List all unique elements across all datasets."""
        return list(set(self._name_to_element.values()))

    def list_by_category(self, category: str) -> List[ElementSpec]:
        """List all elements in a category.

        Args:
            category: "jump" | "spin" | "sequence" | "combination"

        Returns:
            List of ElementSpec in this category
        """
        return [elem for elem in self._name_to_element.values() if elem.category == category]
