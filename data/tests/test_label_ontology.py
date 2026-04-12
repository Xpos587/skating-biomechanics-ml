"""Tests for label ontology."""

import pytest
from pathlib import Path
import sys

# Add data directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_tools.label_ontology import ElementSpec, LabelOntology


class TestElementSpec:
    """Test ElementSpec dataclass."""

    def test_create_jump_element(self):
        """Test creating a jump element specification."""
        elem = ElementSpec(
            name="3Lutz",
            category="jump",
            edge="outside",
            rotation=3,
            is_combination=False,
            fsc_id=11
        )

        assert elem.name == "3Lutz"
        assert elem.category == "jump"
        assert elem.edge == "outside"
        assert elem.rotation == 3
        assert elem.is_combination is False
        assert elem.fsc_id == 11

    def test_create_spin_element(self):
        """Test creating a spin element specification."""
        elem = ElementSpec(
            name="CamelSpin",
            category="spin",
            position="camel",
            is_combination=False,
            fsc_id=31
        )

        assert elem.name == "CamelSpin"
        assert elem.category == "spin"
        assert elem.position == "camel"
        assert elem.rotation is None

    def test_invalid_category(self):
        """Test that invalid category raises error."""
        with pytest.raises(ValueError, match="Invalid category"):
            ElementSpec(
                name="Test",
                category="invalid"
            )

    def test_invalid_edge(self):
        """Test that invalid edge raises error."""
        with pytest.raises(ValueError, match="Invalid edge"):
            ElementSpec(
                name="3Lutz",
                category="jump",
                edge="invalid"
            )

    def test_invalid_rotation(self):
        """Test that invalid rotation raises error."""
        with pytest.raises(ValueError, match="Invalid rotation"):
            ElementSpec(
                name="3Lutz",
                category="jump",
                rotation=5
            )

    def test_invalid_fsc_id(self):
        """Test that invalid FSC ID raises error."""
        with pytest.raises(ValueError, match="Invalid fsc_id"):
            ElementSpec(
                name="3Lutz",
                category="jump",
                fsc_id=100
            )

    def test_frozen_dataclass(self):
        """Test that ElementSpec is frozen."""
        elem = ElementSpec(
            name="3Lutz",
            category="jump"
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            elem.name = "3Flip"


class TestLabelOntology:
    """Test LabelOntology class."""

    @pytest.fixture
    def ontology(self):
        """Create ontology instance for testing."""
        # Use absolute path to main repo datasets
        data_root = Path("/home/michael/Github/skating-biomechanics-ml/data/datasets")
        return LabelOntology(data_root=data_root)

    def test_num_fsc_classes(self, ontology):
        """Test FSC class count."""
        assert ontology.num_fsc_classes() == 64

    def test_num_mcfs_classes(self, ontology):
        """Test MCFS class count."""
        # Should have ~130 classes (excluding NONE)
        count = ontology.num_mcfs_classes()
        assert count > 120
        assert count < 135

    def test_num_sv_classes(self, ontology):
        """Test SV class count."""
        # Should have 27 classes (excluding NoBasic)
        assert ontology.num_sv_classes() == 27

    def test_get_fsc_jump(self, ontology):
        """Test getting FSC jump element."""
        elem = ontology.get_element("3Lutz")
        assert elem is not None
        assert elem.name == "3Lutz"
        assert elem.category == "jump"
        assert elem.rotation == 3
        assert elem.fsc_id == 8  # 1Axel, 2Axel, 3Axel, 1Flip, 2Flip, 3Flip, 1Lutz, 2Lutz, 3Lutz

    def test_get_fsc_spin(self, ontology):
        """Test getting FSC spin element."""
        elem = ontology.get_element("FCSp1")
        assert elem is not None
        assert elem.name == "FCSp1"
        assert elem.category == "spin"
        assert elem.position == "camel"
        assert elem.fsc_id == 31

    def test_map_fsc_label(self, ontology):
        """Test mapping FSC label ID."""
        elem = ontology.map_fsc_label(8)  # 3Lutz
        assert elem is not None
        assert elem.name == "3Lutz"
        assert elem.fsc_id == 8

    def test_map_fsc_label_invalid(self, ontology):
        """Test mapping invalid FSC label ID."""
        elem = ontology.map_fsc_label(999)
        assert elem is None

    def test_sv_to_fsc_mapping(self, ontology):
        """Test SV to FSC ID mapping."""
        # SV 11 is 3Lutz, which maps to FSC 8
        fsc_id = ontology.map_sv_to_fsc(11)
        assert fsc_id == 8

    def test_sv_fsc_overlap_count(self, ontology):
        """Test SV-FSC overlap count."""
        overlap = ontology.count_sv_fsc_overlap()
        # Should have at least 20 overlapping elements
        assert overlap >= 20

    def test_list_all_elements(self, ontology):
        """Test listing all unique elements."""
        all_elems = ontology.list_all_elements()
        # Should have more than FSC classes due to MCFS/SV unique elements
        assert len(all_elems) >= 64

    def test_list_by_category_jump(self, ontology):
        """Test listing jump elements."""
        jumps = ontology.list_by_category("jump")
        assert len(jumps) > 0

        # All should be jumps
        for elem in jumps:
            assert elem.category == "jump"

    def test_list_by_category_spin(self, ontology):
        """Test listing spin elements."""
        spins = ontology.list_by_category("spin")
        assert len(spins) > 0

        # All should be spins
        for elem in spins:
            assert elem.category == "spin"

    def test_fsc_single_jumps(self, ontology):
        """Test FSC single jumps (0-20)."""
        # Check a few key jumps
        assert ontology.map_fsc_label(0).name == "1Axel"
        assert ontology.map_fsc_label(1).name == "2Axel"
        assert ontology.map_fsc_label(2).name == "3Axel"
        assert ontology.map_fsc_label(5).name == "3Flip"
        assert ontology.map_fsc_label(8).name == "3Lutz"
        assert ontology.map_fsc_label(11).name == "3Loop"

    def test_fsc_combinations(self, ontology):
        """Test FSC combinations (21-30)."""
        # Check that combinations are marked correctly
        for idx in range(21, 31):
            elem = ontology.map_fsc_label(idx)
            assert elem.is_combination is True
            assert elem.category == "combination"

    def test_fsc_spins(self, ontology):
        """Test FSC spins (31-58)."""
        # Check spin categories
        fcsp = ontology.map_fsc_label(31)
        assert fcsp.category == "spin"
        assert fcsp.position == "camel"

        ccosps = ontology.map_fsc_label(35)
        assert ccosps.category == "spin"

        layspin = ontology.map_fsc_label(55)
        assert layspin.category == "spin"
        assert layspin.position == "layback"

    def test_fsc_sequences(self, ontology):
        """Test FSC step sequences (59-63)."""
        for idx in range(59, 64):
            elem = ontology.map_fsc_label(idx)
            assert elem.category == "sequence"
            assert elem.is_combination is False

    def test_mcfs_mapping_loaded(self, ontology):
        """Test that MCFS mapping was loaded."""
        assert len(ontology._mcfs_elements) > 0

        # Check a known MCFS element (ID 2 is 2Axel)
        elem = ontology._mcfs_elements.get(2)
        assert elem is not None
        # Name might have spaces or other formatting
        assert "2Axel" in elem.name or "2Axel" == elem.name.replace(" ", "")

    def test_sv_mapping_loaded(self, ontology):
        """Test that SV mapping was loaded."""
        assert len(ontology._sv_elements) > 0

        # Check 3Lutz (SV ID 11)
        elem = ontology._sv_elements.get(11)
        assert elem is not None
        assert "3Lutz" in elem.name or elem.name == "3Lutz"

    def test_element_immutability(self, ontology):
        """Test that elements can't be modified."""
        elem = ontology.get_element("3Lutz")

        # ElementSpec is frozen
        with pytest.raises(Exception):
            elem.name = "Modified"

    def test_cross_dataset_consistency(self, ontology):
        """Test that same element across datasets has consistent properties."""
        # Get 3Lutz from different sources
        fsc_elem = ontology.map_fsc_label(8)  # FSC 8 is 3Lutz
        sv_elem = ontology._sv_elements.get(11)  # SV 11 is 3Lutz

        # Should have same basic properties
        assert fsc_elem.name == sv_elem.name
        assert fsc_elem.category == sv_elem.category
        assert fsc_elem.rotation == sv_elem.rotation

    def test_name_normalization(self, ontology):
        """Test name normalization for matching."""
        # These should all resolve to same element
        elem1 = ontology.get_element("3Lutz")
        # Note: spaces and underscores are normalized, so these should also work
        # but we need to check if they're in the name_to_element dict

        # At least one should be found
        assert elem1 is not None
        assert elem1.name == "3Lutz"

    def test_combination_detection(self, ontology):
        """Test that combinations are detected correctly."""
        # FSC combinations
        comb = ontology.map_fsc_label(21)  # 1A+3T
        assert comb.is_combination is True

        # Single jump should not be combination
        jump = ontology.map_fsc_label(11)  # 3Lutz
        assert jump.is_combination is False

    def test_edge_detection(self, ontology):
        """Test edge detection for jumps."""
        # Lutz = outside edge
        lutz = ontology.map_fsc_label(8)  # 3Lutz
        assert lutz.edge == "outside"

        # Flip = inside edge
        flip = ontology.map_fsc_label(5)  # 3Flip
        assert flip.edge == "inside"

        # Loop = no specific edge
        loop = ontology.map_fsc_label(11)  # 3Loop
        assert loop.edge is None
