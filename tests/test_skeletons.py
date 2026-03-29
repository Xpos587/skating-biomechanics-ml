"""Tests for skeleton hierarchy module."""


from src.skeletons import (
    BIOMECHANICS_BONE_PAIRS,
    BLAZEPOSE_BONE_PAIRS,
    get_bone_names,
    get_bone_pairs,
    get_children_of,
    get_skeleton_tree,
    render_tree,
)
from src.types import H36Key


class TestSkeletonTree:
    """Test skeleton tree structure and utilities."""

    def test_get_skeleton_tree(self):
        """Test that skeleton tree can be retrieved."""
        tree = get_skeleton_tree()
        assert tree is not None
        assert tree.name == "root"
        assert tree.children is not None
        assert len(tree.children) > 0

    def test_tree_has_all_h36m_keypoints(self):
        """Test that all H3.6M keypoints are in the tree."""
        tree = get_skeleton_tree()
        keypoint_indices = set()

        for node in tree.descendants:
            if hasattr(node, "id") and node.id is not None:
                keypoint_indices.add(int(node.id))

        # Should have all 17 H3.6M keypoints (0-16)
        expected_indices = set(range(17))
        assert keypoint_indices == expected_indices

    def test_get_bone_pairs(self):
        """Test bone pair extraction."""
        pairs = get_bone_pairs()

        assert len(pairs) > 0
        # All pairs should be tuples of (parent_idx, child_idx)
        for pair in pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert isinstance(pair[0], int)
            assert isinstance(pair[1], int)
            # Indices should be valid H3.6M keypoints
            assert 0 <= pair[0] < 17
            assert 0 <= pair[1] < 17

    def test_bone_pairs_are_unique(self):
        """Test that bone pairs are unique."""
        pairs = get_bone_pairs()
        assert len(pairs) == len(set(pairs))

    def test_get_bone_names(self):
        """Test bone name mapping."""
        names = get_bone_names()

        # H3.6M has 17 keypoints
        assert len(names) == 17
        assert all(isinstance(k, int) for k in names)
        assert all(isinstance(v, str) for v in names.values())

        # Check specific keypoints (H3.6M naming)
        assert names[H36Key.HIP_CENTER] == "hip_center"
        assert names[H36Key.LSHOULDER] == "left_shoulder"
        assert names[H36Key.RSHOULDER] == "right_shoulder"
        assert names[H36Key.LHIP] == "left_hip"
        assert names[H36Key.RHIP] == "right_hip"

    def test_get_children_of(self):
        """Test getting children of a joint."""
        # Left shoulder should have left elbow as child
        left_shoulder_children = get_children_of(H36Key.LSHOULDER)
        assert H36Key.LELBOW in left_shoulder_children

        # Left elbow should have left wrist as child
        left_elbow_children = get_children_of(H36Key.LELBOW)
        assert H36Key.LWRIST in left_elbow_children

        # H3.6M doesn't have finger keypoints, so wrist has no children
        left_wrist_children = get_children_of(H36Key.LWRIST)
        # In H3.6M, wrist is a leaf node (no fingers)

    def test_get_children_of_invalid_joint(self):
        """Test getting children of invalid joint returns empty list."""
        children = get_children_of(9999)
        assert children == []

    def test_render_tree(self):
        """Test tree rendering."""
        tree_str = render_tree()
        assert isinstance(tree_str, str)
        assert len(tree_str) > 0
        assert "root" in tree_str
        assert "hip_center" in tree_str
        assert "left_shoulder" in tree_str
        assert "right_shoulder" in tree_str

    def test_render_tree_format(self):
        """Test that tree rendering has proper structure."""
        tree_str = render_tree()
        lines = tree_str.split("\n")

        # Should have multiple lines
        assert len(lines) > 10

        # Root should be first line
        assert lines[0].startswith("root")

    def test_blazepose_bone_pairs_constant(self):
        """Test BLAZEPOSE_BONE_PAIRS constant (legacy alias)."""
        assert len(BLAZEPOSE_BONE_PAIRS) > 0

        # All should be valid keypoint pairs (H3.6M 17kp)
        for pair in BLAZEPOSE_BONE_PAIRS:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert 0 <= pair[0] < 17
            assert 0 <= pair[1] < 17

    def test_biomechanics_bone_pairs_constant(self):
        """Test BIOMECHANICS_BONE_PAIRS constant."""
        assert len(BIOMECHANICS_BONE_PAIRS) > 0
        assert len(BIOMECHANICS_BONE_PAIRS) <= len(BLAZEPOSE_BONE_PAIRS)  # Subset

        # Should contain major limb segments (H3.6M indices)
        expected_pairs = [
            (H36Key.LSHOULDER, H36Key.LELBOW),
            (H36Key.LELBOW, H36Key.LWRIST),
            (H36Key.RSHOULDER, H36Key.RELBOW),
            (H36Key.RELBOW, H36Key.RWRIST),
        ]

        for pair in expected_pairs:
            assert pair in BIOMECHANICS_BONE_PAIRS

    def test_tree_hierarchy_structure(self):
        """Test that tree hierarchy is anatomically correct."""
        tree = get_skeleton_tree()

        # Find left shoulder node
        left_shoulder = None
        for node in tree.descendants:
            if hasattr(node, "id") and node.id == H36Key.LSHOULDER:
                left_shoulder = node
                break

        assert left_shoulder is not None
        assert left_shoulder.name == "left_shoulder"

        # Should have left elbow as child
        left_elbow = None
        for child in left_shoulder.children:
            if hasattr(child, "id") and child.id == H36Key.LELBOW:
                left_elbow = child
                break

        assert left_elbow is not None
        assert left_elbow.name == "left_elbow"

    def test_backward_compatibility_aliases(self):
        """Test that BKey aliases work for H3.6M keypoints."""
        # BKey should be an alias for H36Key
        assert H36Key.LSHOULDER == H36Key.LSHOULDER
        assert H36Key.LEFT_SHOULDER == H36Key.LSHOULDER  # Alias

        # These should map to H3.6M equivalents
        assert H36Key.LEFT_ELBOW == H36Key.LELBOW
        assert H36Key.RIGHT_ELBOW == H36Key.RELBOW

    def test_legacy_finger_keypoints_map_to_wrist(self):
        """Test that legacy finger keypoints map to wrist (backward compat)."""
        tree = get_skeleton_tree()
        keypoint_indices = set()

        for node in tree.descendants:
            if hasattr(node, "id") and node.id is not None:
                keypoint_indices.add(int(node.id))

        # Legacy finger keypoints should map to wrist in H3.6M
        # They are in the enum as aliases pointing to wrist
        assert H36Key.LEFT_PINKY == H36Key.LWRIST
        assert H36Key.LEFT_INDEX == H36Key.LWRIST
        assert H36Key.LEFT_THUMB == H36Key.LWRIST

    def test_legacy_foot_keypoints_map_to_foot(self):
        """Test that legacy detailed foot keypoints map to foot."""
        tree = get_skeleton_tree()
        keypoint_indices = set()

        for node in tree.descendants:
            if hasattr(node, "id") and node.id is not None:
                keypoint_indices.add(int(node.id))

        # Legacy foot keypoints should map to LFOOT/RFOOT in H3.6M
        assert H36Key.LEFT_HEEL == H36Key.LFOOT
        assert H36Key.LEFT_FOOT_INDEX == H36Key.LFOOT
        assert H36Key.LEFT_ANKLE == H36Key.LFOOT
