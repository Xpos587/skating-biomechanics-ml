"""H3.6M 17-keypoint skeleton hierarchy tree.

Provides hierarchical structure for skeleton operations:
- Bone pair extraction for angle calculations
- Parent-child relationships for constraints
- Tree traversal for biomechanics analysis

Based on:
- H3.6M 17kp topology (standard for 3D pose estimation)
- Pose2Sim skeleton structure
- AthletePose3D dataset format
"""

from dataclasses import dataclass

from anytree import Node, RenderTree

from .types import H36Key


@dataclass
class BoneNode:
    """Node in skeleton hierarchy.

    Attributes:
        name: Joint/bone name.
        keypoint_idx: H3.6M keypoint index (None for intermediate nodes).
        children: Child bones/joints.
    """

    name: str
    keypoint_idx: int | None = None
    children: list["BoneNode"] | None = None

    def to_anytree(self, parent: Node | None = None) -> Node:
        """Convert to anytree Node.

        Args:
            parent: Parent node in tree.

        Returns:
            anytree Node with this node's data.
        """
        node = Node(self.name, id=self.keypoint_idx, parent=parent)
        if self.children:
            for child in self.children:
                child.to_anytree(node)
        return node


# Define H3.6M 17kp hierarchy
# Standard format for 3D human pose estimation
H36M_HIERARCHY = BoneNode(
    name="root",
    children=[
        # Root/Hips
        BoneNode(
            name="hip_center",
            keypoint_idx=H36Key.HIP_CENTER,
            children=[
                # Right leg
                BoneNode(
                    name="right_hip",
                    keypoint_idx=H36Key.RHIP,
                    children=[
                        BoneNode(
                            name="right_knee",
                            keypoint_idx=H36Key.RKNEE,
                            children=[
                                BoneNode(name="right_foot", keypoint_idx=H36Key.RFOOT),
                            ],
                        ),
                    ],
                ),
                # Left leg
                BoneNode(
                    name="left_hip",
                    keypoint_idx=H36Key.LHIP,
                    children=[
                        BoneNode(
                            name="left_knee",
                            keypoint_idx=H36Key.LKNEE,
                            children=[
                                BoneNode(name="left_foot", keypoint_idx=H36Key.LFOOT),
                            ],
                        ),
                    ],
                ),
                # Spine/Torso
                BoneNode(
                    name="spine",
                    keypoint_idx=H36Key.SPINE,
                    children=[
                        BoneNode(
                            name="thorax",
                            keypoint_idx=H36Key.THORAX,
                            children=[
                                BoneNode(
                                    name="neck",
                                    keypoint_idx=H36Key.NECK,
                                    children=[
                                        BoneNode(name="head", keypoint_idx=H36Key.HEAD),
                                    ],
                                ),
                                # Right arm (from thorax in H3.6M)
                                BoneNode(
                                    name="right_shoulder",
                                    keypoint_idx=H36Key.RSHOULDER,
                                    children=[
                                        BoneNode(
                                            name="right_elbow",
                                            keypoint_idx=H36Key.RELBOW,
                                            children=[
                                                BoneNode(
                                                    name="right_wrist", keypoint_idx=H36Key.RWRIST
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                # Left arm (from thorax in H3.6M)
                                BoneNode(
                                    name="left_shoulder",
                                    keypoint_idx=H36Key.LSHOULDER,
                                    children=[
                                        BoneNode(
                                            name="left_elbow",
                                            keypoint_idx=H36Key.LELBOW,
                                            children=[
                                                BoneNode(
                                                    name="left_wrist", keypoint_idx=H36Key.LWRIST
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# Build the tree once at import time
_ROOT_NODE = H36M_HIERARCHY.to_anytree()


def get_skeleton_tree() -> Node:
    """Get the H3.6M skeleton hierarchy tree.

    Returns:
        Root node of the skeleton tree (anytree Node).
    """
    return _ROOT_NODE


def get_bone_pairs(tree: Node | None = None) -> list[tuple[int, int]]:
    """Extract all bone pairs (parent-child) from skeleton hierarchy.

    Useful for:
    - Drawing skeleton connections
    - Computing bone lengths
    - Applying kinematic constraints

    Args:
        tree: Skeleton tree (uses default H3.6M tree if None).

    Returns:
        List of (parent_idx, child_idx) tuples for all bones.
        Excludes nodes with None keypoint indices.
    """
    if tree is None:
        tree = _ROOT_NODE

    pairs = []
    for parent in tree.descendants:
        for child in parent.children:
            parent_idx = getattr(parent, "id", None)
            child_idx = getattr(child, "id", None)
            if parent_idx is not None and child_idx is not None:
                pairs.append((int(parent_idx), int(child_idx)))

    return pairs


def get_bone_names(tree: Node | None = None) -> dict[int, str]:
    """Get mapping from H3.6M keypoint indices to joint names.

    Args:
        tree: Skeleton tree (uses default H3.6M tree if None).

    Returns:
        Dictionary mapping keypoint_idx -> joint_name.
    """
    if tree is None:
        tree = _ROOT_NODE

    return {
        int(node.id): node.name
        for node in tree.descendants
        if hasattr(node, "id") and node.id is not None
    }


def get_children_of(parent_idx: int, tree: Node | None = None) -> list[int]:
    """Get child keypoint indices for a given parent joint.

    Args:
        parent_idx: H3.6M keypoint index.
        tree: Skeleton tree (uses default H3.6M tree if None).

    Returns:
        List of child keypoint indices.
    """
    if tree is None:
        tree = _ROOT_NODE

    for node in tree.descendants:
        if getattr(node, "id", None) == parent_idx:
            return [
                int(child.id)
                for child in node.children
                if hasattr(child, "id") and child.id is not None
            ]

    return []


def render_tree(tree: Node | None = None) -> str:
    """Render skeleton tree as ASCII string.

    Useful for debugging and documentation.

    Args:
        tree: Skeleton tree (uses default H3.6M tree if None).

    Returns:
        ASCII representation of the tree.
    """
    if tree is None:
        tree = _ROOT_NODE

    lines = []
    for pre, _, node in RenderTree(tree):
        keypoint_info = f" (kp={node.id})" if hasattr(node, "id") and node.id is not None else ""
        lines.append(f"{pre}{node.name}{keypoint_info}")

    return "\n".join(lines)


# H3.6M bone pairs for drawing (17-keypoint topology)
# These match the standard H3.6M skeleton connections
H36M_BONE_PAIRS = [
    # Torso/Spine
    (0, 7),  # hip_center -> spine
    (7, 8),  # spine -> thorax
    (8, 9),  # thorax -> neck
    (9, 10),  # neck -> head
    # Right leg
    (0, 1),  # hip_center -> right_hip
    (1, 2),  # right_hip -> right_knee
    (2, 3),  # right_knee -> right_foot
    # Left leg
    (0, 4),  # hip_center -> left_hip
    (4, 5),  # left_hip -> left_knee
    (5, 6),  # left_knee -> left_foot
    # Right arm (from thorax)
    (8, 14),  # thorax -> right_shoulder
    (14, 15),  # right_shoulder -> right_elbow
    (15, 16),  # right_elbow -> right_wrist
    # Left arm (from thorax)
    (8, 11),  # thorax -> left_shoulder
    (11, 12),  # left_shoulder -> left_elbow
    (12, 13),  # left_elbow -> left_wrist
]

# Legacy alias for backward compatibility
BLAZEPOSE_BONE_PAIRS = H36M_BONE_PAIRS

# Subset for biomechanics analysis (major limb segments)
BIOMECHANICS_BONE_PAIRS = [
    # Arms (detailed, including elbow)
    (11, 12),  # left_shoulder -> left_elbow
    (12, 13),  # left_elbow -> left_wrist
    (14, 15),  # right_shoulder -> right_elbow
    (15, 16),  # right_elbow -> right_wrist
    # Legs
    (4, 5),  # left_hip -> left_knee
    (5, 6),  # left_knee -> left_foot
    (1, 2),  # right_hip -> right_knee
    (2, 3),  # right_knee -> right_foot
    # Torso
    (0, 8),  # hip_center -> thorax (simplified)
]
