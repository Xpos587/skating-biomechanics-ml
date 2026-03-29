"""Base layer class for visualization system.

Provides:
- Layer base class
- Layer context for passing data between layers
- Layer composition utilities
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.visualization.config import LayerConfig

# =============================================================================
# TYPE ALIASES
# =============================================================================

Frame = NDArray[np.uint8]  # OpenCV image (H, W, 3)
Pose2D = NDArray[np.float32]  # (17, 2) normalized or pixel coords
Pose3D = NDArray[np.float32]  # (17, 3) in meters


# =============================================================================
# LAYER CONTEXT
# =============================================================================


@dataclass
class LayerContext:
    """Context object passed to layers for rendering.

    Contains all the data a layer might need for rendering:
    - Frame metadata (width, height, fps)
    - Pose data (2D, 3D, confidences)
    - Analysis results (metrics, phases, blade states)
    - Rendering options (normalized coords, etc.)

    Attributes:
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        fps: Video frame rate.
        frame_idx: Current frame index.
        total_frames: Total number of frames.
        pose_2d: 2D pose array (17, 2) normalized or pixel coords.
        pose_3d: 3D pose array (17, 3) in meters.
        confidences: Joint confidence values (17,).
        metrics: List of analysis metrics.
        phase: Current phase name.
        blade_state: Current blade edge state.
        normalized: Whether 2D poses are normalized [0, 1].
        camera_distance: Camera distance for 3D projection.
        focal_length: Camera focal length for 3D projection.
        custom_data: Dict for custom layer-specific data.
    """

    # Frame metadata
    frame_width: int = 1920
    frame_height: int = 1080
    fps: float = 30.0
    frame_idx: int = 0
    total_frames: int | None = None

    # Pose data
    pose_2d: Pose2D | None = None
    pose_3d: Pose3D | None = None
    confidences: NDArray[np.float32] | None = None

    # Analysis results
    metrics: list[Any] = field(default_factory=list)
    phase: str | None = None
    blade_state: Any = None

    # Rendering options
    normalized: bool = True
    camera_distance: float = 3.0
    focal_length: int = 800

    # Custom data
    custom_data: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# BASE LAYER CLASS
# =============================================================================


class Layer(ABC):
    """Base class for visualization layers.

    All layers must inherit from this class and implement the render() method.

    Attributes:
        config: LayerConfig for this layer.
        name: Human-readable layer name.
        enabled: Whether layer is rendered.
        z_index: Drawing order (higher = drawn on top).
        opacity: Transparency level [0, 1].
    """

    def __init__(
        self,
        config: LayerConfig | None = None,
        name: str = "Layer",
    ):
        """Initialize layer.

        Args:
            config: LayerConfig for this layer.
            name: Human-readable layer name.
        """
        self.config = config or LayerConfig()
        self.name = name

    @property
    def enabled(self) -> bool:
        """Check if layer is enabled."""
        return self.config.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set layer enabled state."""
        self.config.enabled = value

    @property
    def z_index(self) -> int:
        """Get layer z-index."""
        return self.config.z_index

    @z_index.setter
    def z_index(self, value: int) -> None:
        """Set layer z-index."""
        self.config.z_index = value

    @property
    def opacity(self) -> float:
        """Get layer opacity."""
        return self.config.opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set layer opacity."""
        self.config.opacity = value

    def is_visible(self) -> bool:
        """Check if layer should be rendered.

        Returns:
            True if layer is enabled and opacity > 0.
        """
        return self.config.is_visible()

    @abstractmethod
    def render(
        self,
        frame: Frame,
        context: LayerContext,
    ) -> Frame:
        """Render this layer to frame.

        Args:
            frame: OpenCV image (H, W, 3) BGR format.
            context: LayerContext with rendering data.

        Returns:
            Frame with layer rendered (modified in place).

        Raises:
            NotImplementedError: Subclass must implement.
        """
        raise NotImplementedError("Subclass must implement render()")

    def __call__(
        self,
        frame: Frame,
        context: LayerContext,
    ) -> Frame:
        """Convenience method for rendering.

        Args:
            frame: OpenCV image.
            context: LayerContext.

        Returns:
            Frame with layer rendered if visible.
        """
        if self.is_visible():
            return self.render(frame, context)
        return frame


# =============================================================================
# LAYER UTILITIES
# =============================================================================


def sort_layers_by_z_index(layers: list[Layer]) -> list[Layer]:
    """Sort layers by z-index (drawing order).

    Args:
        layers: List of Layer objects.

    Returns:
        Sorted list with lowest z-index first.

    Example:
        >>> layers = [SkeletonLayer(), VelocityLayer(), HUDLayer()]
        >>> sorted_layers = sort_layers_by_z_index(layers)
    """
    return sorted(layers, key=lambda l: l.z_index)


def render_layers(
    frame: Frame,
    layers: list[Layer],
    context: LayerContext,
) -> Frame:
    """Render multiple layers to frame in z-order.

    Args:
        frame: OpenCV image.
        layers: List of Layer objects.
        context: LayerContext.

    Returns:
        Frame with all visible layers rendered.

    Example:
        >>> layers = [SkeletonLayer(), VelocityLayer(), HUDLayer()]
        >>> frame = render_layers(frame, layers, context)
    """
    # Sort by z-index
    sorted_layers = sort_layers_by_z_index(layers)

    # Render each visible layer
    for layer in sorted_layers:
        if layer.is_visible():
            frame = layer.render(frame, context)

    return frame


def create_layer_composite(
    layers: list[Layer],
) -> Layer:
    """Create composite layer from multiple layers.

    Args:
        layers: List of Layer objects.

    Returns:
        Composite Layer that renders all child layers.

    Example:
        >>> skeleton = SkeletonLayer()
        >>> velocity = VelocityLayer()
        >>> composite = create_layer_composite([skeleton, velocity])
        >>> frame = composite.render(frame, context)
    """

    class CompositeLayer(Layer):
        def __init__(self, child_layers: list[Layer]):
            super().__init__(name="Composite")
            self.child_layers = child_layers

        def render(self, frame: Frame, context: LayerContext) -> Frame:
            return render_layers(frame, self.child_layers, context)

    return CompositeLayer(layers)
