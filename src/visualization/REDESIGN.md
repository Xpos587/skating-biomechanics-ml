# Visualization System Redesign Plan

**Date:** 2026-03-29
**Goal:** Refactor 1402-line visualization.py into modular, maintainable architecture

---

## Current Problems

1. **Monolithic file** - 1402 lines, 20 functions in one file
2. **Mixed responsibilities** - 2D, 3D, HUD, text, colors all mixed
3. **Magic numbers** - hardcoded coordinates `(10, 30)`, colors, sizes
4. **Hard to debug** - no isolated components, can't test individual parts
5. **Hard to extend** - adding new HUD elements requires editing core file

---

## New Architecture

### Module Structure

```
src/visualization/
├── __init__.py              # Public API exports
├── config.py                 # Configuration classes
│   ├── VisualizationConfig    # Layout, colors, fonts, sizes
│   ├── LayerConfig            # Layer definitions
│   └── ThemeConfig            # Color schemes
├── core/                    # Core rendering utilities
│   ├── __init__.py
│   ├── colors.py              # Color gradients, palettes
│   ├── text.py                # Text rendering (Cyrillic support)
│   └── geometry.py            # Coordinate transformations
├── skeleton/                # Skeleton drawing
│   ├── __init__.py
│   ├── drawer.py              # Main skeleton renderer
│   └── joints.py              # Joint styling
├── hud/                     # HUD components
│   ├── __init__.py
│   ├── panel.py               # HUD panel system
│   ├── elements.py            # Individual HUD elements
│   └── layout.py              # Layout manager
├── layers/                  # Visualization layers
│   ├── __init__.py
│   ├── base.py                # Layer base class
│   ├── skeleton_layer.py       # Skeleton overlay
│   ├── velocity_layer.py      # Velocity vectors
│   ├── trail_layer.py         # Motion trails
│   ├── blade_layer.py         # Blade indicators
│   └── hud_layer.py           # Debug HUD
├── renderers/                # Specialized renderers
│   ├── __init__.py
│   ├── renderer_2d.py         # 2D pose rendering
│   └── renderer_3d.py         # 3D pose rendering
└── utils.py                  # Utility functions (keep existing)
```

---

## Phase 1: Configuration System (2 hours)

**Create `config.py`:**
```python
@dataclass
class VisualizationConfig:
    # Layout
    margin: int = 10
    panel_spacing: int = 15
    corner_radius: int = 5

    # Text
    font_path: str = "/usr/share/fonts/TTF/DejaVuSans.ttf"
    font_scale: float = 0.6
    font_thickness: int = 2

    # Skeleton
    line_width: int = 2
    joint_radius: int = 4
    confidence_threshold: float = 0.5

    # Colors
    color_left_side: tuple = (255, 0, 0)
    color_right_side: tuple = (0, 0, 255)
    color_center: tuple = (0, 255, 0)

    # HUD
    hud_bg_alpha: float = 0.6
    hud_text_color: tuple = (255, 255, 255)

@dataclass
class LayerConfig:
    enabled: bool = True
    z_index: int = 0
    opacity: float = 1.0
```

---

## Phase 2: Core Utilities (2 hours)

**Extract to `core/` module:**
- `colors.py` - Color gradients, palettes, theme management
- `text.py` - Text rendering with Cyrillic support
- `geometry.py` - Coordinate transformations, projections

---

## Phase 3: Skeleton Module (3 hours)

**Extract to `skeleton/` module:**
- `drawer.py` - Main skeleton renderer
- `joints.py` - Joint styling (confidence-based colors, sizes)

---

## Phase 4: HUD System (4 hours)

**Extract to `hud/` module:**
- `panel.py` - Reusable panel components
- `elements.py` - Individual HUD elements (frame counter, metrics display)
- `layout.py` - Layout manager (grid-based positioning)

---

## Phase 5: Layer System (3 hours)

**Create `layers/` module:**
- `base.py` - Layer base class with `render(frame, context)` interface
- `skeleton_layer.py` - Skeleton overlay
- `velocity_layer.py` - Velocity vectors
- `trail_layer.py` - Motion trails
- `blade_layer.py` - Blade indicators
- `hud_layer.py` - Debug HUD

---

## Phase 6: Migration (6 hours)

**Migrate existing code:**
1. Keep existing `visualization.py` as `visualization/legacy.py`
2. Create new modular implementations
3. Add deprecation warnings to legacy functions
4. Update `scripts/visualize_with_skeleton.py` to use new API

---

## Phase 7: Testing (2 hours)

**Add tests:**
- Config validation
- Individual layer rendering
- Integration tests for full pipeline

---

## Implementation Order

1. **config.py** - Configuration system
2. **core/colors.py** - Color utilities
3. **skeleton/drawer.py** - Skeleton module
4. **hud/panel.py** - HUD system
5. **layers/base.py** - Layer system
6. **Migration** - Move existing code, update imports
7. **Tests** - Add tests for new modules

---

## Benefits

✅ **Modularity** - Each module has single responsibility
✅ **Testability** - Can test individual components
✅ **Maintainability** - Easy to add features
✅ **Debuggability** - Clear separation of concerns
✅ **Configurability** - No more magic numbers
✅ **Extensibility** - Easy to add new layers/HUD elements

---

## Time Estimate

**Total: 22 hours** (spread over 2-3 days)

- Phase 1: 2 hours
- Phase 2: 2 hours
- Phase 3: 3 hours
- Phase 4: 4 hours
- Phase 5: 3 hours
- Phase 6: 6 hours
- Phase 7: 2 hours
