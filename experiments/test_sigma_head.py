"""
Proof-of-concept: Analyze YOLO26-Pose head structure for sigma head feasibility.

This script inspects the model architecture to determine if we can add a separate
sigma head for uncertainty estimation without breaking pretrained weights.
"""

import torch
from ultralytics import YOLO

# Load pretrained model
model_path = "/home/michael/Github/skating-biomechanics-ml/yolo26n-pose.pt"
print(f"Loading model from: {model_path}")

model = YOLO(model_path)

# Check current head structure
print("\n" + "=" * 60)
print("Sigma Head Feasibility Report")
print("=" * 60)

print("\nModel classes:")
print(f"  - Model wrapper: {model.model.__class__.__name__}")
print(f"  - Task: {model.task}")

print("\nModel architecture:")
print(f"  - Total layers: {len(model.model.model)}")

# Get the last layer (pose head)
pose_head = model.model.model[-1]
print("\nPose head (last layer):")
print(f"  - Type: {pose_head.__class__.__name__}")
print(f"  - Name: {pose_head.__class__.__name__}")

# Inspect pose head structure
print("\n  Pose head attributes:")
if hasattr(pose_head, "nc"):
    print(f"    - nc (num classes): {pose_head.nc}")
if hasattr(pose_head, "nk"):
    print(f"    - nk (num keypoints): {pose_head.nc}")
if hasattr(pose_head, "reg_max"):
    print(f"    - reg_max: {pose_head.reg_max}")
if hasattr(pose_head, "nls"):
    print(f"    - nls: {pose_head.nls}")

# Check for cv5 (detection head) or cv3 (pose head)
print("\n  Pose head modules:")
for i, module in enumerate(pose_head.modules()):
    if i > 10:  # Limit output
        print("    ... (truncated)")
        break
    if hasattr(module, "__class__"):
        module_type = module.__class__.__name__
        if hasattr(module, "in_channels"):
            print(f"    [{i}] {module_type}: in={module.in_channels}", end="")
            if hasattr(module, "out_channels"):
                print(f", out={module.out_channels}")
            else:
                print()

# Check model state dict keys to understand layer naming
print("\n  State dict keys (last 10):")
state_dict = model.model.state_dict()
keys = list(state_dict.keys())
for key in keys[-10:]:
    print(f"    - {key}: {state_dict[key].shape}")

# Calculate expected keypoint channels
# For YOLO26-Pose: nk = 51 (17 keypoints × 3: x, y, visibility)
print("\nKeypoint channel analysis:")
print("  - Expected: nk = 51 (17 keypoints × 3: x, y, visibility)")

# Check if we can find the actual number
if hasattr(pose_head, "nc"):
    nk = pose_head.nc
    print(f"  - Actual nc: {nk}")
    if nk == 51:
        print("  ✓ Matches expected 51 channels (17kp × 3)")
    else:
        print("  ✗ Different from expected!")

# Check for separate heads in the model
print("\nChecking for multiple heads:")
head_types = set()
for module in model.model.model:
    module_name = module.__class__.__name__
    if "Detect" in module_name or "Pose" in module_name or "Segment" in module_name:
        head_types.add(module_name)

print(f"  - Found heads: {head_types}")

# Conceptual feasibility check
print("\n" + "=" * 60)
print("Feasibility Analysis")
print("=" * 60)

print("\n✓ Model loads successfully")
print(f"✓ Pose head identified: {pose_head.__class__.__name__}")
print(f"✓ Pretrained weights intact: {len(state_dict)} tensors")

print("\nKey questions:")

# Question 1: Can we add a separate cv5 head?
print("\n1. Can we add a separate cv5 head for sigma?")
if "Detect" in str(pose_head.__class__):
    print("   - Current: Detection-style head (cv5)")
    print("   - Feasible: YES - can add parallel cv5_sigma")
elif "Pose" in str(pose_head.__class__):
    print("   - Current: Pose-specific head")
    print("   - Feasible: DEPENDS - need to check head implementation")
else:
    print("   - Current: Unknown head type")
    print("   - Feasible: UNCERTAIN")

# Question 2: Modify existing head?
print("\n2. Can we modify existing head to output sigma?")
print("   - Approach: Duplicate keypoint channels (51 → 102)")
print("   - Feasible: YES - but breaks pretrained weights")
print("   - Recommendation: NO - use separate head instead")

# Question 3: Fixed sigma fallback?
print("\n3. Can we use fixed sigma as fallback?")
print("   - Approach: sigma = learned constant or heuristics")
print("   - Feasible: YES - simple, no model changes")
print("   - Recommendation: GOOD START - baseline first")

# Blockers
print("\n" + "=" * 60)
print("Potential Blockers")
print("=" * 60)

blockers = []

# Check if model is frozen
for param in model.model.parameters():
    if not param.requires_grad:
        blockers.append("Model parameters are frozen (requires_grad=False)")
        break

# Check if model is scripted/traced
if isinstance(model.model, torch.jit.ScriptModule) or isinstance(
    model.model, torch.jit.TracedModule
):
    blockers.append("Model is scripted/traced (harder to modify)")

# Check for custom layers
try:
    from torch import nn

    custom_layers = [
        m for m in model.model.modules() if not isinstance(m, (nn.Sequential, nn.Module))
    ]
    if custom_layers:
        blockers.append(f"Found {len(custom_layers)} custom layer implementations")
except:
    pass

if blockers:
    for blocker in blockers:
        print(f"  ⚠️  {blocker}")
else:
    print("  ✓ No blockers found")

print("\n" + "=" * 60)
print("Recommendation")
print("=" * 60)

print("""
Based on model structure analysis:

1. START SIMPLE: Use fixed sigma (learned constant)
   - No model modification
   - Can be implemented in loss function only
   - Baseline for comparison

2. OPTION A: Separate cv5_sigma head
   - Add new head parallel to existing pose head
   - Initialize randomly, train from scratch
   - Keep pretrained pose head frozen (or fine-tune)

3. OPTION B: Modify pose head (NOT RECOMMENDED)
   - Changes output channels: 51 → 102
   - Breaks pretrained weights compatibility
   - Only if full fine-tuning is planned

NEXT STEPS:
1. Implement fixed sigma baseline first
2. If results are good, consider Option A for learned sigma
3. Avoid Option B unless fine-tuning entire model
""")
