#!/usr/bin/env python3
"""Generate MogaNet-B teacher features for DWPose Two-Stage KD training.

Extracts intermediate backbone features from layers [4, 6, 8] before the final head.
These features are used for feature distillation (MSE loss) during training.

MogaNet-B Backbone Structure:
    Stage 1 (64 channels): 4 blocks
    Stage 2 (160 channels): 6 blocks
    Stage 3 (320 channels): 22 blocks
    Stage 4 (512 channels): 3 blocks

We extract from blocks [4, 6, 8] which correspond to:
    Layer 4: End of Stage 1 (64 channels, 1/4 resolution)
    Layer 6: Middle of Stage 2 (160 channels, 1/8 resolution)
    Layer 8: Middle of Stage 2 (160 channels, 1/8 resolution)

Usage:
    # Dry run: count images without inference
    python generate_teacher_features.py --dry-run \
        --data-dirs data/ap3d-fs/train data/coco-10pct/train

    # Test on 10 images (check shapes)
    python generate_teacher_features.py --test \
        --data-dirs data/ap3d-fs/train

    # Full run
    python generate_teacher_features.py \
        --data-dirs data/ap3d-fs/train data/coco-10pct/train \
        --output teacher_features.h5 \
        --batch-size 128

    # Resume (skip already computed)
    python generate_teacher_features.py \
        --data-dirs data/ap3d-fs/train \
        --output teacher_features.h5 \
        --skip-existing
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from timm.models.layers import DropPath

# ---------------------------------------------------------------------------
# MogaNet-B model (same as generate_teacher_heatmaps.py)
# ---------------------------------------------------------------------------
from torch import nn
from tqdm import tqdm


def _build_act_layer(act_type):
    if act_type is None:
        return nn.Identity()
    assert act_type in ["GELU", "ReLU", "SiLU"]
    return {"SiLU": nn.SiLU, "ReLU": nn.ReLU, "GELU": nn.GELU}[act_type]()


def _build_norm_layer(norm_type, embed_dims):
    assert norm_type in ["BN", "GN", "LN2d", "SyncBN"]
    if norm_type == "GN":
        return nn.GroupNorm(embed_dims, embed_dims, eps=1e-5)
    if norm_type == "LN2d":
        return _LayerNorm2d(embed_dims, eps=1e-6)
    if norm_type == "SyncBN":
        return nn.SyncBatchNorm(embed_dims, eps=1e-5)
    return nn.BatchNorm2d(embed_dims, eps=1e-5)


class _LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class _ElementScale(nn.Module):
    def __init__(self, embed_dims, init_value=0.0, requires_grad=True):
        super().__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class _ChannelAggregationFFN(nn.Module):
    def __init__(
        self, embed_dims, feedforward_channels, kernel_size=3, act_type="GELU", ffn_drop=0.0
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.fc1 = nn.Conv2d(embed_dims, feedforward_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(
            feedforward_channels,
            feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=feedforward_channels,
        )
        self.act = _build_act_layer(act_type)
        self.fc2 = nn.Conv2d(feedforward_channels, embed_dims, kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)
        self.decompose = nn.Conv2d(feedforward_channels, 1, kernel_size=1)
        self.sigma = _ElementScale(feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = _build_act_layer(act_type)

    def feat_decompose(self, x):
        return x + self.sigma(x - self.decompose_act(self.decompose(x)))

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.feat_decompose(x)
        x = self.fc2(x)
        return self.drop(x)


class _MultiOrderDWConv(nn.Module):
    def __init__(self, embed_dims, dw_dilation=[1, 2, 3], channel_split=[1, 3, 4]):
        super().__init__()
        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        self.DW_conv0 = nn.Conv2d(
            embed_dims,
            embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=embed_dims,
            stride=1,
            dilation=dw_dilation[0],
        )
        self.DW_conv1 = nn.Conv2d(
            self.embed_dims_1,
            self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1,
            dilation=dw_dilation[1],
        )
        self.DW_conv2 = nn.Conv2d(
            self.embed_dims_2,
            self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1,
            dilation=dw_dilation[2],
        )
        self.PW_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(x_0[:, self.embed_dims_0 : self.embed_dims_0 + self.embed_dims_1, ...])
        x_2 = self.DW_conv2(x_0[:, self.embed_dims - self.embed_dims_2 :, ...])
        x = torch.cat([x_0[:, : self.embed_dims_0, ...], x_1, x_2], dim=1)
        return self.PW_conv(x)


class _MultiOrderGatedAggregation(nn.Module):
    def __init__(
        self,
        embed_dims,
        attn_dw_dilation=[1, 2, 3],
        attn_channel_split=[1, 3, 4],
        attn_act_type="SiLU",
        attn_force_fp32=False,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.value = _MultiOrderDWConv(
            embed_dims, dw_dilation=attn_dw_dilation, channel_split=attn_channel_split
        )
        self.proj_2 = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.act_value = _build_act_layer(attn_act_type)
        self.act_gate = _build_act_layer(attn_act_type)
        self.sigma = _ElementScale(embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type="cuda", enabled=False):
            return self.proj_2(
                self.act_gate(g.to(torch.float32)) * self.act_gate(v.to(torch.float32))
            )

    def forward(self, x):
        shortcut = x.clone()
        x = self.feat_decompose(x)
        x = self.act_value(x)
        g = self.gate(x)
        v = self.value(x)
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        return x + shortcut


class _MogaBlock(nn.Module):
    def __init__(
        self,
        embed_dims,
        ffn_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        act_type="GELU",
        norm_type="BN",
        init_value=1e-5,
        attn_dw_dilation=[1, 2, 3],
        attn_channel_split=[1, 3, 4],
        attn_act_type="SiLU",
        attn_force_fp32=False,
    ):
        super().__init__()
        self.out_channels = embed_dims
        self.norm1 = _build_norm_layer(norm_type, embed_dims)
        self.attn = _MultiOrderGatedAggregation(
            embed_dims, attn_dw_dilation, attn_channel_split, attn_act_type, attn_force_fp32
        )
        self.drop_path = nn.Identity() if drop_path_rate <= 0 else DropPath(drop_path_rate)
        self.norm2 = _build_norm_layer(norm_type, embed_dims)
        self.mlp = _ChannelAggregationFFN(
            embed_dims=embed_dims,
            feedforward_channels=int(embed_dims * ffn_ratio),
            act_type=act_type,
            ffn_drop=drop_rate,
        )
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True
        )

    def forward(self, x):
        identity = x
        x = identity + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        identity = x
        x = identity + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class _ConvPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dims, kernel_size=3, stride=2, norm_type="BN"):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = _build_norm_layer(norm_type, embed_dims)

    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        return x, (x.shape[2], x.shape[3])


class _StackConvPatchEmbed(nn.Module):
    def __init__(
        self, in_channels, embed_dims, kernel_size=3, stride=2, act_type="GELU", norm_type="BN"
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                embed_dims // 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
            _build_norm_layer(norm_type, embed_dims // 2),
            _build_act_layer(act_type),
            nn.Conv2d(
                embed_dims // 2,
                embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
            _build_norm_layer(norm_type, embed_dims),
        )

    def forward(self, x):
        x = self.projection(x)
        return x, (x.shape[2], x.shape[3])


class _MogaNetBackbone(nn.Module):
    """MogaNet backbone with feature extraction hooks."""

    arch_zoo = {
        **dict.fromkeys(
            ["b", "base"],
            {
                "embed_dims": [64, 160, 320, 512],
                "depths": [4, 6, 22, 3],
                "ffn_ratios": [8, 8, 4, 4],
            },
        ),
    }

    def __init__(
        self,
        arch="base",
        in_channels=3,
        drop_rate=0.0,
        drop_path_rate=0.0,
        init_value=1e-5,
        stem_norm_type="BN",
        conv_norm_type="BN",
        patch_sizes=[3, 3, 3, 3],
        patchembed_types=["ConvEmbed", "Conv", "Conv", "Conv"],
        attn_dw_dilation=[1, 2, 3],
        attn_channel_split=[1, 3, 4],
        attn_act_type="SiLU",
        attn_final_dilation=True,
        attn_force_fp32=False,
    ):
        super().__init__()
        self.arch_settings = self.arch_zoo[arch]
        self.embed_dims = self.arch_settings["embed_dims"]
        self.depths = self.arch_settings["depths"]
        self.ffn_ratios = self.arch_settings["ffn_ratios"]
        self.num_stages = len(self.depths)
        self.attn_force_fp32 = attn_force_fp32
        self.use_layer_norm = stem_norm_type == "LN"

        total_depth = sum(self.depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        cur_block_idx = 0
        for i, depth in enumerate(self.depths):
            if i == 0 and patchembed_types[i] == "ConvEmbed":
                patch_embed = _StackConvPatchEmbed(
                    in_channels=in_channels,
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    act_type="GELU",
                    norm_type=conv_norm_type,
                )
            else:
                patch_embed = _ConvPatchEmbed(
                    in_channels=in_channels if i == 0 else self.embed_dims[i - 1],
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    norm_type=conv_norm_type,
                )

            dil = (
                attn_dw_dilation if (i != self.num_stages - 1 or attn_final_dilation) else [1, 2, 1]
            )

            blocks = nn.ModuleList(
                [
                    _MogaBlock(
                        embed_dims=self.embed_dims[i],
                        ffn_ratio=self.ffn_ratios[i],
                        drop_rate=drop_rate,
                        drop_path_rate=dpr[cur_block_idx + j],
                        norm_type=conv_norm_type,
                        init_value=init_value,
                        attn_dw_dilation=dil,
                        attn_channel_split=attn_channel_split,
                        attn_act_type=attn_act_type,
                        attn_force_fp32=attn_force_fp32,
                    )
                    for j in range(depth)
                ]
            )
            cur_block_idx += depth
            norm = _build_norm_layer(stem_norm_type, self.embed_dims[i])

            self.add_module(f"patch_embed{i + 1}", patch_embed)
            self.add_module(f"blocks{i + 1}", blocks)
            self.add_module(f"norm{i + 1}", norm)

        # Global block index for feature extraction
        self._block_index_map = {}
        global_idx = 0
        for i in range(self.num_stages):
            blocks = getattr(self, f"blocks{i + 1}")
            for j, block in enumerate(blocks):
                self._block_index_map[global_idx] = (i, j, block)
                global_idx += 1

    def forward(self, x):
        outs = []
        for i in range(self.num_stages):
            x, hw_shape = getattr(self, f"patch_embed{i + 1}")(x)
            for block in getattr(self, f"blocks{i + 1}"):
                x = block(x)
            norm = getattr(self, f"norm{i + 1}")
            if self.use_layer_norm:
                x = x.flatten(2).transpose(1, 2)
                x = norm(x)
                x = (
                    x.reshape(-1, *hw_shape, getattr(self, f"blocks{i + 1}").out_channels)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
            else:
                x = norm(x)
            outs.append(x)
        return outs

    def get_block(self, global_idx):
        """Get block by global index (0-indexed across all stages)."""
        if global_idx not in self._block_index_map:
            raise ValueError(
                f"Block index {global_idx} out of range (max {len(self._block_index_map) - 1})"
            )
        stage_idx, block_idx, block = self._block_index_map[global_idx]
        return stage_idx, block_idx, block


class FeatureExtractorMogaNet(nn.Module):
    """MogaNet-B with feature extraction hooks at specified layers."""

    def __init__(self, backbone, extract_layers=[3, 9, 31]):
        super().__init__()
        self.backbone = backbone
        self.extract_layers = extract_layers
        self.features = {}

        # Register forward hooks
        self.hooks = []
        for layer_idx in extract_layers:
            stage_idx, block_idx, block = self.backbone.get_block(layer_idx)

            def make_hook(idx):
                def hook(module, input, output):
                    self.features[idx] = output.detach()

                return hook

            handle = block.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(handle)

    def forward(self, x):
        self.features.clear()
        _ = self.backbone(x)

        # Return features in layer order
        return [self.features[idx] for idx in self.extract_layers]

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()


def load_moganet(model, ckpt_path):
    """Load weights from MMPose checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    model_sd = model.state_dict()
    loaded = {k: v for k, v in sd.items() if k in model_sd}
    model.load_state_dict(loaded, strict=False)
    missing = [k for k in model_sd if k not in loaded]
    if missing:
        print(f"WARNING: {len(missing)} missing keys: {missing[:5]}...")
    print(f"Loaded {len(loaded)}/{len(model_sd)} parameters from {ckpt_path}")
    return model


# ---------------------------------------------------------------------------
# Data helpers (same as generate_teacher_heatmaps.py)
# ---------------------------------------------------------------------------

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

HEATMAP_H, HEATMAP_W = 72, 96
MOGANET_H, MOGANET_W = 288, 384
NUM_KEYPOINTS = 17


def discover_images(data_dirs):
    """Find all images with matching label files across data directories.

    Returns list of (image_path, label_path, dataset_name) tuples.
    """
    entries = []
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        img_dir = data_dir / "images"
        lbl_dir = data_dir / "labels"
        if not img_dir.is_dir():
            print(f"WARNING: {img_dir} does not exist, skipping")
            continue

        dataset_name = data_dir.parent.name
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() not in IMG_EXTENSIONS:
                continue
            lbl = lbl_dir / (f.stem + ".txt")
            if not lbl.is_file():
                continue
            entries.append((str(f), str(lbl), dataset_name))

    return entries


def parse_yolo_label(label_path, img_w, img_h):
    """Parse YOLO pose label.

    Format: class cx cy w h kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v

    Returns (cx, cy, w, h) in pixel coordinates, or None if no valid bbox.
    """
    with open(label_path) as f:
        line = f.readline().strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) < 5:
        return None

    cls, cx_n, cy_n, w_n, h_n = parts[:5]
    cx = float(cx_n) * img_w
    cy = float(cy_n) * img_h
    w = float(w_n) * img_w
    h = float(h_n) * img_h
    return cx, cy, w, h


def crop_and_resize(img_pil, cx, cy, bw, bh):
    """Crop image around bbox center, resize to MogaNet input size.

    Expands bbox by 20% padding to prevent truncating edge keypoints.
    """
    img_w, img_h = img_pil.size

    # Expand bbox by 20% padding
    pad = 0.2
    bw_pad = bw * (1 + 2 * pad)
    bh_pad = bh * (1 + 2 * pad)

    # Convert center to corner coords
    x1 = cx - bw_pad / 2
    y1 = cy - bh_pad / 2
    x2 = cx + bw_pad / 2
    y2 = cy + bh_pad / 2

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    crop = img_pil.crop((x1, y1, x2, y2))
    crop = crop.resize((MOGANET_W, MOGANET_H), Image.BILINEAR)
    return crop


def load_existing_indices(output_path):
    """Load set of image paths already computed in HDF5."""
    idx_path = Path(output_path).with_suffix(".h5.json")
    if idx_path.is_file():
        with open(idx_path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate MogaNet-B teacher features for DWPose KD"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/root/data/datasets/raw/athletepose3d/model_params/moganet_b_ap2d_384x288.pth",
        help="Path to MogaNet-B weights",
    )
    parser.add_argument(
        "--data-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Training data directories (YOLO format: images/ + labels/)",
    )
    parser.add_argument(
        "--output", type=str, default="teacher_features.h5", help="Output HDF5 path"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference (default: 128, max for RTX 5090)",
    )
    parser.add_argument(
        "--extract-layers",
        type=str,
        default="3,9,31",
        help="Comma-separated layer indices to extract (default: 3,9,31)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Count images without inference")
    parser.add_argument("--test", action="store_true", help="Run on 10 images only (check shapes)")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip already computed features"
    )
    args = parser.parse_args()

    # Parse layer indices
    extract_layers = [int(x) for x in args.extract_layers.split(",")]
    print(f"Extracting features from backbone layers: {extract_layers}")

    # Discover images
    entries = discover_images(args.data_dirs)
    print(f"Found {len(entries)} images with labels across {len(args.data_dirs)} data dirs")

    if not entries:
        print("No images found. Exiting.")
        sys.exit(1)

    # Count by dataset
    from collections import Counter

    ds_counts = Counter(name for _, _, name in entries)
    for ds, count in sorted(ds_counts.items()):
        print(f"  {ds}: {count} images")

    if args.test:
        entries = entries[:10]
        print(f"\n[--test mode: using first {len(entries)} images]")

    if args.dry_run:
        print(f"\nDRY RUN: {len(entries)} images would be processed")
        sys.exit(0)

    # Check for existing
    existing = load_existing_indices(args.output) if args.skip_existing else {}
    if existing:
        print(f"\nFound {len(existing)} existing features in {args.output}")
        entries = [(img, lbl, ds) for img, lbl, ds in entries if img not in existing]
        print(f"  Remaining: {len(entries)} images to process")
        if not entries:
            print("All features already computed. Exiting.")
            sys.exit(0)

    # Load model
    print(f"\nLoading MogaNet-B from {args.model}...")
    backbone = _MogaNetBackbone(
        arch="base",
        drop_path_rate=0.4,
        stem_norm_type="BN",
        conv_norm_type="BN",
        init_value=1e-5,
    )
    backbone = load_moganet(backbone, args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    backbone.eval()

    # Wrap with feature extractor
    model = FeatureExtractorMogaNet(backbone, extract_layers=extract_layers)
    model = model.to(device)
    model.eval()

    print(f"Model on {device}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Test forward pass to get shapes
    print("\nRunning test forward pass to determine feature shapes...")
    test_input = torch.randn(1, 3, MOGANET_H, MOGANET_W).to(device)
    with torch.inference_mode():
        test_features = model(test_input)

    print("Feature shapes:")
    layer_shapes = []
    for i, (layer_idx, feat) in enumerate(zip(extract_layers, test_features)):
        shape = tuple(feat.shape)
        layer_shapes.append(shape)
        print(f"  Layer {layer_idx}: {shape} (C={shape[1]}, H={shape[2]}, W={shape[3]})")

    # Prepare HDF5
    print(f"\nWriting features to {args.output}...")
    h5_path = Path(args.output)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.skip_existing and h5_path.is_file() else "w"
    idx_map = load_existing_indices(args.output) if mode == "a" else {}
    row_offset = len(idx_map)

    with h5py.File(args.output, mode) as hf:
        if mode == "w":
            # Create datasets for each layer
            for layer_idx, shape in zip(extract_layers, layer_shapes):
                # Shape: (N, C, H, W)
                layer_name = f"layer{layer_idx}"
                hf.create_dataset(
                    layer_name,
                    shape=(0, shape[1], shape[2], shape[3]),
                    maxshape=(None, shape[1], shape[2], shape[3]),
                    dtype="float16",
                    chunks=(32, shape[1], shape[2], shape[3]),
                )

        # Batched inference
        batch_imgs = []
        batch_paths = []
        row_idx = row_offset
        total_time = 0.0
        total_processed = 0

        pbar = tqdm(entries, desc="Generating features", unit="img")
        for img_path, label_path, dataset_name in pbar:
            # Load image
            try:
                img_pil = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"\nWARNING: Cannot open {img_path}: {e}, skipping")
                continue

            img_w, img_h = img_pil.size

            # Parse label
            bbox = parse_yolo_label(label_path, img_w, img_h)
            if bbox is None:
                print(f"\nWARNING: Empty/invalid label {label_path}, skipping")
                continue

            cx, cy, bw, bh = bbox

            # Crop and resize
            crop = crop_and_resize(img_pil, cx, cy, bw, bh)
            img_np = np.array(crop, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))

            batch_imgs.append(img_tensor)
            batch_paths.append(img_path)

            if len(batch_imgs) >= args.batch_size:
                # Run batch
                batch_tensor = torch.stack(batch_imgs).to(device)
                batch_tensor = (batch_tensor - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
                t0 = time.time()
                with torch.inference_mode():
                    features = model(batch_tensor)
                torch.cuda.synchronize()
                t1 = time.time()
                total_time += t1 - t0
                total_processed += len(batch_imgs)

                # Write features to HDF5
                for layer_idx, feat, shape in zip(extract_layers, features, layer_shapes):
                    layer_name = f"layer{layer_idx}"
                    dataset = hf[layer_name]

                    # Convert to float16 and write
                    feat_np = feat.cpu().numpy().astype(np.float16)
                    dataset.resize(row_idx + len(batch_imgs), axis=0)
                    dataset[row_idx : row_idx + len(batch_imgs)] = feat_np

                # Update index map
                for p in batch_paths:
                    idx_map[p] = row_idx
                    row_idx += 1

                # Throughput stats
                throughput = total_processed / total_time if total_time > 0 else 0
                pbar.set_postfix(feat=f"{throughput:.0f}/s", total=row_idx)

                batch_imgs = []
                batch_paths = []

        # Flush remaining batch
        if batch_imgs:
            batch_tensor = torch.stack(batch_imgs).to(device)
            batch_tensor = (batch_tensor - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
            t0 = time.time()
            with torch.inference_mode():
                features = model(batch_tensor)
            torch.cuda.synchronize()
            t1 = time.time()
            total_time += t1 - t0
            total_processed += len(batch_imgs)

            # Write features to HDF5
            for layer_idx, feat, shape in zip(extract_layers, features, layer_shapes):
                layer_name = f"layer{layer_idx}"
                dataset = hf[layer_name]

                feat_np = feat.cpu().numpy().astype(np.float16)
                dataset.resize(row_idx + len(batch_imgs), axis=0)
                dataset[row_idx : row_idx + len(batch_imgs)] = feat_np

            for p in batch_paths:
                idx_map[p] = row_idx
                row_idx += 1

    # Cleanup hooks
    model.remove_hooks()

    # Write index sidecar + HDF5 attrs
    idx_path = h5_path.with_suffix(".h5.json")
    with open(idx_path, "w") as f:
        json.dump(idx_map, f)

    # Also store index in HDF5 attrs for TeacherFeatureLoader
    with h5py.File(str(args.output), "a") as hf:
        hf.attrs["indices"] = json.dumps(idx_map)

    # Summary
    throughput = total_processed / total_time if total_time > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"Done. {row_idx} feature sets written to {args.output}")
    print(f"Index written to {idx_path}")
    print(f"Throughput: {throughput:.0f} images/sec")
    print(f"Total time: {total_time:.1f}s")

    # Calculate storage
    total_storage = 0
    for layer_idx, shape in zip(extract_layers, layer_shapes):
        layer_size = row_idx * shape[1] * shape[2] * shape[3] * 2  # float16 = 2 bytes
        total_storage += layer_size
        print(f"  Layer {layer_idx}: {row_idx} x {shape[1:]} = {layer_size / 1e9:.2f} GB")

    print(f"Total storage: {total_storage / 1e9:.2f} GB")
    print("Storage dtype: float16")


if __name__ == "__main__":
    main()
