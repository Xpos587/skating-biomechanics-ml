#!/usr/bin/env python3
"""Generate MogaNet-B teacher heatmaps for KD training data.

Reads GT bboxes from YOLO labels, crops person, runs MogaNet-B top-down
inference, stores heatmaps in HDF5 (float16).

Heatmap Encoding:
    MogaNet-B DeconvHead outputs raw Gaussian values from JointsMSELoss training.
    Expected peak ~2-6 at keypoint centers. No sigmoid applied.

Usage:
    # Dry run: count images without inference
    python generate_teacher_heatmaps.py --dry-run \
        --data-dirs data/ap3d-fs/train data/coco-10pct/train

    # Test on 10 images
    python generate_teacher_heatmaps.py --test \
        --data-dirs data/ap3d-fs/train

    # Full run
    python generate_teacher_heatmaps.py \
        --data-dirs data/ap3d-fs/train data/coco-10pct/train \
        --output teacher_heatmaps.h5 \
        --batch-size 32

    # Resume (skip already computed)
    python generate_teacher_heatmaps.py \
        --data-dirs data/ap3d-fs/train \
        --output teacher_heatmaps.h5 \
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
# MogaNet-B model (self-contained, no mmpose dependency)
# ---------------------------------------------------------------------------
# The model definition is imported from the inference script on the server.
# We support two import strategies:
#   1. Same directory (moganet_inference.py)
#   2. Python path already set up
# ---------------------------------------------------------------------------
# --- Model components (inlined to avoid import headaches on remote GPU) ---
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


class _DeconvHeatmapHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=17):
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.final_layer(self.deconv_layers(x))


class MogaNetPose(nn.Module):
    """Full MogaNet-B pose model: backbone + deconv head.

    Input:  (N, 3, 288, 384)
    Output: (N, 17, 72, 96) heatmaps
    """

    def __init__(self):
        super().__init__()
        self.backbone = _MogaNetBackbone(
            arch="base",
            drop_path_rate=0.4,
            stem_norm_type="BN",
            conv_norm_type="BN",
            init_value=1e-5,
        )
        self.keypoint_head = _DeconvHeatmapHead(in_channels=512, out_channels=17)

    def forward(self, x):
        feats = self.backbone(x)
        return self.keypoint_head(feats[3])


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
# Data helpers
# ---------------------------------------------------------------------------

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

HEATMAP_H, HEATMAP_W = 72, 96  # MogaNet output (1/4 of 288x384)
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

        dataset_name = data_dir.parent.name  # e.g. 'ap3d-fs' from .../ap3d-fs/train
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() not in IMG_EXTENSIONS:
                continue
            lbl = lbl_dir / (f.stem + ".txt")
            if not lbl.is_file():
                continue
            entries.append((str(f.resolve()), str(lbl), dataset_name))

    return entries


def parse_yolo_label(label_path, img_w, img_h):
    """Parse YOLO pose label.

    Format: class cx cy w h kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v

    Returns (cx, cy, w, h) in pixel coordinates, or None if no valid bbox.
    For multi-person labels, returns the first person bbox (class == 0).
    """
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if cls == 0:  # person class
                cx_n, cy_n, w_n, h_n = parts[1:5]
                cx = float(cx_n) * img_w
                cy = float(cy_n) * img_h
                w = float(w_n) * img_w
                h = float(h_n) * img_h
                return cx, cy, w, h
    return None


def crop_and_resize(img_pil, cx, cy, bw, bh):
    """Crop image around bbox center, resize to MogaNet input size.

    Expands bbox by 20% padding to prevent truncating edge keypoints.

    Returns:
        crop: PIL Image resized to (MOGANET_W, MOGANET_H)
        crop_params: dict with actual crop origin and dimensions in original image pixels
    """
    img_w, img_h = img_pil.size

    # Expand bbox by 20% padding to avoid truncating edge keypoints
    pad = 0.2
    bw_pad = bw * (1 + 2 * pad)
    bh_pad = bh * (1 + 2 * pad)

    # Convert center to corner coords (use padded dimensions)
    x1 = cx - bw_pad / 2
    y1 = cy - bh_pad / 2
    x2 = cx + bw_pad / 2
    y2 = cy + bh_pad / 2

    # Clamp to image bounds
    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    x2_clamped = min(img_w, x2)
    y2_clamped = min(img_h, y2)

    crop = img_pil.crop((x1_clamped, y1_clamped, x2_clamped, y2_clamped))
    crop = crop.resize((MOGANET_W, MOGANET_H), Image.BILINEAR)

    crop_params = {
        "x1": x1_clamped,
        "y1": y1_clamped,
        "crop_w": x2_clamped - x1_clamped,
        "crop_h": y2_clamped - y1_clamped,
        "img_w": img_w,
        "img_h": img_h,
    }
    return crop, crop_params


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
    parser = argparse.ArgumentParser(description="Generate MogaNet-B teacher heatmaps for KD")
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
        "--output", type=str, default="teacher_heatmaps.h5", help="Output HDF5 path"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference (default: 128, max for RTX 5090)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Count images without inference")
    parser.add_argument("--test", action="store_true", help="Run on 10 images only")
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip already computed heatmaps"
    )
    args = parser.parse_args()

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
        print(f"\nFound {len(existing)} existing heatmaps in {args.output}")
        entries = [(img, lbl, ds) for img, lbl, ds in entries if img not in existing]
        print(f"  Remaining: {len(entries)} images to process")
        if not entries:
            print("All heatmaps already computed. Exiting.")
            sys.exit(0)

    # Load model
    print(f"\nLoading MogaNet-B from {args.model}...")
    model = MogaNetPose()
    model = load_moganet(model, args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model on {device}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Prepare HDF5
    print(f"\nWriting heatmaps to {args.output}...")
    h5_path = Path(args.output)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.skip_existing and h5_path.is_file() else "w"
    idx_map = load_existing_indices(args.output) if mode == "a" else {}
    row_offset = len(idx_map)

    with h5py.File(args.output, mode) as hf:
        if mode == "w":
            hf.create_dataset(
                "heatmaps",
                shape=(0, NUM_KEYPOINTS, HEATMAP_H, HEATMAP_W),
                maxshape=(None, NUM_KEYPOINTS, HEATMAP_H, HEATMAP_W),
                dtype="float16",
                chunks=(32, NUM_KEYPOINTS, HEATMAP_H, HEATMAP_W),
            )

        dataset = hf["heatmaps"]

        # Batched inference
        batch_imgs = []
        batch_paths = []
        batch_crop_params = []
        crop_params_map = {}
        row_idx = row_offset
        total_time = 0.0
        total_processed = 0

        pbar = tqdm(entries, desc="Generating heatmaps", unit="img")
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
            crop, crop_params = crop_and_resize(img_pil, cx, cy, bw, bh)
            img_np = np.array(crop, dtype=np.float32) / 255.0
            # HWC -> CHW
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))

            batch_imgs.append(img_tensor)
            batch_paths.append(img_path)
            batch_crop_params.append(crop_params)

            if len(batch_imgs) >= args.batch_size:
                # Run batch
                batch_tensor = torch.stack(batch_imgs).to(device)
                batch_tensor = (batch_tensor - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
                t0 = time.time()
                with torch.inference_mode():
                    heatmaps = model(batch_tensor)
                torch.cuda.synchronize()
                t1 = time.time()
                total_time += t1 - t0
                total_processed += len(batch_imgs)

                # MogaNet-B DeconvHead outputs raw Gaussian values (JointsMSELoss).
                # Do NOT apply sigmoid — raw values preserve peak structure.
                # heatmaps = torch.sigmoid(heatmaps)  # REMOVED: destroys peaks

                # Write to HDF5
                hm_np = heatmaps.cpu().numpy().astype(np.float16)
                dataset.resize(row_idx + len(batch_imgs), axis=0)
                dataset[row_idx : row_idx + len(batch_imgs)] = hm_np

                for p, cp in zip(batch_paths, batch_crop_params):
                    idx_map[p] = row_idx
                    crop_params_map[p] = cp
                    row_idx += 1

                # Throughput stats
                throughput = total_processed / total_time if total_time > 0 else 0
                pbar.set_postfix(hm=f"{throughput:.0f}/s", total=row_idx)

                batch_imgs = []
                batch_paths = []
                batch_crop_params = []

        # Flush remaining batch
        if batch_imgs:
            batch_tensor = torch.stack(batch_imgs).to(device)
            batch_tensor = (batch_tensor - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
            t0 = time.time()
            with torch.inference_mode():
                heatmaps = model(batch_tensor)
            torch.cuda.synchronize()
            t1 = time.time()
            total_time += t1 - t0
            total_processed += len(batch_imgs)

            # MogaNet-B DeconvHead outputs raw Gaussian values (JointsMSELoss).
            # Do NOT apply sigmoid — raw values preserve peak structure.
            # heatmaps = torch.sigmoid(heatmaps)  # REMOVED: destroys peaks

            hm_np = heatmaps.cpu().numpy().astype(np.float16)
            dataset.resize(row_idx + len(batch_imgs), axis=0)
            dataset[row_idx : row_idx + len(batch_imgs)] = hm_np

            for p, cp in zip(batch_paths, batch_crop_params):
                idx_map[p] = row_idx
                crop_params_map[p] = cp
                row_idx += 1

    # Write index + crop_params sidecar
    idx_path = h5_path.with_suffix(".h5.json")
    with open(idx_path, "w") as f:
        json.dump({"index": idx_map, "crop_params": crop_params_map}, f)

    # Summary
    throughput = total_processed / total_time if total_time > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"Done. {row_idx} heatmaps written to {args.output}")
    print(f"Index written to {idx_path}")
    print(f"Throughput: {throughput:.0f} heatmaps/sec")
    print(f"Total time: {total_time:.1f}s")
    print(f"HDF5 shape: ({row_idx}, {NUM_KEYPOINTS}, {HEATMAP_H}, {HEATMAP_W})")
    print("Storage dtype: float16")


if __name__ == "__main__":
    main()
