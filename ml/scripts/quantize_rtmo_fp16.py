"""Convert RTMO ONNX model to FP16 for GPU inference.

Reduces model size by ~50% and enables faster inference on Ampere tensor cores.
Input/output tensors remain FP32 for compatibility with onnxruntime.

Usage:
    cd ml && uv run python scripts/quantize_rtmo_fp16.py          # balanced (default)
    cd ml && uv run python scripts/quantize_rtmo_fp16.py lightweight
    cd ml && uv run python scripts/quantize_rtmo_fp16.py performance
    cd ml && uv run python scripts/quantize_rtmo_fp16.py /path/to/model.onnx  # custom path
"""

from __future__ import annotations

import sys
from pathlib import Path


def quantize(input_path: str, output_path: str) -> None:
    """Convert ONNX FP32 model to FP16 (keep IO in FP32).

    Args:
        input_path: Path to source FP32 ONNX model.
        output_path: Path to write FP16 ONNX model.
    """
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as exc:
        msg = (
            f"{exc}\n\n"
            "onnxconverter-common is required for FP16 quantization.\n"
            "Install with: uv add --optional onnxconverter-common"
        )
        raise SystemExit(msg) from exc

    model = onnx.load(input_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, output_path)

    orig_size = Path(input_path).stat().st_size
    new_size = Path(output_path).stat().st_size
    ratio = new_size / orig_size * 100

    print(f"Converted: {input_path} -> {output_path}")
    print(f"Size: {orig_size / 1e6:.1f} MB -> {new_size / 1e6:.1f} MB ({ratio:.0f}%)")


if __name__ == "__main__":
    from src.pose_estimation.rtmo_batch import RTMO_MODELS

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in RTMO_MODELS:
            input_model = RTMO_MODELS[arg]
        else:
            # Treat as custom path
            input_model = arg
    else:
        input_model = RTMO_MODELS["balanced"]

    if not Path(input_model).exists():
        raise SystemExit(f"Model not found: {input_model}")

    output_model = str(input_model).replace(".onnx", "-fp16.onnx")
    quantize(input_model, output_model)
