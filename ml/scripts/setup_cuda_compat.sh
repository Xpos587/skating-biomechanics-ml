#!/usr/bin/env bash
# Setup CUDA 12 compatibility libraries for onnxruntime-gpu on CUDA 13.x systems
#
# Problem: onnxruntime-gpu 1.24.x is compiled for CUDA 12, but system has CUDA 13.x
# Solution: Download standalone CUDA 12 libraries from NVIDIA and configure RUNPATH
#
# Usage: bash scripts/setup_cuda_compat.sh [venv_path]
set -euo pipefail

VENV="${1:-.venv}"
COMPAT="$VENV/cuda-compat"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== CUDA 12 Compatibility Setup ==="
echo "Venv: $VENV"
echo "Compat dir: $COMPAT"

# Check onnxruntime-gpu is installed
if ! python -c "import onnxruntime; assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()" 2>/dev/null; then
    echo "ERROR: onnxruntime-gpu not installed or CUDA not available"
    echo "Install with: uv add onnxruntime-gpu>=1.24.4"
    exit 1
fi
echo "✓ onnxruntime-gpu with CUDA support found"

# Create compat dir
mkdir -p "$COMPAT"

# Check system libraries
echo ""
echo "Checking system CUDA libraries..."
for lib in libcudnn.so.9 libcurand.so.10; do
    if ldconfig -p | grep -q "$lib"; then
        target=$(ldconfig -p | grep "$lib " | head -1 | awk '{print $NF}')
        echo "  ✓ $lib → $target"
        if [ ! -e "$COMPAT/$lib" ] || [ -L "$COMPAT/$lib" ]; then
            ln -sf "$target" "$COMPAT/$lib"
        fi
    else
        echo "  ✗ $lib NOT FOUND"
    fi
done

# Download CUDA 12 standalone libraries
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo ""
echo "Downloading CUDA 12 standalone libraries..."

# cublas + cublasLt + cudart (nvidia pip packages)
pip download --no-deps --dest "$TMPDIR/pip" nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 2>/dev/null || {
    echo "ERROR: Failed to download nvidia CUDA 12 packages"
    exit 1
}

for pkg in "$TMPDIR/pip"/*.whl; do
    unzip -o -q "$pkg" -d "$TMPDIR/extract" 2>/dev/null || true
done

for lib in libcublas.so.12 libcublasLt.so.12 libcudart.so.12; do
    src=$(find "$TMPDIR/extract/nvidia" -name "$lib" 2>/dev/null | head -1)
    if [ -n "$src" ]; then
        # Copy (not symlink) — symlinks to /tmp break after reboot
        cp -f "$src" "$COMPAT/$lib"
        echo "  ✓ $lib (nvidia-cuda12, copied)"
    else
        echo "  ✗ $lib NOT FOUND in pip packages"
    fi
done

# libcufft.so.11 (NVIDIA redist)
CUFFT_URL="https://developer.download.nvidia.com/compute/cuda/redist/libcufft/linux-x86_64/libcufft-linux-x86_64-11.2.6.59-archive.tar.xz"
CUFFT_FILE="$TMPDIR/cufft11.tar.xz"

if [ -f "$COMPAT/libcufft.so.11" ] && [ ! -L "$COMPAT/libcufft.so.11" ]; then
    echo "  ✓ libcufft.so.11 (already exists)"
else
    echo "  Downloading libcufft.so.11 from NVIDIA redist..."
    curl -sL "$CUFFT_URL" -o "$CUFFT_FILE"
    tar xf "$CUFFT_FILE" -C "$TMPDIR" 2>/dev/null
    cp "$TMPDIR"/libcufft-linux-x86_64-11.2.6.59-archive/lib/libcufft.so.11 "$COMPAT/"
    cp "$TMPDIR"/libcufft-linux-x86_64-11.2.6.59-archive/lib/libcufft.so.11.2.6.59 "$COMPAT/"
    echo "  ✓ libcufft.so.11 (nvidia redist)"
fi

# Patch RUNPATH on onnxruntime CUDA provider
echo ""
echo "Patching onnxruntime CUDA provider RUNPATH..."
ORT_PROVIDER=$(python -c "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi', 'libonnxruntime_providers_cuda.so'))" 2>/dev/null)

if [ -n "$ORT_PROVIDER" ] && [ -f "$ORT_PROVIDER" ]; then
    current_rpath=$(patchelf --print-rpath "$ORT_PROVIDER" 2>/dev/null || echo "")
    compat_abs=$(realpath "$COMPAT")
    if echo "$current_rpath" | grep -q "$compat_abs"; then
        echo "  ✓ RUNPATH already patched"
    else
        patchelf --set-rpath "$compat_abs:${current_rpath}" "$ORT_PROVIDER"
        echo "  ✓ RUNPATH patched → $compat_abs"
    fi
else
    echo "  ⚠ libonnxruntime_providers_cuda.so not found (may not be installed yet)"
fi

# Verify
echo ""
echo "=== Verification ==="
python -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'Available: {providers}')
if 'CUDAExecutionProvider' in providers:
    print('✓ CUDA READY')
else:
    print('✗ CUDA NOT AVAILABLE')
"

echo ""
echo "Done. Compat dir: $COMPAT"
echo "Contents:"
ls -la "$COMPAT/"
