"""Smoke test for Vast.ai GPU worker image.

Runs inside the container. Traces which .so files are actually loaded
by onnxruntime, opencv, scipy, av, etc. Reports what can be safely deleted.

Usage:
    podman run --rm ghcr.io/xpos587/skating-ml-gpu:latest python3 scripts/smoke_test_container.py
    # or on a GPU machine:
    podman run --rm --gpus all ghcr.io/xpos587/skating-ml-gpu:latest python3 scripts/smoke_test_container.py
"""

from __future__ import annotations

import sys
from contextlib import suppress
from pathlib import Path


def _get_loaded_libs() -> set[str]:
    """Get all currently loaded shared libraries from /proc/self/maps."""
    libs: set[str] = set()
    with suppress(FileNotFoundError), Path("/proc/self/maps").open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6 and parts[-1].endswith(".so"):
                libs.add(parts[-1])
    return libs


def _snapshot_libs() -> set[str]:
    """Snapshot of loaded libs before tests."""
    return _get_loaded_libs()


def _new_libs(before: set[str]) -> set[str]:
    """Libs loaded since snapshot."""
    return _get_loaded_libs() - before


def test_imports():
    """Test that all required packages import correctly."""
    errors = []

    print("=== Testing imports ===")
    for module in [
        "numpy",
        "scipy",
        "cv2",
        "PIL",
        "av",
        "fastapi",
        "boto3",
        "httpx",
        "structlog",
        "pydantic_settings",
        "onnxruntime",
    ]:
        try:
            __import__(module)
            print(f"  OK  {module}")
        except ImportError as e:
            errors.append(f"{module}: {e}")
            print(f"  FAIL {module}: {e}")

    return errors


def test_onnxruntime_providers():
    """Test onnxruntime CUDA provider availability."""
    import onnxruntime as ort

    print("\n=== onnxruntime providers ===")
    providers = ort.get_available_providers()
    for p in providers:
        print(f"  {p}")

    has_cuda = "CUDAExecutionProvider" in providers
    if not has_cuda:
        print("  WARNING: No CUDA provider! Image may not work on GPU.")
    return has_cuda


def test_onnx_inference():
    """Run minimal ONNX inference to trigger CUDA .so loading."""
    import numpy as np
    import onnxruntime as ort

    print("\n=== ONNX inference test ===")
    # Create a tiny ONNX model in memory
    from onnxruntime import InferenceSession

    # Use a simple matmul via ORT API (data unused, just triggers provider init)
    np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Try to create a session with CUDA provider
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )

    # Load an actual model if available
    model_paths = [
        "/app/data/models/rvm_mobilenetv3.onnx",
        "/app/data/models/neuflowv2_mixed.onnx",
    ]

    loaded_model = None
    for path in model_paths:
        if Path(path).exists():
            print(f"  Loading model: {path}")
            try:
                sess = InferenceSession(path, providers=providers)
                print(f"  Active provider: {sess.get_providers()[0]}")
                loaded_model = path
                break
            except Exception as e:
                print(f"  Failed: {e}")

    if not loaded_model:
        print("  No ONNX models found, skipping inference test")

    return loaded_model is not None


def test_lib_loading():
    """Trace which .so files are loaded by each package."""
    print("\n=== Shared library analysis ===")

    # Baseline
    _snapshot_libs()

    # Import each heavy package and track new libs
    packages = {
        "numpy": lambda: __import__("numpy"),
        "cv2": lambda: __import__("cv2"),
        "scipy": lambda: __import__("scipy"),
        "av": lambda: __import__("av"),
        "onnxruntime": lambda: __import__("onnxruntime"),
    }

    pkg_libs: dict[str, set[str]] = {}
    for name, loader in packages.items():
        before = _snapshot_libs()
        loader()
        new = _new_libs(before)
        pkg_libs[name] = new
        print(f"\n  {name} loaded {len(new)} new .so files:")
        for lib in sorted(new):
            print(f"    {lib}")

    # All libs loaded by our packages
    all_needed = set()
    for libs in pkg_libs.values():
        all_needed.update(libs)

    return all_needed, pkg_libs


def analyze_cuda_libs(all_needed: set[str]):
    """Compare loaded CUDA libs vs available CUDA libs."""
    print("\n=== CUDA library analysis ===")

    cuda_dir = Path("/usr/local/cuda/lib64")
    if not cuda_dir.exists():
        print("  No CUDA lib64 directory found")
        return

    all_cuda_libs = set()
    for f in cuda_dir.glob("*.so*"):
        # Resolve symlinks to get the real file
        real = f.resolve()
        all_cuda_libs.add(str(real))
        all_cuda_libs.add(str(f))

    # Loaded CUDA libs
    loaded_cuda = set()
    for lib in all_needed:
        if "cuda" in lib.lower() or "nvidia" in lib.lower() or "nvcu" in lib.lower():
            loaded_cuda.add(lib)

    # Find the real .so files (not symlinks) that were NOT loaded
    unused_real_files: dict[str, int] = {}
    for f in cuda_dir.iterdir():
        if f.is_file() and f.suffix.startswith(".so"):
            real = str(f.resolve())
            if real not in all_needed and str(f) not in all_needed:
                with suppress(OSError):
                    unused_real_files[str(f)] = f.stat().st_size

    total_cuda = sum(
        f.stat().st_size for f in cuda_dir.iterdir() if f.is_file() and f.suffix.startswith(".so")
    )
    unused_total = sum(unused_real_files.values())
    used_total = total_cuda - unused_total

    print(f"  Total CUDA .so files: {total_cuda / 1024 / 1024:.0f} MB")
    print(f"  Used by onnxruntime:  {used_total / 1024 / 1024:.0f} MB")
    print(f"  Safe to delete:       {unused_total / 1024 / 1024:.0f} MB")

    if unused_real_files:
        print("\n  Unused CUDA libs (safe to delete):")
        for path, size in sorted(unused_real_files.items(), key=lambda x: -x[1]):
            name = Path(path).name
            print(f"    {size / 1024 / 1024:6.1f} MB  {name}")


def analyze_venv_libs():
    """Check venv .so files and their debug symbol sizes."""
    print("\n=== venv .so analysis ===")

    venv_libs = Path("/opt/venv/lib/python3.11/site-packages")
    if not venv_libs.exists():
        print("  No venv found")
        return

    # Find all .so files
    so_files: list[tuple[str, int]] = []
    for f in venv_libs.rglob("*.so*"):
        if f.is_file():
            so_files.append((str(f), f.stat().st_size))

    # Find .libs directories
    libs_dirs: list[tuple[str, int]] = []
    for d in venv_libs.glob("*.libs"):
        if d.is_dir():
            total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            count = sum(1 for f in d.rglob("*") if f.is_file())
            libs_dirs.append((d.name, total))
            print(f"  {d.name}: {total / 1024 / 1024:.1f} MB ({count} files)")

    total_so = sum(s for _, s in so_files)
    print(f"\n  Total .so in venv: {total_so / 1024 / 1024:.1f} MB ({len(so_files)} files)")

    # Check if strip would help
    largest = sorted(so_files, key=lambda x: -x[1])[:10]
    print("\n  Top 10 largest .so files:")
    for path, size in largest:
        print(f"    {size / 1024 / 1024:6.1f} MB  {Path(path).relative_to(venv_libs)}")


def analyze_stdlib():
    """Check removable Python stdlib modules."""
    print("\n=== Python stdlib removable modules ===")

    stdlib = Path("/usr/lib/python3.11")
    if not stdlib.exists():
        print("  No stdlib found")
        return

    removable = [
        "test",
        "idlelib",
        "tkinter",
        "turtledemo",
        "turtle.py",
        "pydoc.py",
        "pydoc_data",
        "distutils",
        "unittest",
        "asyncio",
        "email",
        "html",
        "xml",
        "xmlrpc",
        "lib2to3",
        "multiprocessing",
        "concurrent",
    ]

    total = 0
    for name in removable:
        p = stdlib / name
        if p.exists():
            if p.is_dir():
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            else:
                size = p.stat().st_size
            total += size
            print(f"  {size / 1024:.0f} KB  {name}")

    print(f"  Total removable: {total / 1024:.0f} KB")


def main():
    print("Vast.ai GPU Worker Image Smoke Test")
    print("=" * 50)

    # 1. Import test
    errors = test_imports()
    if errors:
        print(f"\n!!! {len(errors)} import failures — image is BROKEN")
        sys.exit(1)

    # 2. Provider test
    has_cuda = test_onnxruntime_providers()

    # 3. Inference test (loads CUDA libs)
    test_onnx_inference()

    # 4. Library loading analysis
    all_needed, _pkg_libs = test_lib_loading()

    # 5. CUDA analysis
    analyze_cuda_libs(all_needed)

    # 6. venv .so analysis
    analyze_venv_libs()

    # 7. stdlib analysis
    analyze_stdlib()

    print(f"\n{'=' * 50}")
    if errors:
        print("RESULT: FAIL — fix import errors above")
        sys.exit(1)
    elif not has_cuda:
        print("RESULT: WARN — no CUDA, will use CPU fallback")
        sys.exit(0)
    else:
        print("RESULT: PASS — all imports OK, CUDA available")
        sys.exit(0)


if __name__ == "__main__":
    main()
