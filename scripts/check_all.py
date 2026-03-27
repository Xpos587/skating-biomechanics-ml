#!/usr/bin/env python3
"""Run all quality checks: lint, format check, typecheck, dead code, tests."""
import subprocess
import sys


def run(cmd: list[str], name: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    success = result.returncode == 0
    status = "✓ PASSED" if success else "✗ FAILED"
    print(f"\n{status}: {name}")
    return success


def main() -> int:
    """Run all checks."""
    checks = [
        (["ruff", "check", "."], "Ruff Lint"),
        (["ruff", "format", "--check", "."], "Ruff Format Check"),
        (["mypy", "src/"], "MyPy Type Check"),
        (["vulture", "src/", "tests/", "--min-confidence", "80"], "Vulture Dead Code"),
        (["pytest", "tests/", "-v", "--cov=src/skating_biomechanics_ml"], "Pytest"),
    ]

    results = []
    for cmd, name in checks:
        results.append(run(cmd, name))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for (cmd, name), success in zip(checks, results):
        status = "✓" if success else "✗"
        print(f"{status} {name}")

    all_passed = all(results)
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
