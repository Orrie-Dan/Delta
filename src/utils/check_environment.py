"""Environment verification for Uniform Detection MVP-00 (soldier uniform recognition)."""

from __future__ import annotations

import sys


def check_environment() -> bool:
    """Verify core dependencies and print a PASS/FAIL summary.

    Returns:
        True if all required checks pass, False otherwise.
    """
    results: list[tuple[str, bool, str]] = []

    # 1. Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 11)
    results.append(
        (
            "Python >= 3.11",
            py_ok,
            f"Python {py_version}" if py_ok else f"Found Python {py_version}; need 3.11+",
        )
    )
    print(f"[1] Python version: {py_version}")

    # 2–4. PyTorch + CUDA
    try:
        import torch

        torch_version = torch.__version__
        print(f"[2] PyTorch version: {torch_version}")
        results.append(("PyTorch import", True, f"torch {torch_version}"))

        cuda_available = torch.cuda.is_available()
        print(f"[3] CUDA available: {cuda_available}")
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[4] GPU name: {gpu_name}")
                results.append(("CUDA / GPU", True, gpu_name))
            except Exception as exc:  # noqa: BLE001
                print(f"[4] GPU name: unavailable ({exc})")
                results.append(("CUDA / GPU", True, "CUDA reported available; GPU name failed"))
        else:
            print("[4] GPU name: N/A (CPU-only)")
            results.append(("CUDA / GPU", True, "CUDA not available (CPU-only is OK)"))
    except ImportError as exc:
        print(f"[2] PyTorch version: FAILED — {exc}")
        print("[3] CUDA available: FAILED — PyTorch not installed")
        print("[4] GPU name: FAILED — PyTorch not installed")
        results.append(("PyTorch import", False, str(exc)))
        results.append(("CUDA / GPU", False, "Skipped (PyTorch missing)"))

    # 5. OpenCV
    try:
        import cv2

        cv_version = cv2.__version__
        print(f"[5] OpenCV version: {cv_version}")
        results.append(("OpenCV import", True, f"cv2 {cv_version}"))
    except ImportError as exc:
        print(f"[5] OpenCV version: FAILED — {exc}")
        results.append(("OpenCV import", False, str(exc)))

    # 6. Ultralytics
    try:
        import ultralytics

        ultra_version = getattr(ultralytics, "__version__", "unknown")
        print(f"[6] Ultralytics import: OK (version {ultra_version})")
        results.append(("Ultralytics import", True, f"ultralytics {ultra_version}"))
    except ImportError as exc:
        print(f"[6] Ultralytics import: FAILED — {exc}")
        results.append(("Ultralytics import", False, str(exc)))

    # 7. Torchvision
    try:
        import torchvision

        tv_version = torchvision.__version__
        print(f"[7] Torchvision import: OK (version {tv_version})")
        results.append(("Torchvision import", True, f"torchvision {tv_version}"))
    except ImportError as exc:
        print(f"[7] Torchvision import: FAILED — {exc}")
        results.append(("Torchvision import", False, str(exc)))

    # Required checks (CUDA absence is not a failure)
    required_names = {
        "Python >= 3.11",
        "PyTorch import",
        "OpenCV import",
        "Ultralytics import",
        "Torchvision import",
    }
    failed = [name for name, ok, _ in results if name in required_names and not ok]
    passed = len(failed) == 0

    print()
    print("=" * 50)
    if passed:
        print("RESULT: PASS — environment is ready for MVP-00")
    else:
        print("RESULT: FAIL — fix the issues below, then re-run:")
        for name, ok, detail in results:
            if name in required_names and not ok:
                print(f"  - {name}: {detail}")
    print("=" * 50)

    return passed


def main() -> int:
    ok = check_environment()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
