"""E2E Test Registry – discover and run all e2e_*.py validation modules."""

import asyncio
import importlib
import sys
import time
from pathlib import Path


def run_all() -> None:
    root = Path(__file__).resolve().parent
    test_files = sorted(root.glob("e2e_*.py"))

    print("=" * 50)
    print("E2E Validation Suite")
    print("=" * 50)

    results: list[tuple[str, bool, float]] = []
    total_start = time.monotonic()

    for tf in test_files:
        module_name = tf.stem
        print(f"\n>>> Running {module_name}...")
        sys.stdout.flush()

        start = time.monotonic()
        try:
            mod = importlib.import_module(f"backend.validation.{module_name}")
            if hasattr(mod, "main"):
                asyncio.run(mod.main())
            else:
                mod.run_all()
            elapsed = time.monotonic() - start
            results.append((module_name, True, elapsed))
            print(f"<<< {module_name} PASSED ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.monotonic() - start
            results.append((module_name, False, elapsed))
            print(f"<<< {module_name} FAILED ({elapsed:.2f}s): {e}")

    total_elapsed = time.monotonic() - total_start
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)

    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)
    for name, ok, t in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}  ({t*1000:.0f}ms)")

    print(f"\n{passed}/{total} passed in {total_elapsed*1000:.0f}ms")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    run_all()
