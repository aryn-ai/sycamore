"""WARNING: this test needs to be run separately from other tests because if they
   import sycamore and run it, then the checks for no direct depenencies will be
   incorrect. Similarly, this test has to be run directly as python. pytest will
   pre-import sycamore."""

import inspect
import sys
from datetime import datetime


def test_00_sycamore_not_yet_imported():
    assert "sycamore" not in sys.modules, "Test run with other tests that import ray"


def test_disallow_unconditional_dependencies():
    assert "ray" not in sys.modules, "Test run with other tests that import ray"
    import sycamore

    _ = sycamore  # make sure tools don't remove the import
    assert "torch" not in sys.modules, "import sycamore should not unconditionally import torch"
    assert "ray" not in sys.modules, "import sycamore should not unconditionally import ray"
    assert "guidance" not in sys.modules, "import sycamore should not unconditionally import guidance"


if __name__ == "__main__":
    all_start = datetime.now()
    tests = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if name.startswith("test") and inspect.isfunction(obj):
            tests.append((name, obj))

    tests.sort(key=lambda a: a[0])
    for t in tests:
        (name, obj) = t
        print(f"Testing {name}")
        start = datetime.now()
        obj()
        print(f"Testing {name} took {datetime.now() - start}")

    print(f"All tests ran in {datetime.now() - all_start}")
