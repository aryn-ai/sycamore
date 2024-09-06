"""WARNING: this test needs to be run separately from other tests because if they
   import sycamore and run it, then the checks for no direct depenencies will be
   incorrect. Similarly, this test has to be run directly as python. pytest will
   pre-import sycamore."""

import inspect
import sys
from datetime import datetime


def test_00_sycamore_not_yet_imported():
    assert "sycamore" not in sys.modules, "Test run with other tests that import ray"


optional_modules = [
    "guidance",
    "neo4j",
    "ray",
    "torch",
    "duckdb",
    "elasticsearch",
    "opensearchpy",
    "pinecone",
    "weaviate",
    "easyocr",
    "pdfminer",
    "pytesseract",
    "sentence-transformers",
]


def test_disallow_unconditional_dependencies():
    assert "ray" not in sys.modules, "Test run with other tests that import ray"
    import sycamore

    _ = sycamore  # make sure tools don't remove the import
    for m in optional_modules:
        assert m not in sys.modules, f"import sycamore should not unconditionally import {m}"


def test_disallow_submodule_dependencies():
    import sycamore.reader

    _ = sycamore.reader
    for m in optional_modules:
        assert m not in sys.modules, f"import sycamore.reader should not unconditionally import {m}"

    import sycamore.writer

    _ = sycamore.writer
    for m in optional_modules:
        assert m not in sys.modules, f"import sycamore.writer should not unconditionally import {m}"


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
