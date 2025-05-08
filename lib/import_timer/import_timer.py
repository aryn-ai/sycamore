"""
import_timer.py - Automatically instrument Python imports to measure timing

This script patches Python's import system to measure the time each import takes.
It can be used in two ways:
1. As a module to be imported at the start of your script
2. As a command line tool that runs your script with instrumentation

Usage as module:
    import import_timer
    # All subsequent imports will be timed
    import numpy as np  # This import will be timed
    import_timer.show()

Usage from command line:
    python import_timer.py your_script.py [args...]
"""

import sys
import time
import os
import builtins

original_import = builtins.__import__

import_times: dict[str, float] = {}
parent_imports: dict[str, str] = {}
import_counts: dict[str, int] = {}
import_stack: list[str] = []
import_depths: dict[str, int] = {}


def timed_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    A replacement for the built-in __import__ function that measures execution time.
    """
    parent = import_stack[-1] if import_stack else None

    # record the first parent on import
    if parent and name not in parent_imports:
        parent_imports[name] = parent

    depth = len(import_stack)
    import_depths[name] = min(depth, import_depths.get(name, float("inf")))

    import_stack.append(name)

    start_time = time.time()
    try:
        result = original_import(name, globals, locals, fromlist, level)
    finally:
        end_time = time.time()
        elapsed = end_time - start_time

        if name in import_times:
            # Re-imports can be really slow; not sure why, but sycamore.llms.llm can be 0.19s -> 0.62s
            import_times[name] += elapsed
            import_counts[name] += 1
        else:
            import_times[name] = elapsed
            import_counts[name] = 1

        if import_stack:
            import_stack.pop()

    return result


# TODO: figure out a tree display; that would be nicer to figure out what to fix
def show(sort_by="time", min_time=None):
    """
    Print the import timing results.

    Args:
        sort_by: 'time' (default) or 'name' to sort the output
        min_time: Minimum time threshold to include in the report (in seconds)
                  Defaults to 5% of the longest import if unspecified.
    """
    print("\n=== Import Timing Results ===")

    if min_time is None:
        max_time = max([v for _, v in import_times.items()])
        min_time = max_time * 0.05
    filtered_times = {k: v for k, v in import_times.items() if v >= min_time}

    if sort_by == "time":
        sorted_items = sorted(filtered_times.items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_items = sorted(filtered_times.items(), key=lambda x: x[0])

    if not sorted_items:
        print("No imports took longer than the minimum threshold of {:.3f}s".format(min_time))
        return

    max_name_len = max(len(name) for name in filtered_times.keys())

    print(f"{'Module':<{max_name_len+2}} {'Time (s)':>10} {'Count':>5} {'Depth':>5} {'Parent'}")
    print("-" * (max_name_len + 30))

    for name, elapsed in sorted_items:
        depth = import_depths[name]
        parent = parent_imports.get(name, "")
        count = import_counts[name]

        print(f"{name:<{max_name_len+2}} {elapsed:10.6f} {count:>5} {depth:>5} {parent}")


def initialize():
    """Install the import timer by replacing the built-in __import__ function."""
    builtins.__import__ = timed_import
    import_depths.clear()
    import_times.clear()
    parent_imports.clear()
    import_stack.clear()
    # Add root value so it isn't empty
    import_stack.append("__main__")


def run_script(script_path, args=None):
    """
    Run a Python script with import timing instrumentation.

    Args:
        script_path: Path to the Python script to run
        args: Optional list of command line arguments for the script
    """
    assert os.path.exists(script_path), f'Error: Script "{script_path}" not found'

    if args:
        sys.argv = [script_path] + args
    else:
        sys.argv = [script_path]

    initialize()

    script_globals = {
        "__file__": script_path,
        "__name__": "__main__",
        "__package__": None,
        "__cached__": None,
    }

    with open(script_path, "rb") as file:
        code = compile(file.read(), script_path, "exec")
        exec(code, script_globals)

    show()

    return 0


# Enable timing when module is imported
initialize()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_timer.py your_script.py [args...]")
        sys.exit(1)

    script_path = sys.argv[1]
    script_args = sys.argv[2:]
    sys.exit(run_script(script_path, script_args))
