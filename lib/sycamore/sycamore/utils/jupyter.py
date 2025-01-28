from typing import Optional


def slow_pprint(
    v: object, max_bytes: Optional[int] = None, width: int = 120, chunk_size: int = 1000, delay: float = 0.25
) -> None:
    """
    Prints large outputs slowly to prevent Jupyter from dropping output.

    Args:
        v: Value to be pretty-printed.
        max_bytes: Maximum number of bytes to display, None for no limit.
        width: Width for pretty formatting.
        chunk_size: Number of characters to print at a time.
        delay: Time (in seconds) to wait between chunks.
    """
    from devtools import PrettyFormat
    import time

    s = PrettyFormat(width=width)(v)
    if max_bytes is not None:
        s = s[:max_bytes]

    for i in range(0, len(s), chunk_size):
        print(s[i : i + chunk_size], end="", flush=True)
        time.sleep(delay)
    if not s.endswith("\n"):
        print("", flush=True)


def bound_memory(gb: int = 4) -> None:
    """
    Limits the process's memory usage.

    Args:
        gb: Memory limit in gigabytes.
    """
    import resource
    import platform

    if platform.system() != "Linux":
        print("WARNING: Memory limiting only works on Linux.")

    limit_bytes: int = gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_DATA, (limit_bytes, resource.RLIM_INFINITY))


def reimport(module_name: str) -> None:
    """
    Dynamically reloads a module.

    Args:
        module_name: Name of the module as a string.
    """
    import importlib

    try:
        module = importlib.import_module(module_name)
        importlib.reload(module)
        print(f"Warning: You cneed to re-execute any statements like: `from {module_name} import ...`")

    except ModuleNotFoundError:
        print(f"Error: Module '{module_name}' not found.")

    except Exception as e:
        print(f"Error reloading module '{module_name}': {e}")
