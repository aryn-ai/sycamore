import importlib
from functools import wraps
from typing import Optional, Union


def import_modules(modules: Union[str, list[str]], extra: Optional[str] = None):
    missing = []

    if isinstance(modules, str):
        modules = [modules]

    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)

    if len(missing) > 0:
        msg = f"Unable to locate modules: {missing}."
        if extra is not None:
            msg += f" Please install using `pip install sycamore-ai[{extra}]`"
        else:
            msg += f" Please install using `pip install {','.join(missing)}`"

        raise ImportError(msg)


# Modeled on requires_dependences from
#   https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/utils.py
def requires_modules(modules: Union[str, list[str]], extra: Optional[str] = None):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import_modules(modules, extra)
            return func(*args, **kwargs)

        return wrapper

    return decorator
