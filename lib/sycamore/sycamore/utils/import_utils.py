import importlib
from functools import wraps
from typing import Callable, Optional, Union, TypeVar
from typing_extensions import ParamSpec  # Present in typing module for Python >= 3.10


# See https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
# for more information about how to correctly declare types for decorators so
# that they will work with mypy. https://stackoverflow.com/q/77211348 is also helpful.
P = ParamSpec("P")
T = TypeVar("T")


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
            msg += f' Please install using `pip install "sycamore-ai[{extra}]"`'
        else:
            msg += f" Please install using `pip install {','.join(missing)}`"

        raise ImportError(msg)


# Modeled on requires_dependences from
#   https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/utils.py
def requires_modules(
    modules: Union[str, list[str]], extra: Optional[str] = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            import_modules(modules, extra)
            return func(*args, **kwargs)

        return wrapper

    return decorator
