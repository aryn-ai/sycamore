from functools import wraps
from typing import Optional, Callable, TypeVar
from typing_extensions import ParamSpec
import warnings

P = ParamSpec("P")
T = TypeVar("T")


def deprecated(version: Optional[str] = None, reason: Optional[str] = None):

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        warn_msg = f"{fn.__name__} is deprecated"
        if version is not None:
            warn_msg += f" since version {version}"
        if reason is not None:
            warn_msg += f". Reason: {reason}"

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            warnings.warn(warn_msg, category=FutureWarning)
            return fn(*args, **kwargs)

        return wrapper

    return decorator
