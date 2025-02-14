import warnings


def experimental(cls):
    """
    Decorator to mark a class as experimental.
    """

    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Class {cls.__name__} is experimental and may change in the future.", FutureWarning, stacklevel=2
        )
        return cls(*args, **kwargs)

    return wrapper
