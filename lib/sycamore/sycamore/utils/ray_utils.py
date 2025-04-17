import functools


def check_serializable(*objects):
    from ray.util import inspect_serializability
    import io

    log = io.StringIO()
    ok, s = inspect_serializability(objects, print_file=log)
    if not ok:
        raise ValueError(f"Something isn't serializable: {s}\nLog: {log.getvalue()}")


def handle_serialization_exception(*objects):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except TypeError:
                attrs = [getattr(self, attr) for attr in objects]
                check_serializable(*tuple(attrs))

        return wrapper

    return decorator
