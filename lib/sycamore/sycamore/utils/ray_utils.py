def check_serializable(*objects):
    from ray.util import inspect_serializability
    import io

    log = io.StringIO()
    ok, s = inspect_serializability(objects, print_file=log)
    if not ok:
        raise ValueError(f"Something isnt serializable: {s}\nLog: {log.getvalue()}")
