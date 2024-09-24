from typing import Any


def nested_lookup(d: Any, keys: list[str]) -> Any:
    # Eventually we can support integer indexes into tuples and lists also
    while len(keys) > 0:
        if not isinstance(d, dict):
            return None
        if keys[0] not in d:
            return None
        d = d[keys[0]]
        keys = keys[1:]

    return d


def dotted_lookup(d: Any, keys: str) -> Any:
    return nested_lookup(d, keys.split("."))
