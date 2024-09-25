from typing import Any


def nested_lookup(d: Any, keys: list[str]) -> Any:
    # Eventually we can support integer indexes into tuples and lists also
    while len(keys) > 0:
        if d is None:
            return None
        try:
            d = d.get(keys[0])
        except AttributeError:
            return None

        keys = keys[1:]

    return d


def dotted_lookup(d: Any, keys: str) -> Any:
    return nested_lookup(d, keys.split("."))
