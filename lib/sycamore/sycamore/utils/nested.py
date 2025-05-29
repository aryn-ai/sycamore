from typing import Any


def nested_lookup(d: Any, keys: list[str]) -> Any:
    # Eventually we can support integer indexes into tuples and lists also
    while len(keys) > 0:
        if d is None:
            return None
        try:
            if isinstance(keys[0], str) and hasattr(d, keys[0]):
                # This is necessary to handle attributes with a property
                # getter that returns something different than what's in the
                # underlying dict. For example the text_representation for
                # a TableElement.
                d = getattr(d, keys[0])
            else:
                d = d.get(keys[0])
        except (AttributeError, ValueError):
            return None

        keys = keys[1:]

    return d


def dotted_lookup(d: Any, keys: str) -> Any:
    return nested_lookup(d, keys.split("."))
