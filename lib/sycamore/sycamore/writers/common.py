from dataclasses import dataclass
from typing import Iterator, Union, Iterable, Tuple, Any


@dataclass
class HostAndPort:
    host: str
    port: int


def _add_key_to_prefix(prefix, key):
    if len(prefix) == 0:
        return str(key)
    else:
        return f"{prefix}.{key}"


def flatten_data(
    data: Union[dict, list, tuple],
    prefix: str = "",
    allowed_list_types: list[type] = [],
    homogeneous_lists: bool = True,
) -> Iterable[Tuple[Any, Any]]:
    iterator: Union[Iterator[tuple[str, Any]], enumerate[Any]] = iter([])
    if isinstance(data, dict):
        iterator = iter(data.items())
    if isinstance(data, (list, tuple)):
        iterator = enumerate(data)
    items = []
    for k, v in iterator:
        if isinstance(v, (dict, list, tuple)):
            if isinstance(v, (list, tuple)) and (
                (not homogeneous_lists and all(any(isinstance(innerv, t) for t in allowed_list_types) for innerv in v))
                or (homogeneous_lists and any(all(isinstance(innerv, t) for innerv in v) for t in allowed_list_types))
            ):
                # Lists of strings are allowed
                items.append((_add_key_to_prefix(prefix, k), v))
            else:
                inner_values = flatten_data(v, _add_key_to_prefix(prefix, k), allowed_list_types, homogeneous_lists)
                items.extend([(innerk, innerv) for innerk, innerv in inner_values])
        elif v is not None:
            items.append((_add_key_to_prefix(prefix, k), v))
    return items
