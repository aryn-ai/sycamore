from dataclasses import dataclass
from typing import Callable, Iterator, Union, Iterable, Tuple, Any


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
                # Allow lists of the allowed_list_types
                items.append((_add_key_to_prefix(prefix, k), v))
            else:
                inner_values = flatten_data(v, _add_key_to_prefix(prefix, k), allowed_list_types, homogeneous_lists)
                items.extend([(innerk, innerv) for innerk, innerv in inner_values])
        elif v is not None:
            items.append((_add_key_to_prefix(prefix, k), v))
    return items


def drop_types(
    data: Union[dict, list, tuple],
    drop_nones: bool = True,
    drop_empty_lists: bool = False,
    drop_empty_dicts: bool = False,
    drop_additional_types: list[type] = [],
) -> Union[dict, list, tuple]:
    if isinstance(data, dict):
        droppedd = {k: drop_types(v) for k, v in data.items()}
        if drop_nones:
            droppedd = _filter_dict(_none_filter, droppedd)
        if drop_empty_lists:
            droppedd = _filter_dict(_empty_list_filter, droppedd)
        if drop_empty_dicts:
            droppedd = _filter_dict(_empty_dict_filter, droppedd)
        if len(drop_additional_types) > 0:
            droppedd = _filter_dict(_make_type_filter(drop_additional_types), droppedd)
        return droppedd
    elif isinstance(data, (list, tuple)):
        droppedl = [drop_types(v) for v in data]
        if drop_nones:
            droppedl = _filter_list(_none_filter, droppedl)
        if drop_empty_lists:
            droppedl = _filter_list(_empty_list_filter, droppedl)
        if drop_empty_dicts:
            droppedl = _filter_list(_empty_dict_filter, droppedl)
        if len(drop_additional_types) > 0:
            droppedl = _filter_list(_make_type_filter(drop_additional_types), droppedl)
        return data.__class__(droppedl)
    return data


def _filter_dict(f: Callable[[Any], bool], d: dict):
    return {k: v for k, v in d if f(v)}


def _filter_list(f: Callable[[Any], bool], ls: list):
    return [v for v in ls if f(v)]


def _none_filter(x: Any) -> bool:
    return x is not None


def _empty_list_filter(x: Any) -> bool:
    return x != []


def _empty_dict_filter(x: Any) -> bool:
    return x != {}


def _make_type_filter(types: list[type]) -> Callable[[Any], bool]:
    def _type_filter(x):
        return isinstance(x, tuple(types))

    return _type_filter
