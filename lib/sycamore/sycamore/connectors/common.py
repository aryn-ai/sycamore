from dataclasses import dataclass
from typing import Callable, Iterator, Union, Iterable, Tuple, Any
import json
import string
import random


@dataclass
class HostAndPort:
    host: str
    port: int


DEFAULT_RECORD_PROPERTIES: dict[str, Any] = {
    "doc_id": None,
    "type": None,
    "text_representation": None,
    "elements": [],
    "embedding": None,
    "parent_id": None,
    "properties": {},
    "bbox": None,
    "shingles": None,
}


def generate_random_string(length=8):
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def filter_doc(obj, include):
    return {k: v for k, v in obj.__dict__.items() if k in include}


def check_dictionary_compatibility(dict1: dict[Any, Any], dict2: dict[Any, Any], ignore: list[str] = []):
    for k in dict1:
        if ignore and any(val in k for val in ignore):
            continue
        if k not in dict2:
            return False
        if dict1[k] != dict2[k]:
            return False
    return True


def compare_docs(doc1, doc2):
    filtered_doc1 = filter_doc(doc1, DEFAULT_RECORD_PROPERTIES.keys())
    filtered_doc2 = filter_doc(doc2, DEFAULT_RECORD_PROPERTIES.keys())
    return filtered_doc1 == filtered_doc2


def _add_key_to_prefix(prefix, key, separator="."):
    if len(prefix) == 0:
        return str(key)
    else:
        return f"{prefix}{separator}{key}"


def flatten_data(
    data: Union[dict, list, tuple],
    prefix: str = "",
    allowed_list_types: list[type] = [],
    homogeneous_lists: bool = True,
    separator: str = ".",
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
                items.append((_add_key_to_prefix(prefix, k, separator), v))
            else:
                inner_values = flatten_data(
                    v, _add_key_to_prefix(prefix, k, separator), allowed_list_types, homogeneous_lists, separator
                )
                items.extend([(innerk, innerv) for innerk, innerv in inner_values])
        elif v is not None:
            items.append((_add_key_to_prefix(prefix, k, separator), v))
    return items


def unflatten_data(data: dict[str, Any], separator: str = ".") -> dict[Any, Any]:
    result: dict[Any, Any] = {}

    def parse_key(key: str) -> list:
        # Handle escaped separator
        parts = []
        current = ""
        escape = False
        for char in key:
            if escape:
                if char == separator:
                    current += separator
                else:
                    current += "\\" + char
                escape = False
            elif char == "\\":
                escape = True
            elif char == separator:
                parts.append(current)
                current = ""
            else:
                current += char
        parts.append(current)
        return parts

    for key, value in data.items():
        parts = parse_key(key)
        current = result
        for i, part in enumerate(parts):
            part_key: Union[str, int] = int(part) if part.isdigit() else part
            is_last = i == len(parts) - 1
            if is_last:
                current[part_key] = value
            else:
                next_part_is_digit = parts[i + 1].isdigit() if i + 1 < len(parts) else False
                if part_key not in current:
                    current[part_key] = [] if next_part_is_digit else {}
                current = current[part_key]
                # If current is a list and the next part is a digit, ensure proper length
                if isinstance(current, list):
                    if next_part_is_digit and len(current) <= int(parts[i + 1]):
                        current.extend("" for _ in range(int(parts[i + 1]) - len(current) + 1))

    return result


def convert_to_str_dict(data: dict[str, Any]) -> dict[str, str]:
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = value
        elif isinstance(value, (int, float, bool)):
            result[key] = str(value)
        elif value is None:
            result[key] = ""
        elif isinstance(value, (list, dict)):
            result[key] = json.dumps(value, separators=(",", ":"))
        else:
            result[key] = repr(value)
    return result


def drop_types(
    data: Union[dict, list, tuple],
    drop_nones: bool = True,
    drop_empty_lists: bool = False,
    drop_empty_dicts: bool = False,
    drop_additional_types: list[type] = [],
) -> Union[dict, list, tuple]:
    if isinstance(data, dict):
        dropped_dict = {
            k: (
                drop_types(v, drop_nones, drop_empty_lists, drop_empty_dicts, drop_additional_types)
                if isinstance(v, (list, tuple, dict))
                else v
            )
            for k, v in data.items()
        }
        if drop_nones:
            dropped_dict = _filter_dict(_none_filter, dropped_dict)
        if drop_empty_lists:
            dropped_dict = _filter_dict(_empty_list_filter, dropped_dict)
        if drop_empty_dicts:
            dropped_dict = _filter_dict(_empty_dict_filter, dropped_dict)
        if len(drop_additional_types) > 0:
            dropped_dict = _filter_dict(_make_type_filter(drop_additional_types), dropped_dict)
        return dropped_dict
    elif isinstance(data, (list, tuple)):
        dropped_list = [
            drop_types(v, drop_nones, drop_empty_lists, drop_empty_dicts, drop_additional_types) for v in data
        ]
        if drop_nones:
            dropped_list = _filter_list(_none_filter, dropped_list)
        if drop_empty_lists:
            dropped_list = _filter_list(_empty_list_filter, dropped_list)
        if drop_empty_dicts:
            dropped_list = _filter_list(_empty_dict_filter, dropped_list)
        if len(drop_additional_types) > 0:
            dropped_list = _filter_list(_make_type_filter(drop_additional_types), dropped_list)
        return data.__class__(dropped_list)
    return data


def _filter_dict(f: Callable[[Any], bool], d: dict):
    return {k: v for k, v in d.items() if f(v)}


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
        return not isinstance(x, tuple(types))

    return _type_filter
