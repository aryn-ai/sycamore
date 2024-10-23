from dataclasses import dataclass
from typing import Callable, Iterator, Union, Iterable, Tuple, Any, Dict
from sycamore.data import Document
import json
import string
import random
import math
import numpy as np
import pyarrow as pa
import re


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


def filter_doc(doc: Document, include):
    return {k: v for k, v in doc.items() if k in include}


def check_dictionary_compatibility(dict1: dict[Any, Any], dict2: dict[Any, Any], ignore_list: list[str] = []):
    for k in dict1:
        if not dict1.get(k) or (
            ignore_list
            and any(
                (ignore_value in k and any(k in dict2_k for dict2_k in dict2.keys())) for ignore_value in ignore_list
            )
        ):  # skip if ignored key and if it exists in dict2
            continue
        if k not in dict2:
            return False
        if dict1[k] != dict2[k] and (dict1[k] or dict2[k]):
            return False
    return True


def compare_docs(doc1: Document, doc2: Document):
    filtered_doc1 = filter_doc(doc1, DEFAULT_RECORD_PROPERTIES.keys())
    filtered_doc2 = filter_doc(doc2, DEFAULT_RECORD_PROPERTIES.keys())
    for key in filtered_doc1:
        if isinstance(filtered_doc1[key], (list, np.ndarray)) or isinstance(filtered_doc2.get(key), (list, np.ndarray)):
            assert len(filtered_doc1[key]) == len(filtered_doc2[key])
            for item1, item2 in zip(filtered_doc1[key], filtered_doc2[key]):
                try:
                    # Convert items to float for numerical comparison
                    num1 = float(item1)
                    num2 = float(item2)
                    # Check if numbers are close within tolerance
                    assert math.isclose(num1, num2, rel_tol=1e-5, abs_tol=1e-5)
                except (ValueError, TypeError):
                    # If conversion to float fails, do direct comparison
                    assert item1 == item2
        elif isinstance(filtered_doc1[key], dict) and isinstance(filtered_doc2.get(key), dict):
            assert check_dictionary_compatibility(filtered_doc1[key], filtered_doc2.get(key))
        else:
            assert filtered_doc1[key] == filtered_doc2.get(key)
    return True


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


def unflatten_data(data: dict[Any, Any], separator: str = ".") -> dict[Any, Any]:
    """
    Unflattens a dictionary with keys that contain separators into a nested dictionary. The separator can be escaped,
    and if there are integer keys in the path, the result will be a list instead of a dictionary.
    """

    def split_key(key: str, separator: str = ".") -> list[str]:
        """
        Splits the key by separator (which can be multiple characters), respecting escaped separators.
        """
        parts = []
        current = ""
        i = 0
        while i < len(key):
            if key[i] == "\\":
                # Escape character
                if i + 1 < len(key):
                    current += key[i + 1]
                    i += 2
                else:
                    # Trailing backslash, treat it as literal backslash
                    current += "\\"
                    i += 1
            elif key[i : i + len(separator)] == separator:
                # Found separator
                parts.append(current)
                current = ""
                i += len(separator)
            else:
                current += key[i]
                i += 1
        parts.append(current)
        return parts

    result: dict[Any, Any] = {}
    for flat_key, value in data.items():
        parts = split_key(flat_key, separator)
        current = result
        for i, part in enumerate(parts):
            # Determine whether the key part is an integer (for list indices)
            key: Union[str, int]
            try:
                key = int(part)
            except ValueError:
                key = part

            is_last = i == len(parts) - 1

            if is_last:
                # Set the value at the deepest level
                if isinstance(current, list):
                    # Ensure the list is big enough
                    while len(current) <= key:
                        current.append("")
                    current[key] = value
                else:
                    current[key] = value
            else:
                # Determine the type of the next part
                next_part = parts[i + 1]

                # Check if the next part is an index (integer)
                try:
                    int(next_part)
                    next_is_index = True
                except ValueError:
                    next_is_index = False

                # Initialize containers as needed
                if isinstance(current, list):
                    # Ensure the list is big enough
                    while len(current) <= key:
                        current.append("")
                    if current[key] == "" or current[key] is None:
                        current[key] = [] if next_is_index else {}
                    current = current[key]
                else:
                    if key not in current:
                        current[key] = [] if next_is_index else {}
                    current = current[key]
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


def convert_from_str_dict(data: dict[str, str]) -> dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in data.items():
        if value == "":
            result[key] = None
        elif value.lower() == "true":
            result[key] = True
        elif value.lower() == "false":
            result[key] = False
        else:
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    try:
                        # Try to parse as JSON (for lists and dicts)
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        # If all else fails, keep it as a string
                        result[key] = value
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


def _get_pyarrow_type(key: str, dtype: str) -> pa.DataType:
    if dtype == ("VARCHAR"):
        return pa.string()
    elif dtype == ("DOUBLE"):
        return pa.float64()
    elif dtype == ("BIGINT"):
        return pa.int64()
    elif dtype.startswith("MAP"):
        match = re.match(r"MAP\((.+),\s*(.+)\)", dtype)
        if not match:
            raise ValueError(f"Invalid MAP type format: {dtype}")
        key_type, value_type = match.groups()
        pa_key_type = _get_pyarrow_type(key, key_type)
        pa_value_type = _get_pyarrow_type(key, value_type)
        return pa.map_(pa_key_type, pa_value_type)
    elif dtype == "VARCHAR[]":
        return pa.list_(pa.string())
    elif dtype == "DOUBLE[]" or key == "embedding":  # embedding is a list of floats with a fixed dimension
        return pa.list_(pa.float64())
    elif dtype == "BIGINT[]":
        return pa.list_(pa.int64())
    elif dtype == "FLOAT":
        return pa.float32()
    else:
        raise ValueError(f"Unsupported pyarrow datatype: {dtype}")
