from sycamore.connectors.common import convert_to_str_dict, drop_types, flatten_data, unflatten_data


def test_flatten_data_happy():
    data = {"a": {"b": "c", "d.d": "e", "f": 32, "g": ["1", "2", "3", {"h": "i"}]}}
    flattened = dict(flatten_data(data))
    assert flattened == {"a.b": "c", "a.d.d": "e", "a.f": 32, "a.g.0": "1", "a.g.1": "2", "a.g.2": "3", "a.g.3.h": "i"}


def test_flatten_data_separator():
    data = {"a": {"b": "c", "d": "e", "f": 32, "g": ["1", "2", "3"]}}
    flattened = dict(flatten_data(data, separator="#"))
    assert flattened == {"a#b": "c", "a#d": "e", "a#f": 32, "a#g#0": "1", "a#g#1": "2", "a#g#2": "3"}


def test_flatten_data_allow_list_of_str():
    data = {"a": {"b": "c", "d": "e", "f": 32, "g": ["1", "2", "3"]}}
    flattened = dict(flatten_data(data, allowed_list_types=[str]))
    assert flattened == {"a.b": "c", "a.d": "e", "a.f": 32, "a.g": ["1", "2", "3"]}


def test_flatten_data_allow_heterogeneous_list():
    data = {"a": {"b": "c", "d": "e", "f": 32, "g": ["1", 2, "3"]}}
    flattened = dict(flatten_data(data, allowed_list_types=[str, int], homogeneous_lists=False))
    assert flattened == {"a.b": "c", "a.d": "e", "a.f": 32, "a.g": ["1", 2, "3"]}


def test_convert_to_str_dict():
    data = {
        "string": "string",
        "int": 1,
        "float": 1.5,
        "bool": True,
        "none": None,
        "list": ["item"],
        "dict": {"key": "value"},
        "fn": lambda x: x + 1,
    }
    stringified = convert_to_str_dict(data)
    assert stringified["string"] == "string"
    assert stringified["int"] == "1"
    assert stringified["float"] == "1.5"
    assert stringified["bool"] == "True"
    assert stringified["none"] == ""
    assert stringified["list"] == '["item"]'
    assert stringified["dict"] == '{"key":"value"}'
    assert stringified["fn"].startswith("<function test_convert_to_str_dict")


def test_drop_types_nones():
    data = {"key": "value", "none": None, "empty_list": [], "empty_dict": {}}
    dropped = drop_types(data)
    assert isinstance(dropped, dict)
    assert dropped == {"key": "value", "empty_list": [], "empty_dict": {}}


def test_drop_types_empty_lists():
    data = {"key": "value", "none": None, "empty_list": [], "empty_dict": {}}
    dropped = drop_types(data, drop_nones=False, drop_empty_lists=True)
    assert isinstance(dropped, dict)
    assert dropped == {"key": "value", "none": None, "empty_dict": {}}


def test_drop_types_empty_dicts():
    data = {"key": "value", "none": None, "empty_list": [], "empty_dict": {}}
    dropped = drop_types(data, drop_empty_dicts=True)
    assert isinstance(dropped, dict)
    assert dropped == {"key": "value", "empty_list": []}


def test_drop_types_also_strings():
    data = {"key": "value", "none": None, "empty_list": [], "empty_dict": {}}
    dropped = drop_types(data, drop_additional_types=[str])
    assert isinstance(dropped, dict)
    assert dropped == {"empty_list": [], "empty_dict": {}}


def test_drop_types_nested():
    data = {"outer": {"key": "value", "none": None, "empty_list": [], "empty_dict": {}}}
    dropped = drop_types(data, drop_empty_dicts=True, drop_empty_lists=True, drop_additional_types=[str])
    assert isinstance(dropped, dict)
    assert dropped == {}


def test_drop_types_list_to_list():
    data = [None, "Hello"]
    dropped = drop_types(data)
    assert isinstance(dropped, list)
    assert dropped == ["Hello"]


def test_drop_types_tuple_to_tuple():
    data = (None, "Hello")
    dropped = drop_types(data)
    assert isinstance(dropped, tuple)
    assert dropped == ("Hello",)


def test_basic_unflattening():
    data = {"a.b.c": 1, "a.b.d": 2, "a.e": 3}
    unflattened = unflatten_data(data)
    assert isinstance(unflattened, dict)
    assert unflattened == {"a": {"b": {"c": 1, "d": 2}, "e": 3}}


def test_numeric_keys():
    data = {"a.0": "zero", "a.1": "one", "a.2": "two"}
    unflattened = unflatten_data(data)
    assert isinstance(unflattened, dict)
    assert unflattened == {"a": ["zero", "one", "two"]}


def test_mixed_numeric_and_string_keys():
    data = {"a.0": "zero", "a.1": "one", "a.b": "bee", "a.2": "two"}
    unflattened = unflatten_data(data)
    assert isinstance(unflattened, dict)
    assert unflattened == {"a": {0: "zero", 1: "one", 2: "two", "b": "bee"}}


def test_deep_nesting():
    data = {"a.b.c.d.e.f.g": "deep"}
    unflattened = unflatten_data(data)
    assert isinstance(unflattened, dict)
    assert unflattened == {"a": {"b": {"c": {"d": {"e": {"f": {"g": "deep"}}}}}}}


def test_empty_input():
    data = {}
    unflattened = unflatten_data(data)
    assert isinstance(unflattened, dict)
    assert unflattened == {}


def test_no_nesting():
    data = {"a": 1, "b": 2, "c": 3}
    unflattened = unflatten_data(data)
    assert isinstance(unflattened, dict)
    assert unflattened == {"a": 1, "b": 2, "c": 3}


def test_custom_separator():
    data = {"a/b/c": 1, "a/b/d": 2}
    unflattened = unflatten_data(data, separator="/")
    assert isinstance(unflattened, dict)
    assert unflattened == {"a": {"b": {"c": 1, "d": 2}}}


def test_keys_with_dots():
    data = {"a.b\\.c": 1, "a.b\\.d": 2}
    unflattened = unflatten_data(data)
    assert isinstance(unflattened, dict)
    assert unflattened == {"a": {"b.c": 1, "b.d": 2}}


def test_non_string_values():
    data = {"a.b": 1, "a.c": [1, 2, 3], "a.d": {"nested": "dict"}}
    unflattened = unflatten_data(data)
    assert isinstance(unflattened, dict)
    assert unflattened == {"a": {"b": 1, "c": [1, 2, 3], "d": {"nested": "dict"}}}


def test_invalid_numeric_sequence():
    data = {"a.0": "zero", "a.2": "two"}
    unflattened = unflatten_data(data)
    assert isinstance(unflattened, dict)
    assert unflattened == {"a": {0: "zero", 2: "two"}}
