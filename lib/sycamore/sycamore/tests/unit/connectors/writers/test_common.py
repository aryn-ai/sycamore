from sycamore.connectors.writers.common import convert_to_str_dict, drop_types, flatten_data


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
