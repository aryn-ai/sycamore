import json
import pytest

from sycamore.utils.extract_json import extract_json


def test_perfect():
    want = {"a": 5, "b": {"c": "y"}}
    assert extract_json(json.dumps(want)) == want


def test_none():
    want = {"None": None}
    input = '{ "None": None }'
    assert extract_json(input) == want


def test_bad_escape():
    want = "\xff"
    input = '"\xff"'  # json.loads("\xFF") -> error; escaping is \uHHHH
    assert extract_json(input) == want


def test_code_block():
    want = {"a": 5, "x": "y"}
    # Note extract_json does not tolerate any leading whitespace
    input = """```json
{ "a": 5, "x": "y" }
```
"""
    assert extract_json(input) == want


def test_fails():
    with pytest.raises(ValueError):
        extract_json("1-2")
