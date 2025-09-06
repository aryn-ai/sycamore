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
    # Test without and with leading whitespce
    input = """```json
{ "a": 5, "x": "y" }
```
"""
    assert extract_json(input) == want
    assert extract_json("\n" + input) == want


def test_code_block_no_newline():
    want = {"a": 5, "x": "y"}
    # No LLM tested will do this; they always put in some newlines, but some of our
    # tests do it (and expect whitespace removal); and it's sufficiently unique to be fine.
    input = '```json {"a": 5, "x": "y"} ```'
    assert extract_json(input) == want
    assert extract_json(input + "\n") == want
    assert extract_json(input + " ") == want


def test_nested_code_block():
    want = {"a": 5, "x": "```json ... ```"}
    input = """
I am so helpful, the json you want is:
```json
{ "a": 5, "x": "```json ... ```" }
```
That is some perfect json."""

    got = extract_json(input, verbose=True)
    assert got == want


def test_fails():
    with pytest.raises(ValueError):
        extract_json("1-2")
