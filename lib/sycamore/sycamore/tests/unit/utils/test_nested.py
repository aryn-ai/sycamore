from sycamore.utils.nested import nested_lookup, dotted_lookup


def test_nested_lookup():
    v = {"a": {"b": 1, "c": 2, "": 5}, "d": {}}
    assert nested_lookup(v, ["a", "b"]) == 1
    assert nested_lookup(v, ["a", "b", "c"]) is None
    assert nested_lookup(v, ["a", "c"]) == 2
    assert nested_lookup(v, ["a", ""]) == 5
    assert nested_lookup(v, ["a", "d"]) is None
    assert nested_lookup(v, ["d"]) == {}
    assert nested_lookup(v, ["d", 1]) is None
    assert nested_lookup(v, [3]) is None


def test_dotted_lookup():
    v = {"a": {"b": 1, "c": 2, "": 5}, "d": {}}
    assert dotted_lookup(v, "a.b") == 1
    assert dotted_lookup(v, "a.b.c") is None
    assert dotted_lookup(v, "a.c") == 2
    assert dotted_lookup(v, "a.") == 5
    assert dotted_lookup(v, "a.d") is None
    assert dotted_lookup(v, "d") == {}
    assert dotted_lookup(v, "d.1") is None
    assert dotted_lookup(v, "3") is None
