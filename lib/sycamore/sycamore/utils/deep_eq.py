from sycamore.data import Element


def assert_deep_eq(a, b, path):
    assert type(a) == type(b), f"type {a} {b} {path}"
    if a is None:
        assert b is None
        return True

    if isinstance(a, int) or isinstance(a, float) or isinstance(a, str):
        assert a == b, f"values {a} {b} at {path}"
        return True

    if isinstance(a, list) or isinstance(a, tuple):
        assert len(a) == len(b), f"length {len(a)} {len(b)} {path}"
        for i, v in enumerate(a):
            assert_deep_eq(a[i], b[i], path + [i])
        return True

    if isinstance(a, dict):
        for k, v in a.items():
            assert k in b, f"missing {k} in b={b} at {path} from {a}"
            assert_deep_eq(a[k], b[k], path + [k])
        for k, v in b.items():
            assert k in a, f"missing {k} in a={a} at {path} from {b}"

        return True

    if isinstance(a, Element):
        assert_deep_eq(a.data, b.data, path + [".data"])
        return True

    if "__class__" in dir(a):
        for k in dir(a):
            if k.startswith("__"):
                continue
            assert k in dir(b)
            assert_deep_eq(getattr(a, k), getattr(b, k), path + ["." + k])
        return True
    assert False, f"Don't know how to compare {a}/{type(a)} with {b} at {path}"


def deep_eq(a, b):
    try:
        assert_deep_eq(a, b, [])
        return True
    except AssertionError as e:
        print(f"Fail {e}")
        return False
