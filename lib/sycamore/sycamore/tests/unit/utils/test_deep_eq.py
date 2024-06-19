from sycamore.utils.deep_eq import deep_eq


def test_deep_eq():
    class Tmp:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def doit():
            pass

    assert deep_eq(Tmp(1, 1), Tmp(1, 1))
    assert not deep_eq(Tmp(1, 1), Tmp(1, 2))
