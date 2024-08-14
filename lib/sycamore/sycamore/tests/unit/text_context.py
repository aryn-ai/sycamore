import sycamore


def test_init():
    context = sycamore.init()
    assert context is not None and context.rewrite_rules == []

    another_context = sycamore.init()
    assert another_context is context
