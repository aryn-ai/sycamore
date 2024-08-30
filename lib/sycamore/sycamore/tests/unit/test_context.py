import sycamore


def test_init():
    context = sycamore.init()
    assert context is not None
    assert len(context.rewrite_rules) == 2

    another_context = sycamore.init()
    assert context is context
    assert another_context is not context
