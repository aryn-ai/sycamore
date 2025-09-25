from sycamore.utils.strings import dedent


def test_dedent():
    import textwrap

    s = """
    Some long
    string
    """

    exp = textwrap.dedent(
        """\
    Some long
    string
    """
    )

    assert dedent(s) == exp
