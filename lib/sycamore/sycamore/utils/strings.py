import textwrap


def dedent(text: str) -> str:
    """Remove any common leading whitespace from every line in `text`.

    This can be used to make triple-quoted strings line up with the left
    edge of the display, while still presenting them in the source code
    in indented form.

    If the first character is a newline, drop it. This allows triple-quoted
    strings to avoid having to stick a \\ on the first line in order to dedent
    properly, e.g. '''
       long
       string
       '''

    Note that tabs and spaces are both treated as whitespace, but they
    are not equal: the lines "  hello" and "\\thello" are
    considered to have no common leading whitespace.

    Entirely blank lines are normalized to a newline character.
    """
    if text[0] == "\n":
        text = text[1:]
    return textwrap.dedent(text)
