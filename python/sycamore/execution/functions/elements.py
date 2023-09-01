# functions/elements.py
"""
Provides several functions to operate over document elements.

This module contains the following functions:

- `reorder_elements(document, comparator)` - Reorders the document elements
"""

import functools
from typing import Callable

from sycamore.data import Document, Element


def reorder_elements(
        document: Document,
        comparator: Callable[[Element, Element], int],
):
    """
    Reorders the elements.

    Args:
        document: Document for which the elements need to be re-ordered
        comparator: A comparator function

    """
    elements = document.elements
    elements.sort(key=functools.cmp_to_key(comparator))
    return document


def print_doc():
    help(reorder_elements)
    print(reorder_elements.__doc__)


if __name__ == "__main":
    print_doc()
