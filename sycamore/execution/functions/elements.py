import functools
from typing import Callable

from sycamore.data import Document, Element


def reorder_elements(
    document: Document,
    comparator: Callable[[Element, Element], int],
):
    elements = document.elements
    elements.sort(key=functools.cmp_to_key(comparator))
    return document
