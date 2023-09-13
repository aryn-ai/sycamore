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
