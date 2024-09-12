import functools
from typing import Any, Callable, Optional

from sycamore.data import Document, Element


def reorder_elements(
    document: Document,
    *,
    comparator: Optional[Callable[[Element, Element], int]] = None,
    key: Optional[Callable[[Element], Any]] = None,
) -> Document:
    """Reorders the elements.  Must supply comparator or key.

    Args:
        document: Document for which the elements need to be re-ordered
        comparator: A comparator function
        key: A key as per sorted()

    Returns:
        Document with elements re-ordered
    """
    if key:
        assert not comparator, "passed both comparator and key"
    else:
        assert comparator, "passed neither comparator nor key"
        key = functools.cmp_to_key(comparator)
    elements = document.elements
    elements.sort(key=key)
    document.elements = elements
    return document


def filter_elements(document: Document, filter_function: Callable[[Element], bool]) -> list[Element]:
    """Filters  the elements.

    Args:
        document: Document for which the elements need to be filtered
        filter_function: A filter function

    Returns:
        List of filtered elements
    """
    elements = document.elements
    return list(filter(filter_function, elements))
