from typing import Any, Optional

from sycamore.data import Document, Element
from sycamore.utils.xycut import (
    xycut_sorted_elements,
    xycut_sorted_page,
    xycut_sort_document,
)


def mkElem(
    left: float, top: float, right: float, bot: float, page: Optional[int] = None, type: str = "Text"
) -> Element:
    d: dict[str, Any] = {"bbox": (left, top, right, bot), "type": type}
    if page is not None:
        d["properties"] = {"page_number": page}
    return Element(d)


def test_page_basic() -> None:
    # e1, e2 in left  column, e1.top < e2.top
    # e0, e3 in right column, e0.top < e3.top
    # e4 full width, at top
    e0 = mkElem(0.59, 0.25, 0.90, 0.60)
    e1 = mkElem(0.10, 0.26, 0.40, 0.51)
    e2 = mkElem(0.10, 0.58, 0.40, 0.90)
    e3 = mkElem(0.60, 0.65, 0.90, 0.85)
    e4 = mkElem(0.15, 0.10, 0.85, 0.15)
    elems = [e0, e1, e2, e3, e4]
    elems = xycut_sorted_page(elems)
    answer = [e4, e1, e2, e0, e3]
    assert elems == answer


def test_elements_basic() -> None:
    # e1.top < e0.top = e2.top, e0.left < e2.left both on left
    e0 = mkElem(0.20, 0.50, 0.45, 0.70, 3)
    e1 = mkElem(0.20, 0.21, 0.45, 0.41, 3)
    e2 = mkElem(0.51, 0.50, 0.90, 0.70, 3)

    # e4, e5 in left column, e4.top < e5.top
    # e3, e6 in right column, e3.top < e6.top
    e3 = mkElem(0.59, 0.25, 0.90, 0.60, 1)
    e4 = mkElem(0.10, 0.26, 0.40, 0.51, 1)
    e5 = mkElem(0.10, 0.58, 0.40, 0.90, 1)
    e6 = mkElem(0.60, 0.65, 0.90, 0.85, 1)

    # all the same, test stable
    e7 = mkElem(0.20, 0.21, 0.90, 0.41, 2)
    e8 = mkElem(0.20, 0.21, 0.90, 0.41, 2)
    e9 = mkElem(0.20, 0.21, 0.90, 0.41, 2)

    elems = [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9]
    elems = xycut_sorted_elements(elems)
    answer = [e4, e5, e3, e6, e7, e8, e9, e1, e0, e2]
    assert elems == answer
    assert_element_index_sorted(elems)


def test_document_basic() -> None:
    e0 = mkElem(0.1, 0.5, 0.9, 0.6, 3)
    e1 = mkElem(0.1, 0.1, 0.9, 0.2, 3)
    e2 = mkElem(0.1, 0.5, 0.9, 0.6, 1)
    e3 = mkElem(0.1, 0.1, 0.9, 0.2, 1)
    e4 = mkElem(0.1, 0.5, 0.9, 0.6, 2)
    e5 = mkElem(0.1, 0.1, 0.9, 0.2, 2)
    doc = Document()
    doc.elements = [e0, e1, e2, e3, e4, e5]
    xycut_sort_document(doc)
    answer = [e3, e2, e5, e4, e1, e0]
    assert doc.elements == answer
    assert_element_index_sorted(doc.elements)


def test_page_footer() -> None:
    # e1, e2 in left  column, e1.top < e2.top
    # e0, e3 in right column, e0.top < e3.top
    # e4 full width, at top
    # e5 in left column, but page-footer
    e0 = mkElem(0.59, 0.25, 0.90, 0.60)
    e1 = mkElem(0.10, 0.26, 0.40, 0.51)
    e2 = mkElem(0.10, 0.58, 0.40, 0.90)
    e3 = mkElem(0.60, 0.65, 0.90, 0.85)
    e4 = mkElem(0.15, 0.10, 0.85, 0.15)
    e5 = mkElem(0.25, 0.95, 0.75, 1.0, type="Page-footer")
    elems = [e0, e1, e2, e3, e4, e5]
    elems = xycut_sorted_page(elems)
    answer = [e4, e1, e2, e0, e3, e5]
    assert elems == answer


def assert_element_index_sorted(elements: list[Element]):
    assert all(
        elements[i].element_index < elements[i + 1].element_index for i in range(len(elements) - 1)  # type: ignore
    )
