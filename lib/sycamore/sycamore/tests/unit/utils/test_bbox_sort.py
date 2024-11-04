from typing import Any, Optional

from sycamore.data import Document, Element
from sycamore.utils.bbox_sort import (
    collect_pages,
    col_tag,
    find_overlap,
    bbox_sorted_elements,
    bbox_sort_page,
    bbox_sort_document,
)


def mkElem(
    left: float, top: float, right: float, bot: float, page: Optional[int] = None, type: str = "Text"
) -> Element:
    d: dict[str, Any] = {"bbox": (left, top, right, bot), "type": type}
    if page is not None:
        d["properties"] = {"page_number": page}
    return Element(d)


def test_collect() -> None:
    e0 = mkElem(0.0, 0.0, 0.0, 0.0, 2)
    e1 = mkElem(0.0, 0.0, 0.0, 0.0, 1)
    e2 = mkElem(0.0, 0.0, 0.0, 0.0, 3)
    e3 = mkElem(0.0, 0.0, 0.0, 0.0, 1)
    e4 = mkElem(0.0, 0.0, 0.0, 0.0, 2)
    e5 = mkElem(0.0, 0.0, 0.0, 0.0, 3)
    e6 = mkElem(0.0, 0.0, 0.0, 0.0)
    elems = [e0, e1, e2, e3, e4, e5, e6]
    pages = collect_pages(elems)
    assert e6 in pages[0]
    assert e1 in pages[1]
    assert e3 in pages[1]
    assert e0 in pages[2]
    assert e4 in pages[2]
    assert e2 in pages[3]
    assert e5 in pages[3]


def test_col_tag() -> None:
    full = mkElem(0.1, 0.5, 0.9, 0.7)
    left = mkElem(0.1, 0.1, 0.45, 0.3)
    right = mkElem(0.6, 0.1, 0.9, 0.3)
    none = mkElem(0.4, 0.8, 0.6, 0.9)
    assert col_tag(full) == "full"
    assert col_tag(left) == "left"
    assert col_tag(right) == "right"
    assert col_tag(none) is None


def test_overlap() -> None:
    elems = [
        mkElem(0.6, 1, 0.9, 4),  # 0
        mkElem(0.6, 2, 0.9, 6),  # 1
        mkElem(0.1, 3, 0.4, 10),  # 2
        mkElem(0.1, 6, 0.4, 8),  # 3
        mkElem(0.6, 7, 0.9, 11),  # 4
        mkElem(0.6, 10, 0.9, 13),  # 5
        mkElem(0.6, 11, 0.9, 12),  # 6
    ]
    for elem in elems:
        elem.data["_coltag"] = col_tag(elem)
    assert find_overlap(5, 9, elems) == elems[1:5]


def test_page_basic() -> None:
    # e1, e2 in left  column, e1.top < e2.top
    # e0, e3 in right column, e0.top < e3.top
    # e4 full width, at top
    e0 = mkElem(0.52, 0.21, 0.90, 0.45)
    e1 = mkElem(0.10, 0.21, 0.48, 0.46)
    e2 = mkElem(0.10, 0.58, 0.48, 0.90)
    e3 = mkElem(0.58, 0.51, 0.90, 0.85)
    e4 = mkElem(0.15, 0.10, 0.85, 0.15)
    elems = [e0, e1, e2, e3, e4]
    bbox_sort_page(elems)
    answer = [e4, e1, e2, e0, e3]
    assert elems == answer


def test_elements_basic() -> None:
    # e1.top < e0.top = e2.top, e0.left < e2.left both on left
    e0 = mkElem(0.20, 0.50, 0.45, 0.70, 3)
    e1 = mkElem(0.20, 0.21, 0.45, 0.41, 3)
    e2 = mkElem(0.51, 0.50, 0.90, 0.70, 3)

    # e4, e5 in left column, e4.top < e5.top
    # e3, e6 in right column, e3.top < e6.top
    e3 = mkElem(0.52, 0.21, 0.90, 0.45, 1)
    e4 = mkElem(0.10, 0.21, 0.48, 0.46, 1)
    e5 = mkElem(0.10, 0.58, 0.48, 0.90, 1)
    e6 = mkElem(0.58, 0.51, 0.90, 0.85, 1)

    # all the same, test stable
    e7 = mkElem(0.20, 0.21, 0.90, 0.41, 2)
    e8 = mkElem(0.20, 0.21, 0.90, 0.41, 2)
    e9 = mkElem(0.20, 0.21, 0.90, 0.41, 2)

    elems = [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9]
    elems = bbox_sorted_elements(elems)
    answer = [e4, e5, e3, e6, e7, e8, e9, e1, e0, e2]
    assert elems == answer
    assert_element_index_sorted(elems)


def test_document_basic() -> None:
    e0 = mkElem(0.1, 0.5, 0.9, 0.2, 3)
    e1 = mkElem(0.1, 0.1, 0.9, 0.2, 3)
    e2 = mkElem(0.1, 0.5, 0.9, 0.2, 1)
    e3 = mkElem(0.1, 0.1, 0.9, 0.2, 1)
    e4 = mkElem(0.1, 0.5, 0.9, 0.2, 2)
    e5 = mkElem(0.1, 0.1, 0.9, 0.2, 2)
    doc = Document()
    doc.elements = [e0, e1, e2, e3, e4, e5]
    bbox_sort_document(doc)
    answer = [e3, e2, e5, e4, e1, e0]
    assert doc.elements == answer
    assert_element_index_sorted(doc.elements)


def test_page_footer() -> None:
    # e1, e2 in left  column, e1.top < e2.top
    # e0, e3 in right column, e0.top < e3.top
    # e4 full width, at top
    # e5 in left column, but page-footer
    e0 = mkElem(0.52, 0.21, 0.90, 0.45)
    e1 = mkElem(0.10, 0.21, 0.48, 0.46)
    e2 = mkElem(0.10, 0.58, 0.48, 0.90)
    e3 = mkElem(0.58, 0.51, 0.90, 0.85)
    e4 = mkElem(0.15, 0.10, 0.85, 0.15)
    e5 = mkElem(0.10, 0.95, 0.48, 1.0, type="Page-footer")
    elems = [e0, e1, e2, e3, e4, e5]
    bbox_sort_page(elems)
    answer = [e4, e1, e2, e0, e3, e5]
    assert elems == answer


def assert_element_index_sorted(elements: list[Element]):
    assert all(
        elements[i].element_index < elements[i + 1].element_index for i in range(len(elements) - 1)  # type: ignore
    )
