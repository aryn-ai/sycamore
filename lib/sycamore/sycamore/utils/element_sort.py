from typing import Optional, Union, Protocol

from sycamore.data import Document, Element
from sycamore.data.document import DocumentPropertyTypes
from sycamore.utils.bbox_sort import bbox_sort_page
from sycamore.utils.xycut import xycut_sort_page


class PageSorter(Protocol):
    def __call__(self, elems: list[Element], *, left_to_right: bool = True) -> None: ...


def nop_page(elems: list[Element], *, left_to_right: bool = True) -> None:  # noqa: ARG001
    pass


PAGE_SORT: dict[Optional[str], PageSorter] = {
    "bbox": bbox_sort_page,
    "xycut": xycut_sort_page,
    "nop": nop_page,
    None: bbox_sort_page,
}


def collect_pages(elems: list[Element]) -> list[list[Element]]:
    """
    Collect elements into page-number buckets.  Basically like the first
    stage of a radix sort.
    """
    pagemap: dict[int, list[Element]] = {}
    for elem in elems:
        page = elem.properties.get(DocumentPropertyTypes.PAGE_NUMBER, 0)
        ary = pagemap.get(page)
        if ary:
            ary.append(elem)
        else:
            pagemap[page] = [elem]
    nums = list(pagemap.keys())
    nums.sort()
    rv = []
    for page in nums:
        ary = pagemap[page]
        rv.append(ary)
    return rv


# This allows specification of the "page sort" building block either
# as a string ("bbox", "xycut") or Callable.
SortSpec = Union[Optional[str], PageSorter]


def sort_page(elems: list[Element], *, mode: SortSpec = None, left_to_right: bool = True) -> None:
    if callable(mode):
        mode(elems, left_to_right=left_to_right)
    else:
        func = PAGE_SORT[mode]
        func(elems, left_to_right=left_to_right)


def sort_elements(elements: list[Element], *, mode: SortSpec = None, left_to_right: bool = True) -> None:
    pages = collect_pages(elements)
    for page in pages:
        sort_page(page, mode=mode, left_to_right=left_to_right)
    ordered = [elem for elems in pages for elem in elems]  # flatten
    for idx, elem in enumerate(ordered):
        elem.element_index = idx
    elements[:] = ordered  # replace contents


def sort_document(doc: Document, *, mode: SortSpec = None, left_to_right: bool = True) -> None:
    sort_elements(doc.elements, mode=mode, left_to_right=left_to_right)
