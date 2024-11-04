"""
Utilities to sort elements based on bounding box (bbox) coordinates.

TODO:
- handle page_number not (always) present
- handle bbox not (always) present
"""

from typing import Optional

from sycamore.data import Document, Element
from sycamore.data.document import DocumentPropertyTypes


def elem_top_left(elem: Element) -> tuple:
    bbox = elem.data.get("bbox")
    if bbox:
        return (bbox[1], bbox[0])
    return (0.0, 0.0)


def elem_left_top(elem: Element) -> tuple:
    bbox = elem.data.get("bbox")
    if bbox:
        left = int(5 * bbox[0])  # !!! quantize
        return (left, bbox[1])
    return (0.0, 0.0)


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


def col_tag(elem: Element) -> Optional[str]:
    bbox = elem.data.get("bbox")
    if bbox:
        left = bbox[0]
        right = bbox[2]
        width = right - left
        if width > 0.6 or elem.type == "Page-footer":
            return "full"
        elif (width < 0.1) or (width >= 0.45):
            return None
        if right < 0.5:
            return "left"
        elif left > 0.5:
            return "right"
    return None


def find_overlap(top: float, bot: float, elems: list[Element]) -> list[Element]:
    """
    Returns the elements that overlap (top, bot) in the y-axis,
    if and only if there are elements in both left and right columns.
    Assumes elems is sorted by top, ascending.
    """
    rv: list[Element] = []
    lefts = 0
    rights = 0
    for elem in elems:
        bbox = elem.data["bbox"]
        etop = bbox[1]
        if etop > bot:
            break
        if (etop < bot) and (bbox[3] > top):
            rv.append(elem)
            tag = elem.data["_coltag"]
            if tag == "left":
                lefts += 1
            elif tag == "right":
                rights += 1
    if lefts and rights:
        return rv
    return []


def elems_in_row(elem: Element, elems: list[Element]) -> list[Element]:
    _, top, _, bot = elem.data["bbox"]
    return find_overlap(top, bot, elems)


def tag_two_columns(elems: list[Element]) -> None:
    """
    Tag '2col' when 'left' is next to 'right'
    """
    for elem in elems:
        if elem.data["_coltag"] == "left":
            row = elems_in_row(elem, elems)
            for ee in row:
                ee.data["_coltag"] = "2col"


def bbox_sort_two_columns(elems: list[Element], beg: int, end: int) -> None:
    if (end - beg) > 1:
        elems[beg:end] = sorted(elems[beg:end], key=elem_left_top)


def bbox_sort_based_on_tags(elems: list[Element]) -> None:
    """
    Find sections that are two-column and sort them specially.
    Assumes elems already sorted vertically.
    """
    lidx = 0
    ltag = elems[0].data["_coltag"]
    for idx, elem in enumerate(elems):
        tag = elem.data["_coltag"]
        if (tag in ("full", "2col")) and (tag != ltag):
            if ltag == "2col":
                bbox_sort_two_columns(elems, lidx, idx)
            lidx = idx
            ltag = tag
    if ltag == "2col":
        bbox_sort_two_columns(elems, lidx, len(elems))


def bbox_sort_page(elems: list[Element]) -> None:
    if len(elems) < 2:
        return
    elems.sort(key=elem_top_left)  # sort top-to-bottom, left-to-right
    for elem in elems:  # tag left/right/full based on width/position
        elem.data["_coltag"] = col_tag(elem)
    tag_two_columns(elems)
    bbox_sort_based_on_tags(elems)
    for elem in elems:
        elem.data.pop("_coltag", None)  # clean up tags


def bbox_sorted_elements(elements: list[Element], update_element_indexs: bool = True) -> list[Element]:
    pages = collect_pages(elements)
    for elems in pages:
        bbox_sort_page(elems)
    ordered_elements = [elem for elems in pages for elem in elems]  # flatten
    if update_element_indexs:
        for idx, element in enumerate(ordered_elements):
            element.element_index = idx
    return ordered_elements


def bbox_sort_document(doc: Document, update_element_indexs: bool = True) -> None:
    doc.elements = bbox_sorted_elements(doc.elements, update_element_indexs)
