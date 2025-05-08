"""
Utilities to sort elements based on bounding box (bbox) coordinates.

TODO:
- handle page_number not (always) present
- handle bbox not (always) present
"""

from typing import Optional

import numpy as np

from sycamore.data import Document, Element
from sycamore.data.bbox import BoundingBox
from sycamore.data.document import DocumentPropertyTypes
from sycamore.utils.margin import find_transform_page


cached_bbox_tag = "_transformed_bbox"


def bbox_margin_sort_page(elements: list[Element]) -> None:
    transform = find_transform_page(elements)
    bbox_sort_page(elements, transform)


def bbox_sort_page(elems: list[Element], transform: Optional[np.ndarray] = None) -> None:
    """If you want to sort without accounting for margins, call this function without specifying a transform. Like so:
    bbox_sort_page(elements)"""
    if len(elems) < 2:
        return
    sorter = BBoxSorter(transform)
    elems.sort(key=sorter.elem_top_left)  # sort top-to-bottom, left-to-right
    for elem in elems:  # tag left/right/full based on width/position
        elem.data["_coltag"] = sorter.col_tag(elem)
    tag_two_columns(elems)
    bbox_sort_based_on_tags(elems)
    for elem in elems:
        elem.data.pop("_coltag", None)  # clean up tags
    clear_cached_bboxes(elems)


class BBoxSorter:
    def __init__(self, transform: Optional[np.ndarray]) -> None:
        if transform is None:
            transform = np.eye(3)
        self.transform = transform
        if (transform == np.eye(3)).all():
            self.max_width = 0.45
        else:
            self.max_width = 0.5

    def elem_top_left(self, elem: Element) -> tuple[float, float]:
        cached_bbox = self.get_bbox_prefer_cached(elem)
        if cached_bbox:
            return (cached_bbox.y1, cached_bbox.x1)
        return (0.0, 0.0)

    def col_tag(self, elem: Element) -> Optional[str]:
        cached_bbox = self.get_bbox_prefer_cached(elem)
        if cached_bbox:
            left = cached_bbox.x1
            right = cached_bbox.x2
            width = right - left
            if width > 0.6 or elem.type == "Page-footer":
                return "full"
            elif (width < 0.1) or (width >= self.max_width):
                return None
            if right < 0.5:
                return "left"
            elif left > 0.5:
                return "right"
        return None

    def get_bbox_prefer_cached(self, elem: Element) -> Optional[BoundingBox]:
        if (cached := elem.data.get(cached_bbox_tag)) is not None:
            return cached
        elif (bbox := elem.bbox) is not None:
            cache = apply_transform(bbox, self.transform)
            elem.data[cached_bbox_tag] = cache
            return cache
        else:
            return None


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
        ebot = bbox[3]
        if etop > bot:
            break
        if (etop < bot) and (ebot > top):
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


def bbox_sorted_elements(elements: list[Element]) -> list[Element]:
    pages = collect_pages(elements)
    for elems in pages:
        bbox_sort_page(elems)
    ordered_elements = [elem for elems in pages for elem in elems]  # flatten
    for idx, element in enumerate(ordered_elements):
        element.element_index = idx
    return ordered_elements


def bbox_sort_document(doc: Document) -> None:
    doc.elements = bbox_sorted_elements(doc.elements)


def clear_cached_bboxes(elems: list[Element]) -> None:
    for elem in elems:
        elem.data.pop(cached_bbox_tag, None)


def apply_transform(bbox: BoundingBox, transform: Optional[np.ndarray]) -> BoundingBox:
    if transform is None:
        return bbox
    x1, y1, x2, y2 = bbox.to_list()
    # fmt: off
    old_coords = np.array([[x1, x2],
                            [y1, y2],
                            [1,  1]])
    # fmt: on
    new_coords = np.dot(transform, old_coords)
    new_x1, new_x2 = new_coords[0]
    new_y1, new_y2 = new_coords[1]
    return BoundingBox(new_x1, new_y1, new_x2, new_y2)
