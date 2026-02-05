"""
Utilities to sort elements based on bounding box (bbox) coordinates.

TODO:
- handle page_number not (always) present
- handle bbox not (always) present
"""

from dataclasses import dataclass
from typing import Optional

from sycamore.data import Element


@dataclass
class SortOptions:
    left_to_right: bool = True


def elem_top_left(elem: Element) -> tuple:
    bbox = elem.data.get("bbox")
    if bbox:
        return (bbox[1], bbox[0])
    return (0.0, 0.0)


def elem_top_right(elem: Element) -> tuple:
    bbox = elem.data.get("bbox")
    if bbox:
        return (bbox[1], -bbox[0])
    return (0.0, 0.0)


def elem_left_top(elem: Element) -> tuple:
    bbox = elem.data.get("bbox")
    if bbox:
        left = int(5 * bbox[0])  # !!! quantize
        return (left, bbox[1])
    return (0.0, 0.0)


def elem_right_top(elem: Element) -> tuple:
    bbox = elem.data.get("bbox")
    if bbox:
        left = int(5 * bbox[0])
        return (-left, bbox[1])
    return (0.0, 0.0)


def sort_key_fn(sort_options: SortOptions, horiz_first: bool):
    if horiz_first:
        return elem_left_top if sort_options.left_to_right else elem_right_top
    else:
        return elem_top_left if sort_options.left_to_right else elem_top_right


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


def bbox_sort_two_columns(elems: list[Element], beg: int, end: int, *, sort_options: SortOptions) -> None:
    if (end - beg) > 1:
        key = sort_key_fn(sort_options, horiz_first=True)
        elems[beg:end] = sorted(elems[beg:end], key=key)


def bbox_sort_based_on_tags(elems: list[Element], *, sort_options: SortOptions) -> None:
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
                bbox_sort_two_columns(elems, lidx, idx, sort_options=sort_options)
            lidx = idx
            ltag = tag
    if ltag == "2col":
        bbox_sort_two_columns(elems, lidx, len(elems), sort_options=sort_options)


def bbox_sort_page(elems: list[Element], *, sort_options: Optional[SortOptions] = None) -> None:
    if sort_options is None:
        sort_options = SortOptions()
    if len(elems) < 2:
        return
    sort_key = sort_key_fn(sort_options, horiz_first=False)
    elems.sort(key=sort_key)  # sort top-to-bottom, respecting reading direction
    for elem in elems:  # tag left/right/full based on width/position
        elem.data["_coltag"] = col_tag(elem)
    tag_two_columns(elems)
    bbox_sort_based_on_tags(elems, sort_options=sort_options)
    for elem in elems:
        elem.data.pop("_coltag", None)  # clean up tags
