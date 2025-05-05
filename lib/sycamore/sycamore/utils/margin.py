import numpy as np

from sycamore.data import Element, BoundingBox

margin_tag = "_pre_margin_transform_bbox"


def margin_transform_page(elems: list[Element], leave_original_tags: bool = False) -> None:
    is_reasonable, left, top, right, bottom = find_margins_and_check_are_reasonable(elems)
    width = right - left
    height = bottom - top
    if is_reasonable:
        # fmt: off
        transform = np.array([[1/width, 0,        -left/width],
                            [0,         1/height, -top/height],
                            [0,         0,        1]])
        # fmt: on
    else:
        transform = np.eye(3)
    for elem in elems:
        bbox = elem.data.get("bbox")
        if bbox:
            x1, y1, x2, y2 = bbox
            if leave_original_tags:
                elem.data[margin_tag] = bbox[:]
            # fmt: off
            old_coords = np.array([[x1, x2],
                                   [y1, y2],
                                   [1,  1]])
            # fmt: on
            new_coords = np.dot(transform, old_coords)
            new_x1, new_x2 = new_coords[0]
            new_y1, new_y2 = new_coords[1]
            elem.bbox = BoundingBox(new_x1, new_y1, new_x2, new_y2)


def revert_margin_transform_page(elems: list[Element]) -> None:
    for elem in elems:
        if elem.data.get("bbox") is not None and (old_bbox := elem.data.get(margin_tag)) is not None:
            elem.data["bbox"] = old_bbox
            del elem.data[margin_tag]


def find_margins_and_check_are_reasonable(elements: list[Element]) -> tuple[bool, float, float, float, float]:
    is_reasonable = True
    left, top, right, bottom = find_margin_page(elements)
    if left > 0.4:
        is_reasonable = False
    if right < 0.6:
        is_reasonable = False
    if top > 0.4:
        is_reasonable = False
    if bottom < 0.6:
        is_reasonable = False
    return is_reasonable, left, top, right, bottom


def find_margin_page(elements: list[Element]) -> tuple[float, float, float, float]:
    """
    Find the margin of the page.
    :param elements: list of elements
    :return: tuple of (left, top, right, bottom)
    """
    left = float("inf")
    top = float("inf")
    right = float("-inf")
    bottom = float("-inf")
    set_at_least_once = False

    for elem in elements:
        if elem.type in ("Page-header", "Image", "Page-footer"):
            continue
        bbox = elem.data.get("bbox")
        if bbox:
            left = min(left, bbox[0])
            top = min(top, bbox[1])
            right = max(right, bbox[2])
            bottom = max(bottom, bbox[3])
            set_at_least_once = True

    if not set_at_least_once:
        left = 0.0
        top = 0.0
        right = 1.0
        bottom = 1.0

    return left, top, right, bottom
