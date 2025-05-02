import logging
import numpy as np

from sycamore.data import Element

margin_tag = "_pre_margin_transform_bbox"


def margin_transform_page(elems: list[Element], leave_original_tags: bool = False) -> None:
    margins = find_reasonable_margin_page(elems)
    left, _, right, _ = margins
    width = right - left
    logging.info(f"Margin sort page: left={left}, right={right}, width={width}")
    # fmt: off
    transform = np.array([[1/width,   0, -left],
                          [0,         1, 0],
                          [0,         0, 1]])
    # fmt: on
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
            elem.data["bbox"] = [new_x1, new_y1, new_x2, new_y2]


def revert_margin_transform_page(elems: list[Element]) -> None:
    for elem in elems:
        if elem.data.get("bbox") is not None and (old_bbox := elem.data.get(margin_tag)) is not None:
            elem.data["bbox"] = old_bbox
            del elem.data[margin_tag]


def find_reasonable_margin_page(elements: list[Element]) -> tuple[float, float, float, float]:
    left, top, right, bottom = find_margin_page(elements)
    left = min(left, 0.4)
    right = max(right, 0.6)
    return left, top, right, bottom


def find_margin_page(elements: list[Element]) -> tuple[float, float, float, float]:
    """
    Find the margin of the page.
    :param elements: list of elements
    :return: tuple of (left, top, right, bottom)
    """
    left = float("inf")
    top = float("-inf")
    right = float("-inf")
    bottom = float("inf")
    set_at_least_once = False

    for elem in elements:
        if elem.type in ("Page-header", "Image", "Page-footer"):
            continue
        bbox = elem.data.get("bbox")
        if bbox:
            left = min(left, bbox[0])
            top = max(top, bbox[1])
            right = max(right, bbox[2])
            bottom = min(bottom, bbox[3])
            set_at_least_once = True

    if not set_at_least_once:
        left = 0.0
        top = 0.0
        right = 1.0
        bottom = 1.0

    return left, top, right, bottom
