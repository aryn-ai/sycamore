import numpy as np

from sycamore.data import Element


def find_transform_page(elems: list[Element]) -> tuple[bool, np.ndarray]:
    is_reasonable, left, top, right, bottom = find_margins_and_check_are_reasonable(elems)
    width = right - left
    height = bottom - top
    if is_reasonable:
        # fmt: off
        return is_reasonable, np.array([[1/width, 0,          -left/width],
                                        [0,         1/height, -top/height],
                                        [0,         0,        1]])
        # fmt: on
    else:
        return is_reasonable, np.eye(3)


def find_margins_and_check_are_reasonable(elements: list[Element]) -> tuple[bool, float, float, float, float]:
    is_reasonable = True
    left, top, right, bottom = find_margin_of_pages(elements)
    if left > 0.4:
        is_reasonable = False
    if right < 0.6:
        is_reasonable = False
    if top > 0.4:
        is_reasonable = False
    if bottom < 0.6:
        is_reasonable = False
    return is_reasonable, left, top, right, bottom


def find_margin_of_pages(elements: list[Element]) -> tuple[float, float, float, float]:
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
