from typing import Optional
import numpy as np

from sycamore.data import Element, BoundingBox
from sycamore.utils.sycamore_logger import get_logger, setup_logger

setup_logger()
g_logger = get_logger()

cached_bbox_tag = "_bbox_accounting_for_margins"


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


def get_bbox_prefer_cached(elem: Element, transform: Optional[np.ndarray]) -> Optional[BoundingBox]:
    if (cached := elem.data.get(cached_bbox_tag)) is not None:
        return cached
    elif (bbox := elem.bbox) is not None:
        cache = apply_transform(bbox, transform)
        elem.data[cached_bbox_tag] = cache
        return cache
    else:
        return None


def apply_transform(bbox: BoundingBox, transform: Optional[np.ndarray]) -> BoundingBox:
    g_logger.info(f"Applying transform {transform} to bbox {bbox}")
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


def clear_cached_bboxes(elems: list[Element]) -> None:
    for elem in elems:
        elem.data.pop(cached_bbox_tag, None)


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
