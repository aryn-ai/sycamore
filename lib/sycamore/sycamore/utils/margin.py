import numpy as np

from sycamore.data import Element


def find_matrix_page(elems: list[Element]) -> np.ndarray:
    margins = find_margin_of_pages(elems)
    if margins.are_reasonable():
        width = margins.right - margins.left
        height = margins.bottom - margins.top
        # fmt: off
        return np.array([[1/width, 0,          -margins.left/width],
                         [0,         1/height, -margins.top/height],
                         [0,         0,        1]])
        # fmt: on
    else:
        return np.eye(3)


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

    return Margins(left, top, right, bottom)


class Margins:
    """Warning: `bottom` and `right` are measured from the top left corner of the page, not the bottom right."""

    def __init__(self, left: float, top: float, right: float, bottom: float) -> None:
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __repr__(self) -> str:
        return f"Margins(left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom})"

    def are_reasonable(self) -> bool:
        if self.left > 0.4:
            return False
        if self.right < 0.6:
            return False
        if self.top > 0.4:
            return False
        if self.bottom < 0.6:
            return False
        return True
