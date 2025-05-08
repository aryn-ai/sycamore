from typing import Optional
import pytest

from sycamore.data import Element
from sycamore.utils.margin import find_matrix_page
from sycamore.utils.bbox_sort import apply_matrix


@pytest.mark.parametrize(
    "original_bboxes, expected_final_coordinates",
    [
        (
            [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0), (0.1, 0.2, 0.3, 0.4), (1.0, 1.0, 1.0, 1.0)],
            [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0), (0.1, 0.2, 0.3, 0.4), (1.0, 1.0, 1.0, 1.0)],
        ),
        (
            [(0.1, 0.1, 0.9, 0.9)],
            [(0.0, 0.0, 1.0, 1.0)],
        ),
        ([], []),
        (
            [(0.1, 0.1, 0.6, 0.2), (0.1, 0.8, 0.6, 0.9)],
            [(0.0, 0, 1, 0.125), (0, 0.875, 1, 1)],
        ),
        (
            [(0.3, 0.1, 0.6, 0.9), (0.1, 0.5, 0.9, 0.9)],
            [(0.25, 0.0, 0.625, 1.0), (0, 0.5, 1.0, 1.0)],
        ),
        (
            [(0.1, 0.1, 0.2, 0.2)],
            [(0.1, 0.1, 0.2, 0.2)],
        ),
    ],
    ids=["identity", ".1 margin", "no elements", "list", "different widths", "revert to identity"],
)
def test_margin_matrix_page(
    original_bboxes: list[tuple[float, float, float, float]],
    expected_final_coordinates: list[tuple[float, float, float, float]],
) -> None:
    elements = [Element({"bbox": bbox}) for bbox in original_bboxes]
    transform = find_matrix_page(elements)
    final_bboxes: list[Optional[tuple[float, ...]]] = []
    for element in elements:
        if element.bbox is None:
            final_bboxes.append(None)
        else:
            bbox = apply_matrix(element.bbox, transform)
            final_bboxes.append(tuple(bbox.to_list()))
    for element, actual_bbox, expected_bbox in zip(elements, final_bboxes, expected_final_coordinates):
        assert actual_bbox == expected_bbox
