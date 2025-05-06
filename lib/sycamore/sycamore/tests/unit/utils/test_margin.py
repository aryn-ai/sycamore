from typing import Optional
import pytest

from sycamore.data import Element
from sycamore.utils.margin import find_transform_page, cached_bbox_tag, get_bbox_prefer_cached


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
def test_margin_transform_page(
    original_bboxes: list[tuple[float, float, float, float]],
    expected_final_coordinates: list[tuple[float, float, float, float]],
) -> None:
    elements = [Element({"bbox": bbox}) for bbox in original_bboxes]
    _, transform = find_transform_page(elements)
    final_bboxes: list[Optional[tuple[float, ...]]] = []
    for element in elements:
        if bbox := get_bbox_prefer_cached(element, transform):
            final_bboxes.append(tuple(bbox.to_list()))
        else:
            final_bboxes.append(None)
    for element, actual_bbox, expected_bbox in zip(elements, final_bboxes, expected_final_coordinates):
        assert actual_bbox == expected_bbox
        assert actual_bbox == tuple(element.data[cached_bbox_tag].to_list())
