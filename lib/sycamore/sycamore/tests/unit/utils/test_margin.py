import pytest

from sycamore.data import Element
from sycamore.utils.margin import margin_transform_page, margin_tag


@pytest.mark.parametrize(
    "original_bboxes, expected_final_coordinates",
    [
        (
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0), (0.1, 0.2, 0.3, 0.4)],
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0), (0.1, 0.2, 0.3, 0.4)],
        ),
        (
            [(0.1, 0.1, 0.9, 0.9)],
            [(0.0, 0.1, 1.0, 0.9)],
        ),
        ([], []),
        (
            [(0.1, 0.1, 0.6, 0.2), (0.1, 0.3, 0.6, 0.4)],
            [(0.0, 0.1, 1, 0.2), (0, 0.3, 1, 0.4)],
        ),
        (
            [(0.3, 0.1, 0.6, 0.2), (0.1, 0.3, 0.9, 0.4)],
            [(0.25, 0.1, 0.625, 0.2), (0, 0.3, 1, 0.4)],
        ),
    ],
    ids=["identity", ".1 margin", "no elements", "list", "different widths"],
)
def test_margin_transform_page(
    original_bboxes: list[tuple[float, float, float, float]],
    expected_final_coordinates: list[tuple[float, float, float, float]],
) -> None:
    elements = [Element({"bbox": bbox}) for bbox in original_bboxes]
    margin_transform_page(elements, leave_original_tags=True)
    for element, expected_bbox, original_bbox in zip(elements, expected_final_coordinates, original_bboxes):
        if eb := element.bbox is not None:
            bbox = tuple(eb.to_list())
            if bbox != expected_bbox:
                raise AssertionError(f"When inspecting .bbox, expected {expected_bbox}, got {bbox}")
            saved_bbox = element.data[margin_tag]
            if saved_bbox != original_bbox:
                raise AssertionError(f"When inspecting saved tag, expected {original_bbox}, got {saved_bbox}")
