import math
from typing import List

from data import Document, Element


# As of now we are re-ordering all the elements of a page if a page has more
# than one column
def reorder_elements(document: Document) -> Document:
    elements = document.elements
    page_number = 1
    sorted_elements = []
    elements_per_page = []
    for i, element in enumerate(elements):
        # collect all the elements per page
        # check if a page has 2 columns
        # Reorder elements if it has 2 cols
        if page_number == element.properties.get("page_number"):
            elements_per_page.append(element)
        else:
            x0_values = _get_x0_values(elements_per_page)
            if _is_page_2_columns(x0_values):
                sorted_elements += _sort_elements(elements_per_page, x0_values)

            page_number = element.properties.get("page_number")
            elements_per_page = [element]

    document.elements = sorted_elements

    return document


def _get_x0_values(elements: List[Element]):
    # x0 is in terms of the width percentage
    x0_values = []
    for e in elements:
        width = e.properties.get("coordinates").get("layout_width")
        x0_values.append(e.properties.get("coordinates").get("points")[0][0] / width)
    return x0_values


def _is_page_2_columns(x0_values: List) -> bool:
    # In PixelSpace (default coordinate system), the coordinates of each
    # element starts in the top left corner and proceeds counter-clockwise.
    # Heuristic : Calculate the standard deviation of x0 value of each element.
    mean = sum(x0_values) / len(x0_values)
    std = math.sqrt(sum([(mean - x) ** 2 for x in x0_values]) / len(x0_values))
    return std > max(x0_values) / 6


def _sort_elements(elements: List[Element], x0_values: List) -> List[Element]:
    left_elements = []
    right_elements = []
    for i, element in enumerate(elements):
        if x0_values[i] <= 0.5:
            left_elements.append(element)
        else:
            right_elements.append(element)

    return left_elements + right_elements
