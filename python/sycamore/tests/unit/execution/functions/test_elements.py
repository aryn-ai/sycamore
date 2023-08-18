from sycamore.data import Document, Element
from sycamore.execution.functions import reorder_elements
from sycamore.execution.transforms.partition import \
    _elements_reorder_comparator


class TestElementFunctions:
    def test_reorder_elements_for_pdf(self):
        doc = Document()
        element1 = Element(
            {
                "properties": {
                    "page_number": 1,
                    "element_id": 1,
                    "coordinates": {
                        "layout_width": 612,
                        "points": [
                            [116.345, 124.06215579999991],
                            [116.345, 138.40835579999998],
                            [495.6585279999998, 138.40835579999998],
                            [495.6585279999998, 124.06215579999991],
                        ],
                    },
                }
            }
        )
        element2 = Element(
            {
                "properties": {
                    "page_number": 1,
                    "element_id": 2,
                    "coordinates": {
                        "layout_width": 612,
                        "points": [
                            [71.99999999999994, 252.50643679999996],
                            [71.99999999999994, 264.46163679999995],
                            [116.48529919999994, 264.46163679999995],
                            [116.48529919999994, 252.50643679999996],
                        ],
                    },
                }
            }
        )

        element3 = Element(
            {
                "properties": {
                    "page_number": 1,
                    "element_id": 3,
                    "coordinates": {
                        "layout_width": 612,
                        "points": [
                            [314.641, 254.15232159999994],
                            [314.641, 480.5549216],
                            [541.656813616, 480.5549216],
                            [541.656813616, 254.15232159999994],
                        ],
                    },
                }
            }
        )

        element4 = Element(
            {
                "properties": {
                    "page_number": 1,
                    "element_id": 4,
                    "coordinates": {
                        "layout_width": 612,
                        "points": [
                            [71.691, 272.8323216],
                            [71.691, 450.16692159999997],
                            [298.7414036760001, 450.16692159999997],
                            [298.7414036760001, 272.8323216],
                        ],
                    },
                }
            }
        )

        element5 = Element(
            {
                "properties": {
                    "page_number": 2,
                    "element_id": 5,
                    "coordinates": {
                        "layout_width": 612,
                        "points": [
                            [116.345, 124.06215579999991],
                            [116.345, 138.40835579999998],
                            [495.6585279999998, 138.40835579999998],
                            [495.6585279999998, 124.06215579999991],
                        ],
                    },
                }
            }
        )
        element6 = Element(
            {
                "properties": {
                    "page_number": 2,
                    "element_id": 6,
                    "coordinates": {
                        "layout_width": 612,
                        "points": [
                            [71.99999999999994, 252.50643679999996],
                            [71.99999999999994, 264.46163679999995],
                            [116.48529919999994, 264.46163679999995],
                            [116.48529919999994, 252.50643679999996],
                        ],
                    },
                }
            }
        )
        doc.elements = [element1, element2, element3, element4, element5,
                        element6]
        comparator = _elements_reorder_comparator
        doc = reorder_elements(doc, comparator)
        assert doc.elements[2] == element4
