from sycamore.data import Document, Element
from sycamore.functions import reorder_elements
from sycamore.transforms.partition import _elements_reorder_comparator


class TestElementFunctions:
    def test_reorder_elements_for_pdf(self):
        doc = Document()
        element1 = Element(
            {
                "bbox": (116.345 / 612, 124.06215579999991, 495.6585279999998 / 612, 138.40835579999998),
                "properties": {
                    "page_number": 1,
                    "element_id": 1,
                },
            }
        )
        element2 = Element(
            {
                "bbox": (71.99999999999994 / 612, 252.50643679999996, 116.48529919999994 / 612, 264.46163679999995),
                "properties": {
                    "page_number": 1,
                    "element_id": 2,
                },
            }
        )

        element3 = Element(
            {
                "bbox": (314.641 / 612, 254.15232159999994, 541.656813616 / 612, 480.5549216),
                "properties": {
                    "page_number": 1,
                    "element_id": 3,
                },
            }
        )

        element4 = Element(
            {
                "bbox": (71.691 / 612, 272.8323216, 298.7414036760001 / 612, 450.16692159999997),
                "properties": {
                    "page_number": 1,
                    "element_id": 4,
                },
            }
        )

        element5 = Element(
            {
                "bbox": (116.345 / 612, 124.06215579999991, 495.6585279999998 / 612, 138.40835579999998),
                "properties": {
                    "page_number": 2,
                    "element_id": 5,
                },
            }
        )
        element6 = Element(
            {
                "bbox": (71.99999999999994 / 612, 252.50643679999996, 116.48529919999994 / 612, 264.46163679999995),
                "properties": {
                    "page_number": 2,
                    "element_id": 6,
                },
            }
        )
        doc.elements = [element1, element2, element3, element4, element5, element6]
        comparator = _elements_reorder_comparator
        doc = reorder_elements(doc, comparator)
        assert doc.elements[2] == element4
