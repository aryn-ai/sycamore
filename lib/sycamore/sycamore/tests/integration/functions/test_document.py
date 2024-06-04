from typing import Optional

import sycamore
from sycamore.data import Element
from sycamore.functions.document import split_and_convert_to_image
from sycamore.transforms.partition import UnstructuredPdfPartitioner
from sycamore.tests.config import TEST_DIR


def test_split_and_convert_to_image_empty_page():
    def _drop_page2(element: Element) -> Optional[Element]:
        if element.properties["page_number"] == 2:
            return None
        return element

    path = TEST_DIR / "resources/data/pdfs/Ray.pdf"

    context = sycamore.init()

    # Remove all elements from page 2, and make sure that page2 still shows up in the output.
    docs = (
        context.read.binary(paths=[str(path)], binary_format="pdf")
        .partition(partitioner=UnstructuredPdfPartitioner())
        .map_elements(_drop_page2)
        .flat_map(split_and_convert_to_image)
        .take_all()
    )

    assert len(docs) == 17

    page2_candidate = [d for d in docs if d.properties["page_number"] == 2]

    assert len(page2_candidate) == 1
    assert len(page2_candidate[0].elements) == 0
