import io
from typing import Any, Dict, Optional

from ray.data import Dataset

from execution.functions import reorder_elements
from sycamore.data import Document, Element
from sycamore.execution import Node, Transform


class Partitioner:
    @staticmethod
    def to_element(dict: Dict[str, Any]) -> Element:
        element = Element()
        element.type = dict.pop("type")
        element.content = dict.pop("text")
        element.properties.update(dict.pop("metadata"))
        element.properties.update(dict)
        return element


# This comparator helps sort the elements per page specifically when a page
# has two columns
def _elements_reorder_comparator(element1: Element, element2: Element) -> int:
    # In PixelSpace (default coordinate system), the coordinates of each
    # element starts in the top left corner and proceeds counter-clockwise. The
    # following function checks if the x0 point of the element is in the
    # left column
    def element_in_left_col(e: Element) -> bool:
        width = e.properties.get("coordinates").get("layout_width")
        x0 = e.properties.get("coordinates").get("points")[0][0]
        return x0 / width <= 0.5

    page1 = element1.properties.get("page_number")
    page2 = element2.properties.get("page_number")

    if page1 < page2:
        return -1
    elif page1 > page2:
        return 1
    else:
        if element_in_left_col(element1) and not element_in_left_col(element2):
            return -1
        elif not element_in_left_col(element1) and element_in_left_col(
                element2):
            return 1
        else:
            return 0


class PdfPartitioner(Partitioner):
    def __init__(
            self,
            include_page_breaks: bool = False,
            strategy: str = "auto",
            infer_table_structure: bool = False,
            ocr_languages: str = "eng",
            max_partition: Optional[int] = None,
            include_metadata: bool = True,
            **kwargs
    ):
        self._include_page_breaks = include_page_breaks
        self._strategy = strategy
        self._infer_table_structure = infer_table_structure
        self._ocr_languages = ocr_languages
        self._max_partition = max_partition
        self._include_metadata = include_metadata
        self._unresolved = kwargs

    def partition(self, dict: Dict[str, Any]) -> Dict[str, Any]:
        document = Document(dict)
        from unstructured.partition.pdf import partition_pdf

        binary = io.BytesIO(document.content)
        elements = partition_pdf(
            file=binary,
            include_page_breaks=self._include_page_breaks,
            strategy=self._strategy,
            infer_table_structure=self._infer_table_structure,
            ocr_languages=self._ocr_languages,
            max_partition=self._max_partition,
            include_metadata=self._include_metadata,
        )
        elements = [self.to_element(element.to_dict()) for element in elements]
        document.elements.extend(elements)
        document = reorder_elements(document, _elements_reorder_comparator)
        return document.to_dict()


class UnstructuredPartition(Transform):
    def __init__(self, child: Node, **kwargs):
        super().__init__(child)
        self._kwargs = kwargs
        self.partitioner = None

    def execute(self) -> "Dataset":
        # TODO, apply rule to bind partitioner dynamically during rewriting
        if self.partitioner is None:
            self.partitioner = PdfPartitioner(**self._kwargs)
        input_dataset = self.child().execute()
        dataset = input_dataset.map(self.partitioner.partition)
        return dataset
