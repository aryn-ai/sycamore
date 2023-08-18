import io
from typing import Any, Dict, Optional

from ray.data import Dataset

from sycamore.execution.functions import reorder_elements
from sycamore.data import (Document, Element)
from sycamore.execution import (
    Node, Transform, SingleThreadUser, NonGPUUser)


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


class PartitionerOptions:
    pass


class PdfPartitionerOptions(PartitionerOptions):
    def __init__(
            self,
            include_page_breaks: bool = False,
            strategy: str = "auto",
            infer_table_structure: bool = False,
            ocr_languages: str = "eng",
            max_partition: Optional[int] = None,
            include_metadata: bool = True):
        self.include_page_breaks = include_page_breaks
        self.strategy = strategy
        self.infer_table_structure = infer_table_structure
        self.ocr_languages = ocr_languages
        self.max_partition = max_partition
        self.include_metadata = include_metadata


class PdfPartitioner(Partitioner):
    def __init__(self, options: PdfPartitionerOptions):
        self._options = options

    def partition(self, dict: Dict[str, Any]) -> Dict[str, Any]:
        document = Document(dict)
        from unstructured.partition.pdf import partition_pdf

        binary = io.BytesIO(document.content)
        elements = partition_pdf(
            file=binary,
            include_page_breaks=self._options.include_page_breaks,
            strategy=self._options.strategy,
            infer_table_structure=self._options.infer_table_structure,
            ocr_languages=self._options.ocr_languages,
            max_partition=self._options.max_partition,
            include_metadata=self._options.include_metadata)
        elements = [self.to_element(element.to_dict()) for element in elements]
        document.elements.extend(elements)
        document = reorder_elements(document, _elements_reorder_comparator)
        return document.to_dict()


class UnstructuredPartition(SingleThreadUser, NonGPUUser, Transform):
    def __init__(
            self, child: Node, options: PartitionerOptions, **resource_args):
        super().__init__(child, **resource_args)
        match options:
            case PdfPartitionerOptions():
                self._partitioner = PdfPartitioner(options)
            case _:
                raise RuntimeError("Invalid Options")

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        dataset = input_dataset.map(self._partitioner.partition)
        return dataset
