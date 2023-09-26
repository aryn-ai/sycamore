from abc import abstractmethod, ABC
import io
from typing import Any, Optional

from bs4 import BeautifulSoup
from ray.data import Dataset

from sycamore.execution.functions.chunker import TokenOverlapChunker, Chunker
from sycamore.execution.functions.tokenizer import CharacterTokenizer, Tokenizer
from sycamore.data.document import TableElement
from sycamore.execution.functions import reorder_elements
from sycamore.data import Document, Element
from sycamore.execution import Node, Transform, SingleThreadUser, NonGPUUser
from sycamore.execution.transforms.mapping import generate_map_function
from sycamore.execution.transforms.table_extraction import TableExtractor


# This comparator helps sort the elements per page specifically when a page
# has two columns
def _elements_reorder_comparator(element1: Element, element2: Element) -> int:
    # In PixelSpace (default coordinate system), the coordinates of each
    # element starts in the top left corner and proceeds counter-clockwise. The
    # following function checks if the x0 point of the element is in the
    # left column
    def element_in_left_col(e: Element) -> bool:
        width = e.properties["coordinates"]["layout_width"]
        x0 = e.properties["coordinates"]["points"][0][0]
        return x0 / width <= 0.5

    page1 = element1.properties["page_number"]
    page2 = element2.properties["page_number"]

    if page1 < page2:
        return -1
    elif page1 > page2:
        return 1
    else:
        if element_in_left_col(element1) and not element_in_left_col(element2):
            return -1
        elif not element_in_left_col(element1) and element_in_left_col(element2):
            return 1
        else:
            return 0


class Partitioner(ABC):
    @staticmethod
    def to_element(dict: dict[str, Any]) -> Element:
        element = Element()
        element.type = dict.pop("type")
        element.binary_representation = dict.pop("text")
        element.text_representation = str(element.binary_representation)
        element.properties.update(dict.pop("metadata"))
        element.properties.update(dict)

        # TODO, we need handle cases of different types for same column
        if element.properties.get("coordinates") is not None:
            coordinates = element.properties["coordinates"]
            if coordinates.get("layout_height") is not None:
                coordinates["layout_height"] = float(coordinates["layout_height"])
            if coordinates.get("layout_width") is not None:
                coordinates["layout_width"] = float(coordinates["layout_width"])
        return element

    @abstractmethod
    def partition(self, document: Document) -> Document:
        pass


class UnstructuredPdfPartitioner(Partitioner):
    def __init__(
        self,
        include_page_breaks: bool = False,
        strategy: str = "auto",
        infer_table_structure: bool = False,
        ocr_languages: str = "eng",
        max_partition_length: Optional[int] = None,
        min_partition_length: Optional[int] = None,
        include_metadata: bool = True,
    ):
        self._include_page_breaks = include_page_breaks
        self._strategy = strategy
        self._infer_table_structure = infer_table_structure
        self._ocr_languages = ocr_languages
        self._max_partition_length = max_partition_length
        self._min_partition_length = min_partition_length
        self._include_metadata = include_metadata

    def partition(self, document: Document) -> Document:
        from unstructured.partition.pdf import partition_pdf

        binary = io.BytesIO(document.data["binary_representation"])
        elements = partition_pdf(
            file=binary,
            include_page_breaks=self._include_page_breaks,
            strategy=self._strategy,
            infer_table_structure=self._infer_table_structure,
            ocr_languages=self._ocr_languages,
            max_partition=self._max_partition_length,
            min_partition=self._min_partition_length,
            include_metadata=self._include_metadata,
        )
        elements = [self.to_element(element.to_dict()) for element in elements]
        document.elements.extend(elements)
        document = reorder_elements(document, _elements_reorder_comparator)
        return document


class HtmlPartitioner(Partitioner):
    def __init__(
        self,
        include_page_breaks: bool = False,
        skip_headers_and_footers: bool = True,
        include_metadata: bool = False,
        extract_tables: bool = False,
        text_chunker: Chunker = TokenOverlapChunker(),
        tokenizer: Tokenizer = CharacterTokenizer(),
    ):
        self._include_page_breaks = include_page_breaks
        self._skip_headers_and_footers = skip_headers_and_footers
        self._include_metadata = include_metadata
        self._extract_tables = extract_tables
        self._text_chunker = text_chunker
        self._tokenizer = tokenizer

    def partition(self, document: Document) -> Document:
        properties = document.properties
        raw_html = document.binary_representation

        if raw_html is None:
            raise RuntimeError("Attempting to partition invalid document where content=None")

        # note: if content is bytes, BeautifulSoup default to utf-8 encoding
        soup = BeautifulSoup(raw_html, "html.parser")

        # extract title
        titles = soup.find_all("title")
        title = document["doc_id"]
        if len(titles) > 0:
            title = titles[0].text.replace("\n", "").strip()
        document.properties["title"] = title

        # chunk text and create text elements
        elements = []
        text = soup.get_text()
        tokens = self._tokenizer.tokenize(text)
        for chunk in self._text_chunker.chunk(tokens):
            content = "".join(chunk)
            element = Element()
            element.type = "text"
            element.text_representation = content
            element.properties.update(properties)
            elements += [element]
        document.elements.extend(elements)

        # extract tables
        if self._extract_tables:
            for table in soup.find_all("table"):
                # ignore nested tables
                if len(table.find_all("table")) > 0:
                    continue

                table_element = TableElement()

                # find headers if they exist
                headers = table.findAll("th")
                if len(headers) > 0:
                    table_element.columns = [tag.text for tag in headers]

                table_element.text_representation = table.text
                table_element.properties.update(properties)

                # parse all rows, use all text as content
                rows = table.findAll("tr")
                table_element.rows = []
                for row in rows:
                    cols = row.findAll("td")
                    if len(cols) > 0:
                        row_vals = [tag.text for tag in cols]
                        table_element.rows += [row_vals]

                document.elements.extend([table_element])

        return document


class Partition(SingleThreadUser, NonGPUUser, Transform):
    def __init__(
        self, child: Node, partitioner: Partitioner, table_extractor: Optional[TableExtractor] = None, **resource_args
    ):
        super().__init__(child, **resource_args)
        self._partitioner = partitioner
        self._table_extractor = table_extractor

    def execute(self) -> Dataset:
        input_dataset = self.child().execute()
        dataset = input_dataset.map(generate_map_function(self._partitioner.partition))
        if self._table_extractor:
            dataset = dataset.map(generate_map_function(self._table_extractor.extract_tables))
        return dataset
