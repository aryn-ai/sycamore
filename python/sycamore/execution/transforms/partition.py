import io
from typing import Any, Dict, Optional

from bs4 import BeautifulSoup
from ray.data import Dataset

from sycamore.execution.functions.chunker import TokenOverlapChunker, Chunker
from sycamore.execution.functions.tokenizer import CharacterTokenizer, Tokenizer
from sycamore.data.document import TableElement
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
        element.text_representation = element.content
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


class HtmlPartitionerOptions(PartitionerOptions):
    def __init__(
            self,
            include_page_breaks: bool = False,
            skip_headers_and_footers: bool = True,
            include_metadata: bool = False,
            extract_tables: bool = False,
            text_chunker: Chunker = TokenOverlapChunker(),
            tokenizer: Tokenizer = CharacterTokenizer()):
        self.include_page_breaks = include_page_breaks
        self.skip_headers_and_footers = skip_headers_and_footers
        self.include_metadata = include_metadata
        self.extract_tables = extract_tables
        self.text_chunker = text_chunker
        self.tokenizer = tokenizer


class HtmlPartitioner(Partitioner):
    def __init__(
            self,
            options: HtmlPartitionerOptions):
        self._options = options

    def partition(self, dict: Dict[str, Any]) -> Dict[str, Any]:
        document = Document(dict)
        properties = document.properties
        raw_html = document.content

        # note: if content is bytes, BeautifulSoup default to utf-8 encoding
        soup = BeautifulSoup(raw_html, 'html.parser')

        # extract title
        titles = soup.find_all("title")
        title = document["doc_id"]
        if len(titles) > 0:
            title = titles[0].text.replace("\n", "").strip()
        document.properties["title"] = title

        # chunk text and create text elements
        elements = []
        text = soup.get_text()
        tokens = self._options.tokenizer.tokenize(text)
        for chunk in self._options.text_chunker.chunk(tokens):
            content = "".join(chunk)
            element = Element()
            element.type = "text"
            element.content = content
            element.text_representation = content
            if "metadata" in dict:
                element.properties.update(dict.pop("metadata"))
            element.properties.update(properties)
            elements += [element]
        document.elements.extend(elements)

        # extract tables
        if self._options.extract_tables:
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
                table_element.content = table.text
                if "metadata" in dict:
                    table_element.properties.update(dict.pop("metadata"))
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

        return document.to_dict()


class Partition(SingleThreadUser, NonGPUUser, Transform):
    def __init__(
            self, child: Node, options: PartitionerOptions, **resource_args):
        super().__init__(child, **resource_args)
        match options:
            case PdfPartitionerOptions():
                self._partitioner = PdfPartitioner(options)
            case HtmlPartitionerOptions():
                self._partitioner = HtmlPartitioner(options)
            case _:
                raise RuntimeError("Invalid Options")

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        dataset = input_dataset.map(self._partitioner.partition)
        return dataset
