from abc import abstractmethod, ABC
import io
from typing import Any, Optional

from bs4 import BeautifulSoup
from ray.data import Dataset

from sycamore.functions import TextOverlapChunker, Chunker
from sycamore.functions import CharacterTokenizer, Tokenizer
from sycamore.functions import reorder_elements
from sycamore.data import BoundingBox, Document, Element, TableElement
from sycamore.plan_nodes import Node, Transform, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import generate_map_function
from sycamore.transforms.extract_table import TableExtractor


# This comparator helps sort the elements per page specifically when a page
# has two columns
def _elements_reorder_comparator(element1: Element, element2: Element) -> int:
    # In PixelSpace (default coordinate system), the coordinates of each
    # element starts in the top left corner and proceeds counter-clockwise. The
    # following function checks if the x0 point of the element is in the
    # left column
    def element_in_left_col(e: Element) -> bool:
        if e.bbox is None:
            raise RuntimeError("Element BBox is None")
        return e.bbox.x1 <= 0.5

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
    @abstractmethod
    def partition(self, document: Document) -> Document:
        pass


class UnstructuredPPTXPartitioner(Partitioner):
    """
    UnstructuredPPTXPartitioner utilizes open-source Unstructured library to extract structured elements from
    unstructured PPTX files.

    Args:
        include_page_breaks: Whether to include page breaks as separate elements.
        strategy: The partitioning strategy to use ("auto" for automatic detection).
        infer_table_structure: Whether to infer table structures in the document.
        ocr_languages: The languages to use for OCR. Default is "eng" (English).
        max_partition_length: The maximum length of each partition (in characters).
        include_metadata: Whether to include metadata in the partitioned elements.

    Example:
         .. code-block:: python

            pptx_partitioner = UnstructuredPPTXPartitioner(
                include_page_breaks=False,
                include_metadata=True,
                include_slide_notes=False,
                chunking_strategy=None,
                **kwargs
            )

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pptx")
                .partition(partitioner=pptx_partitioner)

    """

    @staticmethod
    def to_element(dict: dict[str, Any]) -> Element:
        text = dict.pop("text")
        if isinstance(text, str):
            binary = text.encode("utf-8")
        else:
            binary = text
            text = str(binary, "utf-8")

        element = Element()
        element.type = dict.pop("type", "unknown")
        element.binary_representation = binary
        element.text_representation = text
        properties = element.properties
        properties.update(dict.pop("metadata"))
        properties.update(dict)
        element.properties = properties

        return element

    def __init__(
        self,
        include_page_breaks: bool = False,
        include_metadata: bool = True,
        include_slide_notes: bool = False,
        chunking_strategy: Optional[str] = None,
        **kwargs
    ):
        self._include_page_breaks = include_page_breaks
        self._include_metadata = include_metadata
        self._include_slide_notes = include_slide_notes
        self._chunking_strategy = chunking_strategy
        self._kwargs = kwargs

    def partition(self, document: Document) -> Document:
        from unstructured.partition.pptx import partition_pptx

        binary_file = io.BytesIO(document.data["binary_representation"])

        elements = partition_pptx(
            file=binary_file,
            include_page_breaks=self._include_page_breaks,
            include_metadata=self._include_metadata,
            include_slide_notes=self._include_slide_notes,
            chunking_strategy=self._chunking_strategy,
            **self._kwargs
        )

        # Here we convert unstructured.io elements into our elements and
        # append them as child elements to the document.
        document.elements = [self.to_element(element.to_dict()) for element in elements]
        del elements

        return document


class UnstructuredPdfPartitioner(Partitioner):
    """
    UnstructuredPdfPartitioner utilizes open-source Unstructured library to extract structured elements from
    unstructured PDFs.

    Args:
        include_page_breaks: Whether to include page breaks as separate elements.
        strategy: The partitioning strategy to use ("auto" for automatic detection).
        infer_table_structure: Whether to infer table structures in the document.
        ocr_languages: The languages to use for OCR. Default is "eng" (English).
        max_partition_length: The maximum length of each partition (in characters).
        include_metadata: Whether to include metadata in the partitioned elements.

    Example:
         .. code-block:: python

            pdf_partitioner = UnstructuredPdfPartitioner(
                include_page_breaks=True,
                strategy="auto",
                infer_table_structure=True,
                ocr_languages="eng",
                max_partition_length=2000,
                include_metadata=True,
            )

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=pdf_partitioner)

    """

    def __init__(
        self,
        include_page_breaks: bool = False,
        strategy: str = "auto",
        infer_table_structure: bool = False,
        languages: list[str] = ["eng"],
        max_partition_length: Optional[int] = None,
        min_partition_length: Optional[int] = 500,
        include_metadata: bool = True,
    ):
        self._include_page_breaks = include_page_breaks
        self._strategy = strategy
        self._infer_table_structure = infer_table_structure
        self._languages = languages
        self._max_partition_length = max_partition_length
        self._min_partition_length = min_partition_length
        self._include_metadata = include_metadata

    @staticmethod
    def to_element(dict: dict[str, Any]) -> Element:
        text = dict.pop("text")
        if isinstance(text, str):
            binary = text.encode("utf-8")
        else:
            binary = text
            text = str(binary, "utf-8")

        element = Element()
        element.type = dict.pop("type", "unknown")
        element.binary_representation = binary
        element.text_representation = text
        properties = element.properties
        properties.update(dict.pop("metadata"))
        properties.update(dict)
        coordinates = properties.pop("coordinates")
        element.properties = properties

        if coordinates is not None:
            x1 = coordinates.get("points")[0][0] / coordinates.get("layout_width")
            y1 = coordinates.get("points")[0][1] / coordinates.get("layout_height")
            x2 = coordinates.get("points")[2][0] / coordinates.get("layout_width")
            y2 = coordinates.get("points")[2][1] / coordinates.get("layout_height")
            element.bbox = BoundingBox(x1, y1, x2, y2)

        return element

    def partition(self, document: Document) -> Document:
        from unstructured.partition.pdf import partition_pdf

        binary = io.BytesIO(document.data["binary_representation"])
        elements = partition_pdf(
            file=binary,
            include_page_breaks=self._include_page_breaks,
            strategy=self._strategy,
            infer_table_structure=self._infer_table_structure,
            languages=self._languages,
            max_partition=self._max_partition_length,
            min_partition=self._min_partition_length,
            include_metadata=self._include_metadata,
        )

        # Here we convert unstructured.io elements into our elements and
        # set them as the child elements of the document.
        document.elements = [self.to_element(ee.to_dict()) for ee in elements]
        del elements

        document = reorder_elements(document, _elements_reorder_comparator)
        return document


class HtmlPartitioner(Partitioner):
    """
    HtmlPartitioner processes HTML documents extracting structured content.

    Args:
        skip_headers_and_footers: Whether to skip headers and footers in the document. Default is True.
        extract_tables: Whether to extract tables from the HTML document. Default is False.
        text_chunker: The text chunking strategy to use for processing text content.
        tokenizer: The tokenizer to use for tokenizing text content.

    Example:
         .. code-block:: python

            html_partitioner = HtmlPartitioner(
                skip_headers_and_footers=True,
                extract_tables=True,
                text_chunker=TokenOverlapChunker(chunk_token_count=1000, chunk_overlap_token_count=100),
                tokenizer=CharacterTokenizer(),
            )

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="html")
                .partition(partitioner=html_partitioner)
    """

    def __init__(
        self,
        skip_headers_and_footers: bool = True,
        extract_tables: bool = False,
        text_chunker: Chunker = TextOverlapChunker(),
        tokenizer: Tokenizer = CharacterTokenizer(),
    ):
        self._skip_headers_and_footers = skip_headers_and_footers
        self._extract_tables = extract_tables
        self._text_chunker = text_chunker
        self._tokenizer = tokenizer

    def partition(self, document: Document) -> Document:
        raw_html = document.binary_representation

        if raw_html is None:
            raise RuntimeError("Attempting to partition invalid document where content=None")

        # note: if content is bytes, BeautifulSoup default to utf-8 encoding
        soup = BeautifulSoup(raw_html, "html.parser")

        # extract title
        titles = soup.find_all("title")
        title = document.doc_id
        if len(titles) > 0:
            title = titles[0].text.replace("\n", "").strip()
        properties = document.properties
        properties["title"] = title
        document.properties = properties

        # chunk text and create text elements
        elements = []
        text = soup.get_text()
        tokens = self._tokenizer.tokenize(text)
        for chunk in self._text_chunker.chunk(tokens):
            content = "".join(chunk)
            element = Element()
            element.type = "text"
            element.text_representation = content

            element_properties = element.properties
            element_properties.update(properties)
            element.properties = element_properties
            elements += [element]

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
                table_properties = table_element.properties
                table_properties.update(properties)
                table_element.properties = table_properties

                # parse all rows, use all text as content
                rows = table.findAll("tr")
                table_element.rows = []
                for row in rows:
                    cols = row.findAll("td")
                    if len(cols) > 0:
                        row_vals = [tag.text for tag in cols]
                        table_element.rows += [row_vals]
                elements.append(table_element)
        document.elements = document.elements + elements

        return document


class Partition(SingleThreadUser, NonGPUUser, Transform):
    """
    The Partition transform segments documents into elements. For example, a typical partitioner might chunk a document
    into elements corresponding to paragraphs, images, and tables. Partitioners are format specific, so for instance for
    HTML you can use the HtmlPartitioner and for PDFs, we provide the UnstructuredPdfPartitioner, which utilizes the
    unstructured open-source library.

    Args:
        child: The source node or component that provides the dataset to be embedded.
        partitioner: An instance of a Partitioner class to be applied
        resource_args: Additional resource-related arguments that can be passed to the Partition operation.

    Example:
         .. code-block:: python

            source_node = ...  # Define a source node or component that provides a dataset.
            custom_partitioner = MyPartitioner(partitioner_params)
            partition_transform = Partition(child=source_node, partitioner=custom_partitioner)
            partitioned_dataset = partition_transform.execute()
    """

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
