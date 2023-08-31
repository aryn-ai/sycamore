import io
from typing import (Any, Dict, Optional)

from bs4 import BeautifulSoup
from ray.data import Dataset

from sycamore.execution.functions.chunker import TokenOverlapChunker, Chunker
from sycamore.execution.functions.tokenizer import CharacterTokenizer, Tokenizer
from sycamore.data.document import TableElement
from sycamore.data import (Document, Element)
from sycamore.execution import (Node, Transform)


class Partitioner:
    @staticmethod
    def to_element(dict: Dict[str, Any]) -> Element:
        element = Element()
        element.type = dict.pop("type")
        element.content = dict.pop("text")
        element.properties.update(dict.pop("metadata"))
        element.properties.update(dict)
        return element


class PdfPartitioner(Partitioner):
    def __init__(
            self,
            include_page_breaks: bool = False,
            strategy: str = "auto",
            infer_table_structure: bool = False,
            ocr_languages: str = "eng",
            max_partition: Optional[int] = None,
            include_metadata: bool = True,
            **kwargs):
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
            include_metadata=self._include_metadata)
        elements = [self.to_element(element.to_dict()) for element in elements]
        document.elements.extend(elements)
        return document.to_dict()


class HtmlPartitioner(Partitioner):
    def __init__(
            self,
            include_page_breaks: bool = False,
            skip_headers_and_footers: bool = True,
            include_metadata: bool = False,
            extract_tables: bool = False,
            text_chunker: Chunker = TokenOverlapChunker(),
            tokenizer: Tokenizer = CharacterTokenizer(),
            **kwargs):
        self._include_page_breaks = include_page_breaks
        self._skip_headers_and_footers = skip_headers_and_footers
        self._include_metadata = include_metadata
        self._extract_tables = extract_tables
        self._text_chunker = text_chunker
        self._tokenizer = tokenizer
        self._unresolved = kwargs

    @staticmethod
    def to_element(dict: Dict[str, Any]) -> Element:
        element = Element()
        element.type = dict.pop("type")
        element.content = dict.pop("text")
        element.properties.update(dict.pop("metadata"))
        element.properties.update(dict)
        return element

    def partition(self, dict: Dict[str, Any]) -> Dict[str, Any]:
        document = Document(dict)
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
        tokens = self._tokenizer.tokenize(text)
        for chunk in self._text_chunker.chunk(tokens):
            element = Element()
            element.type = "text"
            element.content = "".join(chunk)
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


class Partition(Transform):
    def __init__(self, child: Node, partitioner: Partitioner = None, **kwargs):
        super().__init__(child)
        self._kwargs = kwargs
        self.partitioner = partitioner

    def execute(self) -> "Dataset":
        # TODO, apply rule to bind partitioner dynamically during rewriting
        if self.partitioner is None:
            self.partitioner = PdfPartitioner(**self._kwargs)
        input_dataset = self.child().execute()
        dataset = input_dataset.map(self.partitioner.partition)
        return dataset
