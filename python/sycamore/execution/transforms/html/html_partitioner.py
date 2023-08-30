from typing import Dict, Any

from bs4 import BeautifulSoup

from data.document import TableElement
from sycamore.data import Element
from sycamore.data import Document
from sycamore.execution.transforms.partition import Partitioner


class HtmlPartitioner(Partitioner):
    def __init__(
            self,
            include_page_breaks: bool = False,
            skip_headers_and_footers: bool = True,
            include_metadata: bool = False,
            extract_tables: bool = False,
            text_chunk_size: int = 1000,
            text_chunk_overlap_size: int = 100,
            **kwargs):
        self._include_page_breaks = include_page_breaks
        self._skip_headers_and_footers = skip_headers_and_footers
        self._include_metadata = include_metadata
        self._extract_tables = extract_tables
        self._text_chunk_size = text_chunk_size
        self._text_chunk_overlap_size = text_chunk_overlap_size
        self._unresolved = kwargs

    @staticmethod
    def to_element(dict: Dict[str, Any]) -> Element:
        element = Element()
        element.type = dict.pop("type")
        element.content = dict.pop("text")
        element.properties.update(dict.pop("metadata"))
        element.properties.update(dict)
        return element

    @staticmethod
    def get_overlapped_chunks(text: str, text_chunk_size: int, text_chunk_overlap_size: int):
        return [text[a:a + text_chunk_size] for a in range(0, len(text), text_chunk_size - text_chunk_overlap_size)]

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
        for chunk in self.get_overlapped_chunks(text, self._text_chunk_size, self._text_chunk_overlap_size):
            element = Element()
            element.type = "text"
            element.content = chunk
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
