from abc import abstractmethod, ABC
from typing import Optional

from textractor import Textractor
from textractor.data.constants import TextractFeatures

from sycamore.data import Document, Element


class TableExtractor(ABC):
    @abstractmethod
    def extract_tables(self, document: Document) -> Document:
        pass


class TextractTableExtractor(TableExtractor):
    def __init__(self, profile_name: Optional[str] = None, region_name: Optional[str] = None, kms_key_id: str = ""):
        self._profile_name = profile_name
        self._region_name = region_name
        self._kms_key_id: str = kms_key_id

    def _extract(self, document: Document) -> list[Element]:
        # https://docs.aws.amazon.com/textract/latest/dg/API_BoundingBox.html
        def bbox_to_coord(bbox):
            return [bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height]

        extractor = Textractor(self._profile_name, self._region_name, self._kms_key_id)
        result = extractor.start_document_analysis(document.properties["path"], TextractFeatures.TABLES)

        # map page_number -> list of tables on that page number
        all_tables = []
        for table in result.tables:
            element = Element()
            element.type = "Table"
            element.properties["boxes"] = []
            element.properties["id"] = table.id
            element.properties["page"] = table.page

            if table.title:
                element.text_representation = table.title.text + "\n"
                element.properties["boxes"].append(bbox_to_coord(table.title.bbox))
            else:
                element.text_representation = ""

            element.text_representation = element.text_representation + table.to_csv() + "\n"
            element.properties["boxes"].append(bbox_to_coord(table.bbox))

            for footer in table.footers:
                element.text_representation = element.text_representation + footer.text + "\n"
                element.properties["boxes"].append(bbox_to_coord(footer.bbox))

            all_tables.append(element)

        return all_tables

    def extract_tables(self, document: Document) -> Document:
        tables = self._extract(document)
        document.elements.extend(tables)
        return document
