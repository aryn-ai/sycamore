from typing import (Any, Dict, List)

from ray.data import Dataset
from textractor import Textractor
from textractor.data.constants import TextractFeatures


from sycamore.data import (Document, Element)
from sycamore.execution import NonGPUUser
from sycamore.execution.basics import (NonCPUUser, Node, Transform)


class TextractorTableExtractor:
    def __init__(
            self,
            profile_name: str = None,
            region_name: str = None,
            kms_key_id: str = ""):
        self._profile_name = profile_name
        self._region_name = region_name
        self._kms_key_id: str = kms_key_id

    def _extract_tables(self, document: Document) -> List[Element]:

        # https://docs.aws.amazon.com/textract/latest/dg/API_BoundingBox.html
        def bbox_to_coord(bbox):
            return [bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height]

        extractor = Textractor(
            self._profile_name, self._region_name, self._kms_key_id)
        result = extractor.start_document_analysis(
            document.properties["path"], TextractFeatures.TABLES)

        # map page_number -> list of tables on that page number
        all_tables = []
        for table in result.tables:
            element = Element()
            element.type = "Table"
            element.properties["boxes"] = []
            element.properties["id"] = table.id
            element.properties["page"] = table.page

            if table.title:
                element.content = table.title.text + "\n"
                element.properties["boxes"].append(
                    bbox_to_coord(table.title.bbox))

            element.content = element.content + table.to_csv() + "\n"
            element.properties["boxes"].append(bbox_to_coord(table.bbox))

            for footer in table.footers:
                element.content = element.content + footer.text + "\n"
                element.properties["boxes"].append(bbox_to_coord(footer.bbox))

            all_tables.append(element)

        return all_tables

    def extract(self, dict: Dict[str, Any]) -> Dict[str, Any]:
        document = Document(dict)
        tables = self._extract_tables(document)
        document.elements.extend(tables)
        return document.to_dict()


class TableExtraction(NonCPUUser, NonGPUUser, Transform):
    def __init__(
            self, child: Node, profile_name: str = None,
            region_name: str = None, kms_key_id: str = "",
            **resource_args):
        super().__init__(child, **resource_args)
        self._table_extractor = \
            TextractorTableExtractor(profile_name, region_name, kms_key_id)

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        dataset = input_dataset.map(self._table_extractor.extract)
        return dataset
