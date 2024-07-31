from typing import cast
from sycamore.data.document import Document
from sycamore.data.element import TableElement
from sycamore.evaluation.tables.benchmark_scans import TableEvalDoc
from sycamore.transforms.table_structure.extract import TableStructureExtractor


class ExtractTableFromImage:

    def __init__(self, extractor: TableStructureExtractor):
        self._extractor = extractor

    def extract_table(self, docs: list[Document]) -> list[Document]:
        ans = []
        for doc in docs:
            doc = TableEvalDoc(doc)
            assert isinstance(doc, TableEvalDoc), f"Wrong kind of doc: {type(doc)}, {doc}"
            image = doc.image
            assert image is not None
            table_bbox_element = TableElement(bbox=(0, 0, 1, 1), tokens=doc.properties["tokens"])
            predicted_elt = self._extractor.extract(table_bbox_element, image)
            if predicted_elt.table is not None:
                doc.pred_table = predicted_elt.data["table"]
            ans.append(cast(Document, doc))
        return ans

    def __call__(self, docs: list[Document]) -> list[Document]:
        return self.extract_table(docs)
