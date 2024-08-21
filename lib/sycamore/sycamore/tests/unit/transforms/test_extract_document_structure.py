import sycamore
from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.reader import DocSetReader
from sycamore.transforms.extract_document_structure import StructureBySection
import logging

logger = logging.getLogger(__name__)

class TestExtractDocumentStructure:
    docs = [
        Document(
            {
                "doc_id": "1",
                "type": "pdf",
                "properties": {"company": "3M", "sector": "Industrial", "doctype": "10K"},
                "elements": [
                    Element(
                        {
                            "type": "Section-header",
                            "text_representation": "header-1",
                            "properties": {},
                        }
                    ),
                    Element(
                        {
                            "type": "text",
                            "text_representation": "i'm text-1",
                            "properties": {},
                        }
                    ),
                    Element(
                        {
                            "type": "Section-header",
                            "text_representation": "header-2",
                            "properties": {},
                        }
                    ),
                    Element(
                        {
                            "type": "text",
                            "text_representation": "i'm text-2",
                            "properties": {},
                        }
                    ),
                ],
            }
        )
    ]

    def test_extract_document_structure(self):
        context = sycamore.init()
        reader = DocSetReader(context)
        ds = reader.document(self.docs)

        ds = ds.extract_document_structure(structure=StructureBySection)
        docs = ds.take_all()

        for document in docs:
            assert document.data["label"] == "DOCUMENT"
            for section in document.children:
                assert "summary" in section.data
                assert section.data["label"] == "SECTION"
                for element in section.children:
                    assert element.data["label"] == "ELEMENT"
