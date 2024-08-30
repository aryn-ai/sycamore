import sycamore
from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.reader import DocSetReader
from sycamore.transforms.extract_document_structure import ExtractSummaries, StructureBySection, StructureByDocument
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

    def test_structure_by_section(self):
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

    def test_structure_by_document(self):
        context = sycamore.init()
        reader = DocSetReader(context)
        ds = reader.document(self.docs)

        ds = ds.extract_document_structure(structure=StructureByDocument)
        docs = ds.take_all()

        for document in docs:
            assert document.data["label"] == "DOCUMENT"
            assert len(document.children) == 1
            for section in document.children:
                assert "summary" in section.data
                assert section.data["label"] == "SECTION"
                assert len(section.children) == 4
                for element in section.children:
                    assert element.data["label"] == "ELEMENT"

    def test_summarize_sections(self):
        context = sycamore.init()
        reader = DocSetReader(context)
        ds = reader.document(self.docs)

        ds = ds.extract_document_structure(structure=StructureBySection)
        ds.plan = ExtractSummaries(ds.plan)
        docs = ds.take_all()

        summaries = [
            "-----SECTION TITLE: header-1-----\n---Element Type: text---\ni'm text-1\n",
            "-----SECTION TITLE: header-2-----\n---Element Type: text---\ni'm text-2\n",
        ]

        for document in docs:
            for index, section in enumerate(document.children):
                logger.warning(section.data["summary"])
                assert section.data["summary"] == summaries[index]
