from typing import Optional
from pydantic import BaseModel
import sycamore
from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.llms.llms import LLM
from sycamore.reader import DocSetReader
from sycamore.transforms.extract_document_structure import StructureBySection
from sycamore.transforms.extract_graph_entities import EntityExtractor
from sycamore.transforms.extract_graph_relationships import RelationshipExtractor

import logging

logger = logging.getLogger(__name__)


class TestResolveGraphEntities:
    docs = [
        Document(
            {
                "doc_id": "1",
                "type": "pdf",
                "properties": {},
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

    class MockEntityLLM(LLM):
        def __init__(self):
            super().__init__(model_name="mock_model")

        def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None):
            pass

        def is_chat_mode(self):
            return True

        async def generate_async(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None):
            return """{
                "Company": [
                    {"name": "Microsoft"},
                    {"name": "Google"},
                    {"name": "3M"}
                ]
            }
            """

    class MockRelationshipLLM(LLM):
        def __init__(self):
            super().__init__(model_name="mock_model")

        def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None):
            pass

        def is_chat_mode(self):
            return True

        async def generate_async(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None):
            return """{
                "Competes": [
                {"start": {"name": "Microsoft"}, "end": {"name": "Google"}}
                ]
            }
            """

    def test_resolve_entities(self):
        context = sycamore.init()
        reader = DocSetReader(context)
        ds = reader.document(self.docs)

        class Company(BaseModel):
            name: str

        class Competes(BaseModel):
            start: Company
            end: Company

        ds = (
            ds.extract_document_structure(structure=StructureBySection)
            .extract_graph_entities([EntityExtractor(self.MockEntityLLM(), [Company])])
            .extract_graph_relationships([RelationshipExtractor(self.MockRelationshipLLM(), [Competes])])
            .resolve_graph_entities(resolvers=[], resolve_duplicates=True)
        )
        docs = ds.take_all()

        for doc in docs:
            if doc.data.get("EXTRACTED_NODES", False) is True:
                assert len(doc.children) == 3
                for node in doc.children:
                    if node["properties"]["name"] == "Microsoft":
                        assert len(node["relationships"]) == 2
                    if node["properties"]["name"] == "Google":
                        assert len(node["relationships"]) == 3
                    if node["properties"]["name"] == "3M":
                        assert len(node["relationships"]) == 2
