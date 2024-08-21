import hashlib
import json
from typing import Optional
from pydantic import BaseModel
import sycamore
from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.llms.llms import LLM
from sycamore.reader import DocSetReader
from sycamore.transforms.extract_document_structure import StructureBySection
from sycamore.transforms.extract_graph_entities import EntityExtractor

import logging

from sycamore.transforms.extract_graph_relationships import RelationshipExtractor

logger = logging.getLogger(__name__)


class TestGraphExtractor:
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

    def test_relationship_extractor(self):
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
            .extract_graph_entities([EntityExtractor(llm=self.MockEntityLLM(), entities=[Company])])
            .extract_graph_relationships([RelationshipExtractor(llm=self.MockRelationshipLLM(), relationships=[Competes])])
        )
        docs = ds.take_all()

        for doc in docs:
            for section in doc.children:
                for node in section["properties"]["nodes"]["Company"].values():
                    if node["properties"]["name"] == "Google":
                        start_hash = hashlib.sha256(json.dumps({"name": "Microsoft"}).encode()).hexdigest()
                        hashes = set(relation.get("START_HASH", None) for relation in node["relationships"].values())
                        assert start_hash in hashes
