from typing import Optional
from pydantic import BaseModel
import sycamore
from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.llms.llms import LLM
from sycamore.reader import DocSetReader
from sycamore.transforms.extract_document_structure import StructureBySection
from sycamore.transforms.extract_graph import GraphMetadata, MetadataExtractor, EntityExtractor
from sycamore.transforms.extract_graph import ExtractSummaries
from collections import defaultdict

import logging

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

    class MockLLM(LLM):
        def __init__(self):
            super().__init__(model_name="mock_model")

        def generate(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None):
            pass

        async def generate_async(self, *, prompt_kwargs: dict, llm_kwargs: Optional[dict] = None):
            return """{
                "Company": [
                    {"name": "Microsoft"},
                    {"name": "Google"},
                    {"name": "3M"}
                ]
            }
            """

        def is_chat_mode(self):
            return True

    def test_metadata_extractor(self):
        context = sycamore.init()
        reader = DocSetReader(context)
        ds = reader.document(self.docs)

        metadata = [
            GraphMetadata(nodeKey="company", nodeLabel="Company", relLabel="FILED_BY"),
            GraphMetadata(nodeKey="sector", nodeLabel="Sector", relLabel="IN_SECTOR"),
            GraphMetadata(nodeKey="doctype", nodeLabel="Document Type", relLabel="IS_TYPE"),
        ]

        ds = (
            ds.extract_document_structure(structure=StructureBySection)
            .extract_graph_structure([MetadataExtractor(metadata=metadata)])
            .explode()
        )

        docs = ds.take_all()

        nested_dict = defaultdict(lambda: defaultdict(list))

        for entry in docs:
            if not {"label", "properties", "relationships"} <= entry.keys():
                continue
            label = entry["label"]
            properties = entry["properties"]
            relations = entry["relationships"]

            for value in properties.values():
                for rel in relations.values():
                    nested_dict[label][value].append(rel)

        nested_dict = {label: dict(properties) for label, properties in nested_dict.items()}

        assert len(nested_dict["Company"]["3M"]) == 1
        assert nested_dict["Company"]["3M"][0]["START_ID"] == "1"
        assert len(nested_dict["Document Type"]["10K"]) == 1
        assert len(nested_dict["Sector"]["Industrial"]) == 1

    def test_entity_extractor(self):
        context = sycamore.init()
        reader = DocSetReader(context)
        ds = reader.document(self.docs)

        class Company(BaseModel):
            name: str

        llm = self.MockLLM()
        ds = (
            ds.extract_document_structure(structure=StructureBySection)
            .extract_graph_structure([EntityExtractor(llm=llm, entities=[Company])])
            .explode()
        )
        docs = ds.take_all()

        nested_dict = defaultdict(lambda: defaultdict(list))

        for entry in docs:
            if not {"label", "properties", "relationships"} <= entry.keys():
                continue
            label = entry["label"]
            properties = entry["properties"]
            relations = entry["relationships"]

            for value in properties.values():
                for rel in relations.values():
                    nested_dict[label][value].append(rel)

        assert len(nested_dict["Company"]["Microsoft"]) == 2
        assert len(nested_dict["Company"]["Google"]) == 2
        assert len(nested_dict["Company"]["3M"]) == 2

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
