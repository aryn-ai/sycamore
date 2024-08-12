from typing import Optional
import sycamore
from sycamore.data.document import Document
from sycamore.data.element import Element
from sycamore.llms.llms import LLM
from sycamore.reader import DocSetReader
from sycamore.transforms.extract_graph import GraphMetadata, MetadataExtractor, GraphEntity, EntityExtractor
from sycamore.data import HierarchicalDocument
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)


class TestGraphExtractor:
    metadata_docs = [
        HierarchicalDocument(
            {
                "doc_id": "1",
                "label": "Document",
                "type": "pdf",
                "relationships": {},
                "properties": {"company": "3M", "sector": "Industrial", "doctype": "10K"},
                "children": [],
            }
        ),
        HierarchicalDocument(
            {
                "doc_id": "2",
                "label": "Document",
                "type": "pdf",
                "relationships": {},
                "properties": {"company": "FedEx", "sector": "Industrial", "doctype": "10K"},
                "children": [],
            }
        ),
        HierarchicalDocument(
            {
                "doc_id": "3",
                "label": "Document",
                "type": "pdf",
                "relationships": {},
                "properties": {"company": "Apple", "sector": "Technology", "doctype": "10K"},
                "children": [],
            }
        ),
    ]

    entity_docs = [
        Document(
            {
                "doc_id": "1",
                "type": "pdf",
                "properties": {"company": "3M", "sector": "Industrial", "doctype": "10K"},
                "elements": [
                    Element(
                        {
                            "type": "Section-header",
                            "text_representation": "header",
                            "properties": {},
                        }
                    ),
                    Element(
                        {
                            "type": "text",
                            "text_representation": "i'm text",
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
                "entities": [
                    {
                        "name": "Microsoft",
                        "type": "Company"
                    },
                    {
                        "name": "Google",
                        "type": "Company"
                    },
                    {
                        "name": "3M",
                        "type": "Company"
                    }
                ]
            }
            """

        def is_chat_mode(self):
            return True

    def test_metadata_extractor(self):
        context = sycamore.init()
        reader = DocSetReader(context)
        ds = reader.document(self.metadata_docs)

        metadata = [
            GraphMetadata(nodeKey="company", nodeLabel="Company", relLabel="FILED_BY"),
            GraphMetadata(nodeKey="sector", nodeLabel="Sector", relLabel="IN_SECTOR"),
            GraphMetadata(nodeKey="doctype", nodeLabel="Document Type", relLabel="IS_TYPE"),
        ]

        ds = ds.extract_graph_structure([MetadataExtractor(metadata=metadata)]).explode()
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
        assert len(nested_dict["Company"]["Apple"]) == 1
        assert nested_dict["Company"]["Apple"][0]["START_ID"] == "3"
        assert len(nested_dict["Company"]["FedEx"]) == 1
        assert nested_dict["Company"]["FedEx"][0]["START_ID"] == "2"
        assert len(nested_dict["Document Type"]["10K"]) == 3
        assert len(nested_dict["Sector"]["Industrial"]) == 2
        assert len(nested_dict["Sector"]["Technology"]) == 1

    def test_entity_extractor(self):
        context = sycamore.init()
        reader = DocSetReader(context)
        ds = reader.document(self.entity_docs)

        entities = [GraphEntity(entityLabel="Company", entityDescription="...")]
        llm = self.MockLLM()

        ds = ds.extract_graph_structure([EntityExtractor(entities=entities, llm=llm)]).explode()
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


        assert len(nested_dict["Company"]["Microsoft"]) == 1
        assert len(nested_dict["Company"]["Google"]) == 1
        assert len(nested_dict["Company"]["3M"]) == 1
