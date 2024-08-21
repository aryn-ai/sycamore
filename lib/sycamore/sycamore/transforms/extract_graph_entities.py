from abc import ABC, abstractmethod
from collections import defaultdict
import hashlib
from typing import Awaitable, Dict, Any, List, Optional
from sycamore.plan_nodes import Node
from sycamore.transforms.map import Map
from sycamore.data import HierarchicalDocument
from sycamore.llms import LLM
from pydantic import BaseModel, create_model

import json
import uuid
import logging

logger = logging.getLogger(__name__)


class GraphEntityExtractor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def extract(self, docset: "HierarchicalDocument") -> "HierarchicalDocument":
        pass


class EntityExtractor(GraphEntityExtractor):
    """
    Extracts entity schemas specified by a user from unstructured text from documents

    Args:
        llm: The LLM that is used to extract the entities
        entities: A list of pydantic models that determines what entity schemas are extracted

    """

    def __init__(self, llm: LLM, entities: Optional[list[BaseModel]] = []):
        self.llm = llm
        self.entities = self._serialize_entities(entities)

    def extract(self, doc: HierarchicalDocument) -> HierarchicalDocument:
        import asyncio

        if "EXTRACTED_NODES" in doc.data:
            return doc

        async def gather_api_calls():
            tasks = [self._extract_from_section(child.data["summary"]) for child in doc.children]
            res = await asyncio.gather(*tasks)
            return res

        res = asyncio.run(gather_api_calls())

        for i, section in enumerate(doc.children):
            nodes: defaultdict[dict, Any] = defaultdict(dict)
            try:
                res[i] = json.loads(res[i])
            except json.JSONDecodeError:
                logger.warn("LLM Output failed to be decoded to JSON")
                logger.warn("Input: " + section.data["summary"])
                logger.warn("Output: " + res[i])
                res[i] = {"entities": []}

            for label, entities in res[i].items():
                for entity in entities:
                    hash = hashlib.sha256(json.dumps(entity).encode()).hexdigest()
                    if hash not in nodes[label]:
                        node = {
                            "type": "extracted",
                            "properties": {},
                            "label": label,
                            "relationships": {},
                            "raw_entity": entity,
                        }
                        for key, value in entity.items():
                            node["properties"][key] = value
                        nodes[label][hash] = node
                    rel: Dict[str, Any] = {
                        "TYPE": "CONTAINS",
                        "properties": {},
                        "START_ID": str(section.doc_id),
                        "START_LABEL": section.data["label"],
                    }
                    nodes[label][hash]["relationships"][str(uuid.uuid4())] = rel
            section["properties"]["nodes"] = nodes
        return doc

    def _serialize_entities(self, entities):
        from sycamore.utils.pickle_pydantic import safe_cloudpickle

        serialized = []
        for entity in entities:
            serialized.append(safe_cloudpickle(entity))
        return serialized

    def _deserialize_entities(self):
        from sycamore.utils.pickle_pydantic import safe_cloudunpickle

        deserialized = []
        for entity in self.entities:
            deserialized.append(safe_cloudunpickle(entity))

        fields = {entity.__name__: (List[entity], ...) for entity in deserialized}
        return create_model("entities", __base__=BaseModel, **fields)

    async def _extract_from_section(self, summary: str) -> Awaitable[str]:
        llm_kwargs = {"response_format": self._deserialize_entities()}

        return await self.llm.generate_async(
            prompt_kwargs={"prompt": str(GraphEntityExtractorPrompt(summary))}, llm_kwargs=llm_kwargs
        )


def GraphEntityExtractorPrompt(query):
    return f"""
    -Goal-
    You are a helpful information extraction system.

    You will be given a sequence of data in different formats(text, table, Section-header) in order.
    Your job is to extract entities that match the entity schemas provided.

    -Real Data-
    ######################
    Text: {query}
    ######################
    Output:"""


class ExtractSummaries(Map):
    """
    Extracts summaries from child documents to be used for entity extraction. This function
    generates summaries for sections within documents which are used during entity extraction.
    """

    def __init__(self, child: Node, **resource_args):
        super().__init__(child, f=ExtractSummaries.summarize_sections, **resource_args)

    @staticmethod
    def summarize_sections(doc: HierarchicalDocument) -> HierarchicalDocument:
        if "EXTRACTED_NODES" in doc.data:
            return doc
        for section in doc.children:
            assert section.text_representation is not None
            summary = f"-----SECTION TITLE: {section.text_representation.strip()}-----\n"
            for element in section.children:
                if element.type == "table":
                    element.text_representation = element.data["table"].to_csv()
                assert element.type is not None
                assert element.text_representation is not None
                summary += f"""---Element Type: {element.type.strip()}---\n{element.text_representation.strip()}\n"""
            section.data["summary"] = summary
        return doc


class ExtractEntities(Map):
    """
    Extracts features determined by a specific extractor from each document
    """

    def __init__(self, child: Node, extractor: GraphEntityExtractor, **resource_args):
        super().__init__(child, f=extractor.extract, **resource_args)
