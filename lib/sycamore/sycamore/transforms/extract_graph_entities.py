from abc import ABC, abstractmethod
from collections import defaultdict
import hashlib
from typing import Dict, Any, List, Optional
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
    Extracts entity schemas specified by a user from unstructured text from document children.

    Args:
        llm: The LLM that is used to extract the entities
        entities: A list of pydantic models that determines what entity schemas are extracted

    """

    def __init__(self, llm: LLM, entities: Optional[list[BaseModel]] = [], split_calls: bool = False):
        self.llm = llm
        self.entities = self._serialize_entities(entities)
        self.split_calls = split_calls

    def extract(self, doc: HierarchicalDocument) -> HierarchicalDocument:
        import asyncio

        async def gather_api_calls():
            tasks = [self._extract_from_section(child.data["summary"]) for child in doc.children]
            res = await asyncio.gather(*tasks)
            return res

        res = asyncio.run(gather_api_calls())

        for i, section in enumerate(doc.children):
            nodes: defaultdict[dict, Any] = defaultdict(dict)
            try:
                output_dict: dict[str, Any] = {}
                for output in res[i]:
                    output_dict |= json.loads(output)
                res[i] = output_dict
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

        return deserialized

    async def _extract_from_section(self, summary: str) -> list[str]:
        # (List[relation], ...) is weird notation required by pydantic, sorry - Ritam
        # Pydantic has weird interaction with mypy, using type ignore!
        # https://docs.pydantic.dev/latest/concepts/models/#required-fields
        deserialized_entities = self._deserialize_entities()
        fields = {entity.__name__: (List[entity], ...) for entity in deserialized_entities}  # type: ignore
        models = []
        if self.split_calls:
            for field_name, field_value in fields.items():
                models.append(create_model("entities", __base__=BaseModel, **{field_name: field_value}))  # type: ignore
        else:
            models.append(create_model("entities", __base__=BaseModel, **fields))  # type: ignore

        outputs = []
        for model in models:
            try:
                llm_kwargs = {"response_format": model}
                outputs.append(
                    await self.llm.generate_async(
                        prompt_kwargs={"prompt": str(GraphEntityExtractorPrompt(summary))}, llm_kwargs=llm_kwargs
                    )
                )
            except Exception as e:
                logger.warn(f"OPENAI CALL FAILED: {e}")
                outputs.append("{}")
        return outputs


def GraphEntityExtractorPrompt(query):
    return f"""
    -Instructions-
    You are a information extraction system.

    You will be given a sequence of data in different formats(text, table, Section-header) in order.
    Your job is to extract entities from the text input that match the entity schemas provided. Each entity
    and property extracted should directly reference part of the text input provided.

    -Real Data-
    ######################
    Text: {query}
    ######################
    Output:"""


class ExtractEntities(Map):
    """
    Extracts features determined by a specific extractor from each document
    """

    def __init__(self, child: Node, extractor: GraphEntityExtractor, **resource_args):
        super().__init__(child, f=extractor.extract, **resource_args)
