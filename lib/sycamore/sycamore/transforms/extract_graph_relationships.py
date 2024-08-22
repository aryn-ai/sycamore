from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
import hashlib
from typing import Awaitable, Dict, Any, List
from sycamore.plan_nodes import Node
from sycamore.transforms.map import Map
from sycamore.data import HierarchicalDocument
from sycamore.llms import LLM
from pydantic import BaseModel, create_model
import asyncio

import json
import uuid
import logging

logger = logging.getLogger(__name__)


class GraphRelationshipExtractor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def extract(self, doc: "HierarchicalDocument") -> "HierarchicalDocument":
        pass


class RelationshipExtractor(GraphRelationshipExtractor):
    """
    Extracts relationships between entities found in each child of a document.

    Args:
        llm: OpenAI model that is compatable with structured outputs(gpt-4o-mini)
        relationships: list of entities in the form of pydantic schemas to be extracted

    """

    def __init__(self, llm: LLM, relationships: list[BaseModel] = []):
        self.relationships = self._serialize_relationships(relationships)
        self.llm = llm

    def extract(self, doc: HierarchicalDocument) -> HierarchicalDocument:
        async def gather_api_calls():
            tasks = [self._generate_relationships(child) for child in doc.children]
            res = await asyncio.gather(*tasks)
            return res

        res = asyncio.run(gather_api_calls())

        for i, section in enumerate(doc.children):
            for label, relations in res[i].items():
                for relation in relations:
                    start_hash = hashlib.sha256(json.dumps(relation["start"]).encode()).hexdigest()
                    end_hash = hashlib.sha256(json.dumps(relation["end"]).encode()).hexdigest()

                    start_exists = section["properties"]["nodes"][relation["start_label"]].get(start_hash, None)
                    end_exists = section["properties"]["nodes"][relation["end_label"]].get(end_hash, None)
                    if not (start_exists and end_exists):
                        logger.warn(
                            f"""
                            Entities referenced by relationship does not exist:
                            Start: {relation["start"]}
                            End: {relation["end"]}
                            """
                        )
                        continue

                    rel: Dict[str, Any] = {
                        "TYPE": label,
                        "properties": {},
                        "START_HASH": start_hash,
                        "START_LABEL": relation["start_label"],
                    }

                    for key, value in relation.items():
                        if key not in ["start", "end", "start_label", "end_label"]:
                            rel["properties"][key] = value

                    section["properties"]["nodes"][relation["end_label"]][end_hash]["relationships"][
                        str(uuid.uuid4())
                    ] = rel
        return doc

    def _serialize_relationships(self, entities):
        from sycamore.utils.pickle_pydantic import safe_cloudpickle

        serialized = []
        for entity in entities:
            serialized.append(safe_cloudpickle(entity))
        return serialized

    def _deserialize_relationships(self):
        from sycamore.utils.pickle_pydantic import safe_cloudunpickle

        deserialized = []
        for entity in self.relationships:
            deserialized.append(safe_cloudunpickle(entity))

        return deserialized

    async def _generate_relationships(self, section: HierarchicalDocument) -> dict:
        relations = self._deserialize_relationships()
        parsed_relations = []
        parsed_metadata = dict()
        parsed_nodes: dict[str, set] = defaultdict(lambda: set())
        for relation in relations:
            start_label = relation.__annotations__["start"].__name__
            end_label = relation.__annotations__["end"].__name__

            start_nodes = [
                json.dumps(node["raw_entity"]) for node in section["properties"]["nodes"].get(start_label, {}).values()
            ]
            end_nodes = [
                json.dumps(node["raw_entity"]) for node in section["properties"]["nodes"].get(end_label, {}).values()
            ]

            relation.__annotations__["start"] = Enum(start_label, {entity: entity for entity in start_nodes})
            relation.__annotations__["end"] = Enum(end_label, {entity: entity for entity in end_nodes})

            if start_nodes and end_nodes:
                parsed_relations.append(relation)
                parsed_metadata[relation.__name__] = {"start_label": start_label, "end_label": end_label}
                parsed_nodes[start_label] |= set(start_nodes)
                parsed_nodes[end_label] |= set(end_nodes)

        if not parsed_relations:
            return {}

        # Use mypy ignore type since pydantic has bad interaction with mypy with creating class from a variable class
        fields = {relation.__name__: (List[relation], ...) for relation in parsed_relations}  # type: ignore
        relationships_model = create_model("relationships", __base__=BaseModel, **fields)  # type: ignore

        entities = ""
        for key, nodes in parsed_nodes.items():
            entities += f"{key}:\n"
            for node in nodes:
                entities += f"{node}\n"

        llm_kwargs = {"response_format": relationships_model}
        res = await self.llm.generate_async(
            prompt_kwargs={"prompt": str(GraphRelationshipExtractorPrompt(section.data["summary"], entities))},
            llm_kwargs=llm_kwargs,
        )

        async def _process_llm_output(res: str, parsed_metadata: dict, summary: str):
            try:
                parsed_res = json.loads(res)
            except json.JSONDecodeError:
                logger.warn("LLM Output failed to be decoded to JSON")
                logger.warn("Input: " + summary)
                logger.warn("Output: " + res)
                return {}

            for label, relations in parsed_res.items():
                for relation in relations:
                    relation["start_label"] = parsed_metadata[label]["start_label"]
                    relation["end_label"] = parsed_metadata[label]["end_label"]

            return parsed_res

        return await _process_llm_output(res, parsed_metadata, section.data["summary"])


def GraphRelationshipExtractorPrompt(query, entities):
    return f"""
    -Goal-
    You are a helpful information extraction system.

    You will be given a sequence of data in different formats(text, table, Section-header) in order.
    Your job is to extract relationships that map between entities that have already been extracted from this text.

    -Real Data-
    ######################
    Entities: {entities}
    Text: {query}
    ######################
    Output:"""


class ExtractRelationships(Map):
    """
    Extracts features determined by a specific extractor from each document
    """

    def __init__(self, child: Node, extractor: RelationshipExtractor, **resource_args):
        super().__init__(child, f=extractor.extract, **resource_args)
