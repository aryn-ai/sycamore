from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, Optional
from sycamore.llms import OpenAI
from sycamore.plan_nodes import Node
from sycamore.transforms.map import Map
from sycamore.data import Document, MetadataDocument, HierarchicalDocument
import json
import uuid

if TYPE_CHECKING:
    from sycamore.docset import DocSet


class GraphData(ABC):
    def __init__(self):
        pass


class GraphMetadata(GraphData):
    """
    Object which handles what fields to extract metadata from and what fields to represent them as in neo4j

    Args:
        nodeKey: Key used to access document metadata in the document['properties] dictionary
        nodeLabel: The label used in neo4j the node of a piece of metadata
        relLabel: The label used in neo4j for the relationship between the document and a piece of metadata
    """

    def __init__(self, nodeKey: str, nodeLabel: str, relLabel: str):
        self.nodeKey = nodeKey
        self.nodeLabel = nodeLabel
        self.relLabel = relLabel


class GraphEntity(GraphData):
    def __init__(self, entityLabel: str, entityDescription: Optional[str] = None):
        self.label = entityLabel
        self.description = entityDescription


class GraphExtractor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def extract(self, docset: "DocSet") -> "DocSet":
        pass

    @abstractmethod
    def _extract(self, doc: "HierarchicalDocument") -> "HierarchicalDocument":
        pass

    def resolve(self, docset: "DocSet") -> "DocSet":
        """
        Aggregates 'nodes' from every document and resolves duplicate nodes
        """
        from sycamore import Execution
        from sycamore.reader import DocSetReader
        from ray.data.aggregate import AggregateFn

        execution = Execution(docset.context, docset.plan)
        dataset = execution.execute(docset.plan)

        def extract_nodes(row):
            doc = Document.deserialize(row["doc"])
            if isinstance(doc, MetadataDocument) or "nodes" not in doc["properties"]:
                return {}
            return doc["properties"]["nodes"]

        def accumulate_row(nodes, row):
            extracted = extract_nodes(row)
            for key, value in extracted.items():
                if nodes.get(key, {}) == {}:
                    nodes[key] = value
                else:
                    for rel_uuid, rel in extracted[key]["relationships"].items():
                        nodes[key]["relationships"][rel_uuid] = rel
            return nodes

        def merge(nodes1, nodes2):
            for key, value in nodes2.items():
                if nodes1.get(key, {}) == {}:
                    nodes1[key] = value
                else:
                    for rel_uuid, rel in nodes2[key]["relationships"].items():
                        nodes1[key]["relationships"][rel_uuid] = rel
            return nodes1

        def finalize(nodes):
            for value in nodes.values():
                value["doc_id"] = str(uuid.uuid4())
                for rel in value["relationships"].values():
                    rel["END_ID"] = value["doc_id"]
                    rel["END_LABEL"] = value["label"]
            return nodes

        aggregation = AggregateFn(
            init=lambda group_key: {}, accumulate_row=accumulate_row, merge=merge, finalize=finalize, name="nodes"
        )

        result = dataset.aggregate(aggregation)
        docs = dataset.take_all(None)
        docs = [Document.deserialize(d["doc"]) for d in docs]

        for doc in docs:
            if "properties" in doc:
                if "nodes" in doc["properties"]:
                    del doc["properties"]["nodes"]

        doc = HierarchicalDocument()
        for value in result["nodes"].values():
            node = HierarchicalDocument(value)
            doc.children.append(node)
        doc.data["EXTRACTED_NODES"] = True

        docs.append(doc)

        reader = DocSetReader(docset.context)
        return reader.document(docs)


class MetadataExtractor(GraphExtractor):
    """
    Extracts metadata from documents and represents them as nodes and relationship in neo4j

    Args:
        metadata: a list of GraphMetadata that is used to determine what metadata is extracted
    """

    def __init__(self, metadata: list[GraphMetadata]):
        self.metadata = metadata

    def extract(self, docset: "DocSet") -> "DocSet":
        docset.plan = ExtractFeatures(docset.plan, self)
        docset = self.resolve(docset)

        return docset

    def _extract(self, doc: HierarchicalDocument) -> HierarchicalDocument:
        """
        Extracts metadata from documents and stores them in the 'nodes' key of 'properties in each document
        """
        nodes: Dict[str, Dict[str, Any]] = {}
        for m in self.metadata:
            key = m.nodeKey
            value = str(doc["properties"].get(key))
            if value == "None":
                continue

            node: Dict[str, Any] = {
                "type": "metadata",
                "properties": {key: value},
                "label": m.nodeLabel,
                "relationships": {},
            }
            rel: Dict[str, Any] = {
                "TYPE": m.relLabel,
                "properties": {},
                "START_ID": str(doc.doc_id),
                "START_LABEL": doc.data["label"],
            }

            node["relationships"][str(uuid.uuid4())] = rel
            nodes[key + "_" + value] = node
            del doc["properties"][m.nodeKey]

        doc["properties"]["nodes"] = nodes
        return doc


class EntityExtractor(GraphExtractor):
    def __init__(self, entities: list[GraphEntity], llm: OpenAI):
        self.entities = entities
        self.llm = llm

    def extract(self, docset: "DocSet") -> "DocSet":
        docset.plan = ExtractFeatures(docset.plan, self)
        docset = self.resolve(docset)
        return docset

    def _extract(self, doc: HierarchicalDocument) -> HierarchicalDocument:
        from multiprocessing import Pool
        if "EXTRACTED_NODES" in doc.data:
            return doc

        self.llm.setAsynchronous(True)

        pool = Pool(processes=8)
        res = pool.map(self._extract_from_section,[child.data["summary"] for child in doc.children])
        pool.close()

        nodes = {}
        for i, section in enumerate(doc.children):
            for node in res[i]["entities"]:
                node = {
                    "type": "extracted",
                    "properties": {"name": node["name"]},
                    "label": node["type"],
                    "relationships": {},
                }
                rel: Dict[str, Any] = {
                    "TYPE": "CONTAINS",
                    "properties": {},
                    "START_ID": str(section.doc_id),
                    "START_LABEL": section.data["label"],
                }
                node["relationships"][str(uuid.uuid4())] = rel

                key = str(node["label"] + "_" + node["properties"]["name"])
                if key not in nodes:
                    nodes[key] = node
                else:
                    for rel_uuid, rel in node["relationships"].items():
                        nodes[key]["relationships"][rel_uuid] = rel

        doc["properties"]["nodes"] = nodes

        return doc

    def _extract_from_section(self, summary: str) -> dict:
        labels = [e.label + ": " + e.description for e in self.entities]
        res = self.llm.generate_async(
            prompt_kwargs={
                "prompt": str(GraphEntityExtractorPrompt(labels, summary)),
                "entities": labels,
                "query": summary,
            },
            llm_kwargs={"response_format": {"type": "json_object"}},
        )

        return json.loads(res)


def GraphEntityExtractorPrompt(entities, query):
    return f"""
    -Goal-
    You are a helpful information extraction system.

    You will be given a sequence of data in different formats(text, table, Section-header) in order.
    Your job is to extract entities that match the following types and descriptions.


    -Instructions-
    Entity Types and Descriptions: [{entities}]

    Identify all entities that fit one of the following types and their descriptions.
    For each of these entities extract the following information.
    - entity_name: Name of the entity, capitalized
    - entity_type: One of the following types listed above.

    Format each entity that fits one of the types and their description as a json object.
    Then, collect all json objects into a single json array named entities.

    **Format Example:**
    {{
    entities: [
    {{"name": <entity_name>, "type": <entity_type>}},
    {{"name": <entity_name>, "type": <entity_type>}},
    ...]
    }}

    -Real Data-
    ######################
    Entity_types: {entities}
    Text: {query}
    ######################
    Output:"""


class ExtractFeatures(Map):
    """
    Extracts metadata from each document
    """

    def __init__(self, child: Node, extractor: GraphExtractor, **resource_args):
        super().__init__(child, f=extractor._extract, **resource_args)
