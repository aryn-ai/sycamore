from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Awaitable, Dict, Any, Optional
from sycamore.plan_nodes import Node
from sycamore.transforms.map import Map
from sycamore.data import Document, MetadataDocument, HierarchicalDocument
from sycamore.llms import LLM
import json
import uuid
import logging

if TYPE_CHECKING:
    from sycamore.docset import DocSet

logger = logging.getLogger(__name__)


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
    """
    Object which contains the label and description of an entity type that is to be extracted from unstructured text

    Args:
        entityLabel: Label of entity(i.e. Person, Company, Country)
        entityDescription: Description of what the entity is
    """

    def __init__(self, entityLabel: str, entityDescription: str):
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

        reader = DocSetReader(docset.context)

        # Get list[Document] representation of docset, trigger execute with take_all()
        execution = Execution(docset.context, docset.plan)
        dataset = execution.execute(docset.plan)
        docs = dataset.take_all()
        docs = [Document.deserialize(d["doc"]) for d in docs]

        # Update docset and dataset to version after execute
        docset = reader.document(docs)
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

        return reader.document(docs)


class MetadataExtractor(GraphExtractor):
    """
    Extracts metadata from documents and represents them as nodes and relationship in neo4j

    Args:
        metadata: A list of GraphMetadata that is used to determine what metadata is extracted
    """

    def __init__(self, metadata: list[GraphMetadata]):
        self.metadata = metadata

    def extract(self, docset: "DocSet") -> "DocSet":
        """
        Extracts metadata from documents and creates an additional document in the docset that stores those nodes
        """
        docset.plan = ExtractFeatures(docset.plan, self)
        docset = self.resolve(docset)

        return docset

    def _extract(self, doc: HierarchicalDocument) -> HierarchicalDocument:
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
    """
    Extracts entities chosen by the user by using LLMs

    Args:
        entities: A list of GraphEntity that determines what entities are extracted
        llm: The LLM that is used to extract the entities
    """

    def __init__(self, entities: list[GraphEntity], llm: LLM):
        self.entities = entities
        self.llm = llm

    def extract(self, docset: "DocSet") -> "DocSet":
        """
        Extracts entities from documents then creates a document in the docset where they are stored as nodes
        """
        docset.plan = ExtractSummaries(docset.plan)
        docset.plan = ExtractFeatures(docset.plan, self)
        docset = self.resolve(docset)
        return docset

    def _extract(self, doc: HierarchicalDocument) -> HierarchicalDocument:
        import asyncio

        if "EXTRACTED_NODES" in doc.data or not isinstance(doc, HierarchicalDocument):
            return doc

        async def gather_api_calls():
            labels = [e.label + ": " + e.description for e in self.entities]
            tasks = [self._extract_from_section(labels, child.data["summary"]) for child in doc.children]
            res = await asyncio.gather(*tasks)
            return res

        res = asyncio.run(gather_api_calls())

        nodes = {}
        for i, section in enumerate(doc.children):
            try:
                res[i] = json.loads(res[i])
            except json.JSONDecodeError:
                logger.warn("LLM Output failed to be decoded to JSON")
                logger.warn("Input: " + section.data["summary"])
                logger.warn("Output: " + res[i])
                res[i] = {"entities": []}

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

    async def _extract_from_section(self, labels, summary: str) -> Awaitable[str]:
        return await self.llm.generate_async(
            prompt_kwargs={"prompt": str(GraphEntityExtractorPrompt(labels, summary))},
            llm_kwargs={"response_format": {"type": "json_object"}},
        )


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
    {{"name": <entity_name_1>, "type": <entity_type_1>}},
    {{"name": <entity_name_2>, "type": <entity_type_2>}},
    ...]
    }}

    -Real Data-
    ######################
    Entity_types: {entities}
    Text: {query}
    ######################
    Output:"""


class ExtractDocumentStructure(Map):
    """
    Extracts the structure of the document organizing document elements by their
    respective section headers.
    """

    def __init__(self, child: Node, **resource_args):
        super().__init__(child, f=ExtractDocumentStructure.structure_by_section, **resource_args)

    @staticmethod
    def structure_by_section(doc: Document) -> HierarchicalDocument:
        import uuid

        doc = HierarchicalDocument(doc.data)
        # if the first element is not a section header, insert generic placeholder
        if len(doc.children) > 0 and doc.children[0]["type"] != "Section-header":
            initial_page = HierarchicalDocument(
                {
                    "type": "Section-header",
                    "bbox": (0, 0, 0, 0),
                    "properties": {"score": 1, "page_number": 1},
                    "text_representation": "Front Page",
                    "binary_representation": b"Front Page",
                }
            )
            doc.children.insert(0, initial_page)  # O(n) insert :( we should use deque for everything

        doc.data["relationships"] = doc.get("relationships", {})
        doc.data["label"] = doc.get("label", "DOCUMENT")

        sections = []

        section: Optional[HierarchicalDocument] = None
        element: Optional[HierarchicalDocument] = None
        for child in doc.children:
            child.data["relationships"] = child.get("relationships", {})
            if child.type == "Section-header" and child.data.get("text_representation"):
                if section is not None:
                    next = {
                        "TYPE": "NEXT",
                        "properties": {},
                        "START_ID": section.doc_id,
                        "START_LABEL": "SECTION",
                        "END_ID": child.doc_id,
                        "END_LABEL": "SECTION",
                    }
                    child.data["relationships"][str(uuid.uuid4())] = next
                    element = None
                rel = {
                    "TYPE": "SECTION_OF",
                    "properties": {},
                    "START_ID": child.doc_id,
                    "START_LABEL": "SECTION",
                    "END_ID": doc.doc_id,
                    "END_LABEL": "DOCUMENT",
                }
                child.data["relationships"][str(uuid.uuid4())] = rel
                child.data["label"] = "SECTION"
                section = child
                sections.append(section)
            else:
                assert section is not None
                if element is not None:
                    next = {
                        "TYPE": "NEXT",
                        "properties": {},
                        "START_ID": element.doc_id,
                        "START_LABEL": "ELEMENT",
                        "END_ID": child.doc_id,
                        "END_LABEL": "ELEMENT",
                    }
                    child.data["relationships"][str(uuid.uuid4())] = next
                rel = {
                    "TYPE": "PART_OF",
                    "properties": {},
                    "START_ID": child.doc_id,
                    "START_LABEL": "ELEMENT",
                    "END_ID": section.doc_id,
                    "END_LABEL": "SECTION",
                }
                child.data["relationships"][str(uuid.uuid4())] = rel
                child.data["label"] = "ELEMENT"
                element = child
                section.data["children"].append(element)

        doc.children = sections
        return doc


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


class ExtractFeatures(Map):
    """
    Extracts features determined by a specific extractor from each document
    """

    def __init__(self, child: Node, extractor: GraphExtractor, **resource_args):
        super().__init__(child, f=extractor._extract, **resource_args)
