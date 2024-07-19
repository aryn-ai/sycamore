from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from sycamore.plan_nodes import Node
from sycamore.transforms.map import Map
from sycamore.data import Document, MetadataDocument
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


class GraphExtractor(ABC):
    def __init__(self, entity_name: str):
        self._entity_name = entity_name

    @abstractmethod
    def extract(self, document: "DocSet") -> "DocSet":
        pass


class MetadataExtractor(GraphExtractor):
    """
    Extracts metadata from documents and represents them as nodes and relationship in neo4j

    Args:
        metadata: a list of GraphMetadata that is used to determine what metadata is extracted
    """

    def __init__(self, metadata: list[GraphMetadata]):
        self.metadata = metadata

    def extract(self, docset: "DocSet") -> "DocSet":
        docset.plan = ExtractMetadata(docset.plan, self)
        docset = self.resolve_metadata(docset)

        return docset

    def extract_metadata(self, doc: Document) -> Document:
        """
        Extracts metadata from documents and stores them in the 'nodes' key of 'properties in each document
        """
        nodes = {}
        for m in self.metadata:
            if m.nodeKey not in doc["properties"]:
                continue

            key = str(m.nodeKey)
            value = str(doc["properties"][m.nodeKey])
            node = {"label": str(m.nodeLabel), "type": "metadata", "properties": {key: value}, "relationships": {}}
            nodes[key + "_" + value] = node

            rel = {
                "TYPE": str(m.relLabel),
                "properties": {},
                "START_ID": str(doc.doc_id),
                "START_LABEL": doc.data["label"],
            }
            nodes[key + "_" + value]["relationships"][str(uuid.uuid4())] = rel
            del doc["properties"][m.nodeKey]

        doc["properties"]["nodes"] = nodes
        return doc

    def resolve_metadata(self, docset: "DocSet") -> "DocSet":
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

        for value in result["nodes"].values():
            docs[0]["elements"].append(value)

        reader = DocSetReader(docset.context)
        return reader.document(docs)


class ExtractMetadata(Map):
    """
    Extracts metadata from each document
    """

    def __init__(self, child: Node, extractor: MetadataExtractor, **resource_args):
        super().__init__(child, f=extractor.extract_metadata, **resource_args)
